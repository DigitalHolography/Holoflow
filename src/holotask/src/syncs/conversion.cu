// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "holotask/syncs/conversion.hh"

#include <cuComplex.h>
#include <cub/cub.cuh>

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holotask::syncs {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const ConversionSettings::Target &t) {
  static const std::map<ConversionSettings::Target, std::string> target_to_string{
      {ConversionSettings::Target::U8, "U8"},
      {ConversionSettings::Target::U16, "U16"},
      {ConversionSettings::Target::F32, "F32"},
      {ConversionSettings::Target::CF32, "CF32"},
  };

  HOLOVIBES_CHECK(target_to_string.contains(t), "Unknown target type for conversion");
  j = target_to_string.at(t);
}

void from_json(const nlohmann::json &j, ConversionSettings::Target &t) {
  static const std::map<std::string, ConversionSettings::Target> string_to_target{
      {"U8", ConversionSettings::Target::U8},
      {"U16", ConversionSettings::Target::U16},
      {"F32", ConversionSettings::Target::F32},
      {"CF32", ConversionSettings::Target::CF32},
  };

  const std::string s = j.get<std::string>();
  if (!string_to_target.contains(s)) {
    throw std::invalid_argument("Unknown target type for conversion: " + s);
  }

  t = string_to_target.at(s);
}

void to_json(nlohmann::json &j, const ConversionSettings::Strategy &s) {
  static const std::map<ConversionSettings::Strategy, std::string> strategy_to_string{
      {ConversionSettings::Strategy::Real, "Real"},
      {ConversionSettings::Strategy::Scaled, "Scaled"},
      {ConversionSettings::Strategy::Modulus, "Modulus"},
      {ConversionSettings::Strategy::Argument, "Argument"},
  };

  HOLOVIBES_CHECK(strategy_to_string.contains(s), "Unknown strategy type for conversion");
  j = strategy_to_string.at(s);
}

void from_json(const nlohmann::json &j, ConversionSettings::Strategy &s) {
  static const std::map<std::string, ConversionSettings::Strategy> string_to_strategy{
      {"Real", ConversionSettings::Strategy::Real},
      {"Scaled", ConversionSettings::Strategy::Scaled},
      {"Modulus", ConversionSettings::Strategy::Modulus},
      {"Argument", ConversionSettings::Strategy::Argument},
  };

  const std::string str = j.get<std::string>();
  if (!string_to_strategy.contains(str)) {
    throw std::invalid_argument("Unknown strategy type for conversion: " + str);
  }

  s = string_to_strategy.at(str);
}

void to_json(nlohmann::json &j, const ConversionSettings &s) {
  j = nlohmann::json{{"target", s.target}, {"strategy", s.strategy}};
}

void from_json(const nlohmann::json &j, ConversionSettings &s) {
  j.at("target").get_to(s.target);
  j.at("strategy").get_to(s.strategy);
}

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

using holoflow::core::DType;
using Target   = ConversionSettings::Target;
using Strategy = ConversionSettings::Strategy;
using Cfg      = std::tuple<DType, Target, Strategy>;

void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[ConversionFactory::infer] error: {}", msg);
    throw std::invalid_argument("ConversionFactory inference error: " + msg);
  }
}

bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  if (desc.shape.size() != desc.strides.size()) {
    return false;
  }

  size_t expected = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    if (desc.strides[i] != expected) {
      return false;
    }
    expected *= desc.shape[i];
  }
  return true;
}

const std::map<Cfg, DType> &cfg_to_dtype() {
  static const std::map<Cfg, DType> map{
      {{DType::U8, Target::F32, Strategy::Real}, DType::F32},
      {{DType::U8, Target::CF32, Strategy::Real}, DType::CF32},
      {{DType::U16, Target::CF32, Strategy::Real}, DType::CF32},
      {{DType::F32, Target::U8, Strategy::Scaled}, DType::U8},
      {{DType::F32, Target::U16, Strategy::Scaled}, DType::U16},
      {{DType::F32, Target::CF32, Strategy::Real}, DType::CF32},
      {{DType::CF32, Target::F32, Strategy::Modulus}, DType::F32},
      {{DType::CF32, Target::F32, Strategy::Argument}, DType::F32},
  };
  return map;
}

// -------------------------------------------------------------------------------------------------
// Device kernels
// -------------------------------------------------------------------------------------------------

__global__ void u8_f32_real_kernel(const uint8_t *idata, float *odata, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    odata[idx] = static_cast<float>(idata[idx]);
  }
}

__global__ void u8_cf32_real_kernel(const uint8_t *idata, cuFloatComplex *odata, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    odata[idx] = make_cuFloatComplex(static_cast<float>(idata[idx]), 0.0f);
  }
}

__global__ void u16_cf32_real_kernel(const uint16_t *idata, cuFloatComplex *odata, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    odata[idx] = make_cuFloatComplex(static_cast<float>(idata[idx]), 0.0f);
  }
}

__global__ void f32_u8_scaled_kernel(const float *idata, uint8_t *odata, int size, float *min_val,
                                     float *max_val) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float scale = 255.f / (*max_val - *min_val);
    int   val   = static_cast<int>((idata[idx] - *min_val) * scale + 0.5f);
    odata[idx]  = static_cast<uint8_t>(fminf(fmaxf(val, 0), 255));
  }
}

__global__ void f32_u16_scaled_kernel(const float *idata, uint16_t *odata, int size, float *min_val,
                                      float *max_val) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float scale = 65535.f / (*max_val - *min_val);
    int   val   = static_cast<int>((idata[idx] - *min_val) * scale + 0.5f);
    odata[idx]  = static_cast<uint16_t>(fminf(fmaxf(val, 0), 65535));
  }
}

__global__ void f32_cf32_real_kernel(const float *idata, cuFloatComplex *odata, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    odata[idx] = make_cuFloatComplex(idata[idx], 0.0f);
  }
}

__global__ void cf32_f32_modulus_kernel(const cuFloatComplex *idata, float *odata, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    odata[idx] = hypotf(idata[idx].x, idata[idx].y);
  }
}

__global__ void cf32_f32_argument_kernel(const cuFloatComplex *idata, float *odata, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    odata[idx] = atan2f(idata[idx].y, idata[idx].x);
  }
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Conversion task implementation
// -------------------------------------------------------------------------------------------------

class Conversion : public holoflow::core::ISyncTask {
public:
  Conversion(ConversionSettings settings, holoflow::core::TDesc idesc,
             size_t min_temp_storage_bytes, DevPtr<uint8_t> &&d_min_temp_storage,
             DevPtr<std::byte> &&d_min, size_t max_temp_storage_bytes,
             DevPtr<uint8_t> &&d_max_temp_storage, DevPtr<std::byte> &&d_max, cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)),
        min_temp_storage_bytes_(min_temp_storage_bytes),
        d_min_temp_storage_(std::move(d_min_temp_storage)), d_min_(std::move(d_min)),
        max_temp_storage_bytes_(max_temp_storage_bytes),
        d_max_temp_storage_(std::move(d_max_temp_storage)), d_max_(std::move(d_max)),
        stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto in  = ctx.inputs[0];
    auto out = ctx.outputs[0];

    using LaunchFn = void (Conversion::*)(holoflow::core::TView, holoflow::core::TView);
    static const std::map<Cfg, LaunchFn> launch_map{
        {{DType::U8, Target::F32, Strategy::Real}, &Conversion::launch_u8_f32_real},
        {{DType::U8, Target::CF32, Strategy::Real}, &Conversion::launch_u8_cf32_real},
        {{DType::U16, Target::CF32, Strategy::Real}, &Conversion::launch_u16_cf32_real},
        {{DType::F32, Target::U8, Strategy::Scaled}, &Conversion::launch_f32_u8_scaled},
        {{DType::F32, Target::U16, Strategy::Scaled}, &Conversion::launch_f32_u16_scaled},
        {{DType::F32, Target::CF32, Strategy::Real}, &Conversion::launch_f32_cf32_real},
        {{DType::CF32, Target::F32, Strategy::Modulus}, &Conversion::launch_cf32_f32_modulus},
        {{DType::CF32, Target::F32, Strategy::Argument}, &Conversion::launch_cf32_f32_argument},
    };

    const Cfg cfg{in.desc.dtype, settings_.target, settings_.strategy};
    HOLOVIBES_CHECK(launch_map.contains(cfg), "Unsupported conversion configuration");
    const LaunchFn launch_fn = launch_map.at(cfg);
    (this->*launch_fn)(in, out);
    return holoflow::core::OpResult::Ok;
  }

  void update_stream(cudaStream_t stream) { stream_ = stream; }

  const ConversionSettings    &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }

private:
  void launch_u8_f32_real(holoflow::core::TView in, holoflow::core::TView out) {
    auto *idata = reinterpret_cast<const uint8_t *>(in.data());
    auto *odata = reinterpret_cast<float *>(out.data());
    int   size  = static_cast<int>(in.desc.num_elements());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    u8_f32_real_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size);
    CUDA_CHECK(cudaGetLastError());
  }

  void launch_u8_cf32_real(holoflow::core::TView in, holoflow::core::TView out) {
    auto *idata = reinterpret_cast<const uint8_t *>(in.data());
    auto *odata = reinterpret_cast<cuFloatComplex *>(out.data());
    int   size  = static_cast<int>(in.desc.num_elements());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    u8_cf32_real_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size);
    CUDA_CHECK(cudaGetLastError());
  }

  void launch_u16_cf32_real(holoflow::core::TView in, holoflow::core::TView out) {
    auto *idata = reinterpret_cast<const uint16_t *>(in.data());
    auto *odata = reinterpret_cast<cuFloatComplex *>(out.data());
    int   size  = static_cast<int>(in.desc.num_elements());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    u16_cf32_real_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size);
    CUDA_CHECK(cudaGetLastError());
  }

  void launch_f32_u8_scaled(holoflow::core::TView in, holoflow::core::TView out) {
    auto *idata = reinterpret_cast<float *>(const_cast<std::byte *>(in.data()));
    auto *odata = reinterpret_cast<uint8_t *>(out.data());
    auto  size  = in.desc.num_elements();

    size_t   min_storage_bytes = min_temp_storage_bytes_;
    size_t   max_storage_bytes = max_temp_storage_bytes_;
    uint8_t *d_min_storage     = d_min_temp_storage_.get();
    uint8_t *d_max_storage     = d_max_temp_storage_.get();
    float   *d_min             = reinterpret_cast<float *>(d_min_.get());
    float   *d_max             = reinterpret_cast<float *>(d_max_.get());
    CUDA_CHECK(
        cub::DeviceReduce::Min(d_min_storage, min_storage_bytes, idata, d_min, size, stream_));
    CUDA_CHECK(
        cub::DeviceReduce::Max(d_max_storage, max_storage_bytes, idata, d_max, size, stream_));

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    f32_u8_scaled_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size, d_min, d_max);
    CUDA_CHECK(cudaGetLastError());
  }

  void launch_f32_u16_scaled(holoflow::core::TView in, holoflow::core::TView out) {
    auto *idata = reinterpret_cast<const float *>(in.data());
    auto *odata = reinterpret_cast<uint16_t *>(out.data());
    int   size  = static_cast<int>(in.desc.num_elements());

    size_t   min_storage_bytes = min_temp_storage_bytes_;
    size_t   max_storage_bytes = max_temp_storage_bytes_;
    uint8_t *d_min_storage     = d_min_temp_storage_.get();
    uint8_t *d_max_storage     = d_max_temp_storage_.get();
    float   *d_min             = reinterpret_cast<float *>(d_min_.get());
    float   *d_max             = reinterpret_cast<float *>(d_max_.get());
    CUDA_CHECK(
        cub::DeviceReduce::Min(d_min_storage, min_storage_bytes, idata, d_min, size, stream_));
    CUDA_CHECK(
        cub::DeviceReduce::Max(d_max_storage, max_storage_bytes, idata, d_max, size, stream_));

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    f32_u16_scaled_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size, d_min, d_max);
    CUDA_CHECK(cudaGetLastError());
  }

  void launch_f32_cf32_real(holoflow::core::TView in, holoflow::core::TView out) {
    auto *idata = reinterpret_cast<const float *>(in.data());
    auto *odata = reinterpret_cast<cuFloatComplex *>(out.data());
    int   size  = static_cast<int>(in.desc.num_elements());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    f32_cf32_real_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size);
    CUDA_CHECK(cudaGetLastError());
  }

  void launch_cf32_f32_modulus(holoflow::core::TView in, holoflow::core::TView out) {
    auto *idata = reinterpret_cast<const cuFloatComplex *>(in.data());
    auto *odata = reinterpret_cast<float *>(out.data());
    int   size  = static_cast<int>(in.desc.num_elements());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    cf32_f32_modulus_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size);
    CUDA_CHECK(cudaGetLastError());
  }

  void launch_cf32_f32_argument(holoflow::core::TView in, holoflow::core::TView out) {
    auto *idata = reinterpret_cast<const cuFloatComplex *>(in.data());
    auto *odata = reinterpret_cast<float *>(out.data());
    int   size  = static_cast<int>(in.desc.num_elements());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    cf32_f32_argument_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size);
    CUDA_CHECK(cudaGetLastError());
  }

  ConversionSettings    settings_;
  holoflow::core::TDesc idesc_;
  size_t                min_temp_storage_bytes_;
  DevPtr<uint8_t>       d_min_temp_storage_;
  DevPtr<std::byte>     d_min_;
  size_t                max_temp_storage_bytes_;
  DevPtr<uint8_t>       d_max_temp_storage_;
  DevPtr<std::byte>     d_max_;
  cudaStream_t          stream_;
};

// -------------------------------------------------------------------------------------------------
// ConversionFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
ConversionFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ConversionSettings>();

  check(input_descs.size() == 1, "Expected exactly one input tensor");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "Input tensor must be in device memory");
  check(is_c_contiguous(idesc), "Input tensor must be C-contiguous");

  const auto cfg = Cfg{idesc.dtype, settings.target, settings.strategy};
  check(cfg_to_dtype().contains(cfg), "Unsupported conversion configuration");

  holoflow::core::TDesc odesc(idesc.shape, cfg_to_dtype().at(cfg), holoflow::core::MemLoc::Device);
  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
ConversionFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &jsettings,
                          const holoflow::core::SyncCreateCtx   &ctx) const {
  CUDA_CHECK(cudaGetLastError());

  (void)infer(input_descs, jsettings);
  auto        settings = jsettings.get<ConversionSettings>();
  const auto &idesc    = input_descs[0];
  auto        count    = idesc.num_elements();

  size_t            min_storage_bytes = 0;
  DevPtr<uint8_t>   d_min_storage;
  DevPtr<std::byte> d_min;
  float            *fnull = nullptr;
  if (settings.strategy == ConversionSettings::Strategy::Scaled) {
    CUDA_CHECK(cub::DeviceReduce::Min(nullptr, min_storage_bytes, fnull, fnull, count, ctx.stream));
    d_min_storage = curaii::make_unique_device_ptr<uint8_t>(min_storage_bytes);
    d_min         = curaii::make_unique_device_ptr<std::byte>(sizeof(float));
  }

  size_t            max_storage_bytes = 0;
  DevPtr<uint8_t>   d_max_storage;
  DevPtr<std::byte> d_max;
  if (settings.strategy == ConversionSettings::Strategy::Scaled) {
    CUDA_CHECK(cub::DeviceReduce::Max(nullptr, max_storage_bytes, fnull, fnull, count, ctx.stream));
    d_max_storage = curaii::make_unique_device_ptr<uint8_t>(max_storage_bytes);
    d_max         = curaii::make_unique_device_ptr<std::byte>(sizeof(float));
  }

  return std::make_unique<Conversion>(settings, idesc, min_storage_bytes, std::move(d_min_storage),
                                      std::move(d_min), max_storage_bytes, std::move(d_max_storage),
                                      std::move(d_max), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
ConversionFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                          std::span<const holoflow::core::TDesc>     input_descs,
                          const nlohmann::json                      &jsettings,
                          const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_conversion = dynamic_cast<Conversion *>(old_task.get());
  if (old_conversion == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto &new_idesc = input_descs[0];
  const auto &old_idesc = old_conversion->idesc();
  const auto  settings  = jsettings.get<ConversionSettings>();
  const bool  can_reuse =
      settings == old_conversion->settings() && new_idesc.shape == old_idesc.shape &&
      new_idesc.strides == old_idesc.strides && new_idesc.dtype == old_idesc.dtype &&
      new_idesc.mem_loc == old_idesc.mem_loc;

  if (can_reuse) {
    old_conversion->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
