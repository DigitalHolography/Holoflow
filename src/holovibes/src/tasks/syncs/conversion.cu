#include "conversion.hh"

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <map>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holovibes::tasks {

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

Conversion::Conversion(const ConversionSettings &settings, size_t min_temp_storage_bytes,
                       DevPtr<uint8_t> &&d_min_temp_storage, DevPtr<std::byte> &&d_min,
                       size_t max_temp_storage_bytes, DevPtr<uint8_t> &&d_max_temp_storage,
                       DevPtr<std::byte> &&d_max, cudaStream_t stream)
    : settings_(settings), min_temp_storage_bytes_(min_temp_storage_bytes),
      d_min_temp_storage_(std::move(d_min_temp_storage)), d_min_(std::move(d_min)),
      max_temp_storage_bytes_(max_temp_storage_bytes),
      d_max_temp_storage_(std::move(d_max_temp_storage)), d_max_(std::move(d_max)),
      stream_(stream) {}

void Conversion::launch_u8_cf32_real(holoflow::core::CTView in, holoflow::core::TView out) {
  auto *idata = reinterpret_cast<const uint8_t *>(in.data);
  auto *odata = reinterpret_cast<cuFloatComplex *>(out.data);
  int   size  = static_cast<int>(in.desc.num_elements());

  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;
  u8_cf32_real_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size);
  CUDA_CHECK(cudaGetLastError());
}

void Conversion::launch_u16_cf32_real(holoflow::core::CTView in, holoflow::core::TView out) {
  auto *idata = reinterpret_cast<const uint16_t *>(in.data);
  auto *odata = reinterpret_cast<cuFloatComplex *>(out.data);
  int   size  = static_cast<int>(in.desc.num_elements());

  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;
  u16_cf32_real_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size);
  CUDA_CHECK(cudaGetLastError());
}

void Conversion::launch_f32_u8_scaled(holoflow::core::CTView in, holoflow::core::TView out) {
  auto *idata = reinterpret_cast<float *>(const_cast<std::byte *>(in.data));
  auto *odata = reinterpret_cast<uint8_t *>(out.data);
  auto  size  = in.desc.num_elements();

  // Compute min and max using CUB
  size_t   min_storage_bytes = min_temp_storage_bytes_;
  size_t   max_storage_bytes = max_temp_storage_bytes_;
  uint8_t *d_min_storage     = d_min_temp_storage_.get();
  uint8_t *d_max_storage     = d_max_temp_storage_.get();
  float   *d_min             = reinterpret_cast<float *>(d_min_.get());
  float   *d_max             = reinterpret_cast<float *>(d_max_.get());
  CUDA_CHECK(cub::DeviceReduce::Min(d_min_storage, min_storage_bytes, idata, d_min, size, stream_));
  CUDA_CHECK(cub::DeviceReduce::Max(d_max_storage, max_storage_bytes, idata, d_max, size, stream_));

  // Launch conversion kernel
  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;
  f32_u8_scaled_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size, d_min, d_max);
  CUDA_CHECK(cudaGetLastError());
}

void Conversion::launch_f32_u16_scaled(holoflow::core::CTView in, holoflow::core::TView out) {
  auto *idata = reinterpret_cast<const float *>(in.data);
  auto *odata = reinterpret_cast<uint16_t *>(out.data);
  int   size  = static_cast<int>(in.desc.num_elements());

  // Compute min and max using CUB
  size_t   min_storage_bytes = min_temp_storage_bytes_;
  size_t   max_storage_bytes = max_temp_storage_bytes_;
  uint8_t *d_min_storage     = d_min_temp_storage_.get();
  uint8_t *d_max_storage     = d_max_temp_storage_.get();
  float   *d_min             = reinterpret_cast<float *>(d_min_.get());
  float   *d_max             = reinterpret_cast<float *>(d_max_.get());
  CUDA_CHECK(cub::DeviceReduce::Min(d_min_storage, min_storage_bytes, idata, d_min, size, stream_));
  CUDA_CHECK(cub::DeviceReduce::Max(d_max_storage, max_storage_bytes, idata, d_max, size, stream_));

  // Launch conversion kernel
  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;
  f32_u16_scaled_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size, d_min, d_max);
  CUDA_CHECK(cudaGetLastError());
}

void Conversion::launch_cf32_f32_modulus(holoflow::core::CTView in, holoflow::core::TView out) {
  auto *idata = reinterpret_cast<const cuFloatComplex *>(in.data);
  auto *odata = reinterpret_cast<float *>(out.data);
  int   size  = static_cast<int>(in.desc.num_elements());

  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;
  cf32_f32_modulus_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size);
  CUDA_CHECK(cudaGetLastError());
}

void Conversion::launch_cf32_f32_argument(holoflow::core::CTView in, holoflow::core::TView out) {
  auto *idata = reinterpret_cast<const cuFloatComplex *>(in.data);
  auto *odata = reinterpret_cast<float *>(out.data);
  int   size  = static_cast<int>(in.desc.num_elements());

  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;
  cf32_f32_argument_kernel<<<num_blocks, block_size, 0, stream_>>>(idata, odata, size);
  CUDA_CHECK(cudaGetLastError());
}

holoflow::core::OpResult Conversion::execute(holoflow::core::SyncCtx &ctx) {
  auto in  = ctx.inputs[0];
  auto out = ctx.outputs[0];

  using holoflow::core::DType;
  using Target   = ConversionSettings::Target;
  using Strategy = ConversionSettings::Strategy;
  using Cfg      = std::tuple<DType, Target, Strategy>;
  using LaunchFn = void (Conversion::*)(holoflow::core::CTView, holoflow::core::TView);
  static const std::map<Cfg, LaunchFn> launch_map{
      {{DType::U8, Target::CF32, Strategy::Real}, &Conversion::launch_u8_cf32_real},
      {{DType::U16, Target::CF32, Strategy::Real}, &Conversion::launch_u16_cf32_real},
      {{DType::F32, Target::U8, Strategy::Scaled}, &Conversion::launch_f32_u8_scaled},
      {{DType::F32, Target::U16, Strategy::Scaled}, &Conversion::launch_f32_u16_scaled},
      {{DType::CF32, Target::F32, Strategy::Modulus}, &Conversion::launch_cf32_f32_modulus},
      {{DType::CF32, Target::F32, Strategy::Argument}, &Conversion::launch_cf32_f32_argument},
  };

  Cfg cfg{in.desc.dtype, settings_.target, settings_.strategy};
  HOLOVIBES_CHECK(launch_map.contains(cfg), "Unsupported conversion configuration");
  LaunchFn launch_fn = launch_map.at(cfg);
  (this->*launch_fn)(holoflow::core::CTView{.data = in.data, .desc = in.desc}, out);
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
ConversionFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[ConversionFactory::infer] error: {}", msg);
      throw std::invalid_argument("ConversionFactory inference error: " + msg);
    }
  };

  using holoflow::core::DType;
  using Target   = ConversionSettings::Target;
  using Strategy = ConversionSettings::Strategy;
  using Cfg      = std::tuple<DType, Target, Strategy>;
  std::map<Cfg, DType> cfg_to_dtype{
      {{DType::U8, Target::CF32, Strategy::Real}, DType::CF32},
      {{DType::U16, Target::CF32, Strategy::Real}, DType::CF32},
      {{DType::F32, Target::U8, Strategy::Scaled}, DType::U8},
      {{DType::F32, Target::U16, Strategy::Scaled}, DType::U16},
      {{DType::CF32, Target::F32, Strategy::Modulus}, DType::F32},
      {{DType::CF32, Target::F32, Strategy::Argument}, DType::F32},
  };

  auto settings = jsettings.get<ConversionSettings>();

  // Validate
  check(input_descs.size() == 1, "Expected exactly one input tensor");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "Input tensor must be in device memory");
  auto cfg = Cfg{idesc.dtype, settings.target, settings.strategy};
  check(cfg_to_dtype.contains(cfg), "Unsupported conversion configuration");

  // Success
  holoflow::core::TDesc odesc = idesc;
  odesc.dtype                 = cfg_to_dtype.at(cfg);
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
  // Validate
  auto infer_result = infer(input_descs, jsettings);
  auto settings     = jsettings.get<ConversionSettings>();
  auto count        = input_descs[0].num_elements();

  // CUB min
  size_t            min_storage_bytes = 0;
  DevPtr<uint8_t>   d_min_storage;
  DevPtr<std::byte> d_min;
  float            *fnull = nullptr;
  if (settings.strategy == ConversionSettings::Strategy::Scaled) {
    CUDA_CHECK(cub::DeviceReduce::Min(nullptr, min_storage_bytes, fnull, fnull, count, ctx.stream));
    d_min_storage = curaii::make_unique_device_ptr<uint8_t>(min_storage_bytes);
    d_min         = curaii::make_unique_device_ptr<std::byte>(sizeof(float));
  }

  // CUB max
  size_t            max_storage_bytes = 0;
  DevPtr<uint8_t>   d_max_storage;
  DevPtr<std::byte> d_max;
  if (settings.strategy == ConversionSettings::Strategy::Scaled) {
    CUDA_CHECK(cub::DeviceReduce::Max(nullptr, max_storage_bytes, fnull, fnull, count, ctx.stream));
    d_max_storage = curaii::make_unique_device_ptr<uint8_t>(max_storage_bytes);
    d_max         = curaii::make_unique_device_ptr<std::byte>(sizeof(float));
  }

  // Success
  auto *task =
      new Conversion(settings, min_storage_bytes, std::move(d_min_storage), std::move(d_min),
                     max_storage_bytes, std::move(d_max_storage), std::move(d_max), ctx.stream);
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holovibes::tasks