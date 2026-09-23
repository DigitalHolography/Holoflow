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

#include "holonp/norm.hh"
#include "utils/tensor_common.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#include <cuComplex.h>

#include "curaii/cuda.hh"

namespace holonp {

namespace {

struct NormSettingsView {
  std::vector<int> axis;
  bool             keepdims;
  double           ord;
};

void serialize_axes(nlohmann::json &j, const NormSettings &s) {
  if (s.axis.empty())
    j["axis"] = nullptr;
  else if (s.axis.size() == 1)
    j["axis"] = s.axis[0];
  else
    j["axis"] = s.axis;
  j["keepdims"] = s.keepdims;
  if (std::isinf(s.ord))
    j["ord"] = s.ord > 0 ? "inf" : "-inf";
  else
    j["ord"] = s.ord;
}

void parse_settings(const nlohmann::json &j, NormSettings &s) {
  s.axis.clear();
  s.keepdims = j.value("keepdims", false);
  s.ord      = 2.0;
  if (j.contains("ord")) {
    if (j.at("ord").is_string()) {
      const auto ord = j.at("ord").get<std::string>();
      if (ord == "inf" || ord == "+inf")
        s.ord = std::numeric_limits<double>::infinity();
      else if (ord == "-inf")
        s.ord = -std::numeric_limits<double>::infinity();
      else
        throw std::invalid_argument("Norm: unsupported ord");
    } else {
      s.ord = j.at("ord").get<double>();
    }
  }
  if (!j.contains("axis") || j.at("axis").is_null())
    return;
  if (j.at("axis").is_number_integer())
    s.axis = {j.at("axis").get<int>()};
  else if (j.at("axis").is_array())
    j.at("axis").get_to(s.axis);
  else
    throw std::invalid_argument("Norm: axis must be an integer, array, or null");
}

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("Norm: " + msg);
}

struct Plan {
  std::vector<int>          axes;
  std::vector<int>          output_to_input;
  std::vector<size_t>       output_shape;
  std::vector<std::int64_t> input_strides;
  std::vector<std::int64_t> output_strides;
  std::vector<std::int64_t> reduction_strides;
  std::int64_t              output_elements;
  std::int64_t              reduction_elements;
};

std::int64_t product(std::span<const size_t> shape) {
  std::int64_t result = 1;
  for (const auto dim : shape) {
    check(dim > 0 &&
              result <= std::numeric_limits<std::int64_t>::max() / static_cast<std::int64_t>(dim),
          "input has empty or oversized dimensions");
    result *= static_cast<std::int64_t>(dim);
  }
  return result;
}

Plan make_plan(const holoflow::core::TDesc &desc, const NormSettings &settings) {
  const int         rank = static_cast<int>(desc.shape.size());
  const auto        axes = utils::normalize_axes(settings.axis, rank);
  std::vector<bool> reduced(static_cast<size_t>(rank), false);
  for (const auto axis : axes)
    reduced[static_cast<size_t>(axis)] = true;

  Plan plan;
  plan.axes = axes;
  if (settings.keepdims) {
    plan.output_shape = desc.shape;
    plan.output_to_input.resize(desc.shape.size());
    std::iota(plan.output_to_input.begin(), plan.output_to_input.end(), 0);
    for (const auto axis : axes)
      plan.output_shape[static_cast<size_t>(axis)] = 1;
  } else {
    for (int axis = 0; axis < rank; ++axis) {
      if (!reduced[static_cast<size_t>(axis)]) {
        plan.output_shape.push_back(desc.shape[static_cast<size_t>(axis)]);
        plan.output_to_input.push_back(axis);
      }
    }
  }

  std::vector<size_t> reduction_shape;
  for (const auto axis : axes)
    reduction_shape.push_back(desc.shape[static_cast<size_t>(axis)]);

  plan.output_elements    = product(plan.output_shape);
  plan.reduction_elements = product(reduction_shape);
  if (desc.strides.empty()) {
    const auto compact = utils::compact_strides(desc.shape);
    plan.input_strides.assign(compact.begin(), compact.end());
  } else {
    const auto element_strides =
        utils::bytes_to_elements(desc.strides, holoflow::core::size_of(desc.dtype));
    plan.input_strides.assign(element_strides.begin(), element_strides.end());
  }
  plan.output_strides    = utils::compact_strides_i64(plan.output_shape);
  plan.reduction_strides = utils::compact_strides_i64(reduction_shape);
  return plan;
}

template <typename T> __device__ float magnitude(T value) { return fabsf(value); }

template <> __device__ float magnitude<cuFloatComplex>(cuFloatComplex value) {
  return hypotf(value.x, value.y);
}

template <typename T>
__global__ void
norm_kernel(const T *__restrict__ input, float *__restrict__ output, std::int64_t output_elements,
            std::int64_t reduction_elements, int output_rank, int reduction_rank,
            const std::int64_t *__restrict__ output_strides,
            const int *__restrict__ output_to_input, const std::int64_t *__restrict__ input_strides,
            const int *__restrict__ reduction_axes,
            const std::int64_t *__restrict__ reduction_strides, float ord) {
  const auto output_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_elements)
    return;

  auto         output_coordinate = output_index;
  std::int64_t input_base        = 0;
  for (int i = 0; i < output_rank; ++i) {
    const auto coordinate = output_coordinate / output_strides[i];
    output_coordinate -= coordinate * output_strides[i];
    input_base += coordinate * input_strides[output_to_input[i]];
  }

  const bool positive_inf = isinf(ord) && ord > 0.0f;
  const bool negative_inf = isinf(ord) && ord < 0.0f;
  float      accumulator  = positive_inf ? 0.0f : (negative_inf ? INFINITY : 0.0f);
  int        nonzero      = 0;
  for (std::int64_t reduction_index = 0; reduction_index < reduction_elements; ++reduction_index) {
    auto coordinate_index = reduction_index;
    auto input_offset     = input_base;
    for (int i = 0; i < reduction_rank; ++i) {
      const auto coordinate = coordinate_index / reduction_strides[i];
      coordinate_index -= coordinate * reduction_strides[i];
      input_offset += coordinate * input_strides[reduction_axes[i]];
    }
    const float value = magnitude(input[input_offset]);
    if (value != 0.0f)
      ++nonzero;
    if (ord == 0.0f)
      continue;
    if (positive_inf)
      accumulator = fmaxf(accumulator, value);
    else if (negative_inf)
      accumulator = fminf(accumulator, value);
    else if (ord == 1.0f)
      accumulator += value;
    else
      accumulator += powf(value, ord);
  }

  if (ord == 0.0f)
    output[output_index] = static_cast<float>(nonzero);
  else if (positive_inf || negative_inf || ord == 1.0f)
    output[output_index] = accumulator;
  else
    output[output_index] = powf(accumulator, 1.0f / ord);
}

class Norm : public holoflow::core::ISyncTask {
public:
  Norm(NormSettings settings, holoflow::core::TDesc idesc, Plan plan, cudaStream_t stream)
      : settings_(std::move(settings)), idesc_(std::move(idesc)), plan_(std::move(plan)),
        stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto upload = [&](const auto &values) {
      using T     = typename std::decay_t<decltype(values)>::value_type;
      auto device = curaii::make_unique_device_ptr<T>(values.size());
      if (!values.empty())
        CUDA_CHECK(cudaMemcpyAsync(device.get(), values.data(), values.size() * sizeof(T),
                                   cudaMemcpyHostToDevice, stream_));
      return device;
    };

    const auto    output_strides    = upload(plan_.output_strides);
    const auto    input_strides     = upload(plan_.input_strides);
    const auto    reduction_strides = upload(plan_.reduction_strides);
    const auto    output_to_input   = upload(plan_.output_to_input);
    const auto    reduction_axes    = upload(plan_.axes);
    constexpr int block_size        = 256;
    const int grid_size = static_cast<int>((plan_.output_elements + block_size - 1) / block_size);
    if (idesc_.dtype == holoflow::core::DType::F32) {
      norm_kernel<<<grid_size, block_size, 0, stream_>>>(
          reinterpret_cast<const float *>(ctx.inputs[0].data()),
          reinterpret_cast<float *>(ctx.outputs[0].data()), plan_.output_elements,
          plan_.reduction_elements, static_cast<int>(plan_.output_shape.size()),
          static_cast<int>(plan_.axes.size()), output_strides.get(), output_to_input.get(),
          input_strides.get(), reduction_axes.get(), reduction_strides.get(),
          static_cast<float>(settings_.ord));
    } else {
      norm_kernel<<<grid_size, block_size, 0, stream_>>>(
          reinterpret_cast<const cuFloatComplex *>(ctx.inputs[0].data()),
          reinterpret_cast<float *>(ctx.outputs[0].data()), plan_.output_elements,
          plan_.reduction_elements, static_cast<int>(plan_.output_shape.size()),
          static_cast<int>(plan_.axes.size()), output_strides.get(), output_to_input.get(),
          input_strides.get(), reduction_axes.get(), reduction_strides.get(),
          static_cast<float>(settings_.ord));
    }
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  const NormSettings          &settings() const { return settings_; }
  const holoflow::core::TDesc &idesc() const { return idesc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  NormSettings          settings_;
  holoflow::core::TDesc idesc_;
  Plan                  plan_;
  cudaStream_t          stream_;
};

} // namespace

void to_json(nlohmann::json &j, const NormSettings &s) { serialize_axes(j, s); }
void from_json(const nlohmann::json &j, NormSettings &s) { parse_settings(j, s); }

holoflow::core::InferResult NormFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                               const nlohmann::json &jsettings) const {
  check(input_descs.size() == 1, "expected exactly 1 input");
  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "only Device tensors are supported");
  check(idesc.dtype == holoflow::core::DType::F32 || idesc.dtype == holoflow::core::DType::CF32,
        "supported dtypes are F32 and CF32");
  check(idesc.num_elements() > 0, "input tensor has zero elements");
  const auto settings = jsettings.get<NormSettings>();
  check(settings.ord == 0.0 || settings.ord == 1.0 || settings.ord == 2.0 ||
            std::isinf(settings.ord) || settings.ord > 0.0,
        "ord must be nonnegative, positive infinity, or negative infinity");
  const auto plan  = make_plan(idesc, settings);
  const auto odesc = utils::make_contiguous_desc(plan.output_shape, holoflow::core::DType::F32,
                                                 holoflow::core::MemLoc::Device);
  return {.input_descs   = {idesc},
          .output_descs  = {odesc},
          .in_place      = {},
          .owned_inputs  = {false},
          .owned_outputs = {false},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
NormFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                    const nlohmann::json                  &jsettings,
                    const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  const auto settings = jsettings.get<NormSettings>();
  return std::make_unique<Norm>(settings, input_descs[0], make_plan(input_descs[0], settings),
                                ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
NormFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                    std::span<const holoflow::core::TDesc>     input_descs,
                    const nlohmann::json                      &jsettings,
                    const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);
  auto      *old_norm = dynamic_cast<Norm *>(old_task.get());
  const auto settings = jsettings.get<NormSettings>();
  if (old_norm != nullptr && input_descs.size() == 1 && old_norm->settings() == settings &&
      utils::same_desc(input_descs[0], old_norm->idesc())) {
    old_norm->update_stream(ctx.stream);
    return old_task;
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
