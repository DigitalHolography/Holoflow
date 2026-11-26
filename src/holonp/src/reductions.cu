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

#include "curaii/cuda.hh"
#include "holonp/argmin.hh"
#include "holonp/std.hh"
#include "holonp/sum.hh"
#include "holonp/var.hh"
#include "utils/tensor_common.hh"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace holonp {
namespace {

struct ReductionSettings {
  std::vector<int> axis;
  bool             keepdims = false;
};

void serialize_axes(nlohmann::json &j, const ReductionSettings &s) {
  j["keepdims"] = s.keepdims;
  if (s.axis.empty())
    j["axis"] = nullptr;
  else if (s.axis.size() == 1)
    j["axis"] = s.axis[0];
  else
    j["axis"] = s.axis;
}

void parse_axes(const nlohmann::json &j, ReductionSettings &s) {
  s.axis.clear();
  s.keepdims = j.value("keepdims", false);
  if (!j.contains("axis") || j["axis"].is_null())
    return;
  if (j["axis"].is_number_integer())
    s.axis = {j["axis"].get<int>()};
  else if (j["axis"].is_array())
    j.at("axis").get_to(s.axis);
  else
    throw std::invalid_argument("reduction axis must be an integer, array, or null");
}

std::int64_t product(const std::vector<size_t> &shape) {
  std::int64_t result = 1;
  for (const auto dim : shape) {
    if (dim == 0 ||
        result > std::numeric_limits<std::int64_t>::max() / static_cast<std::int64_t>(dim))
      throw std::invalid_argument("reduction has empty or oversized dimensions");
    result *= static_cast<std::int64_t>(dim);
  }
  return result;
}

struct ReductionPlan {
  std::vector<int>          axes;
  std::vector<int>          output_to_input;
  std::vector<size_t>       output_shape;
  std::vector<std::int64_t> input_strides;
  std::vector<std::int64_t> output_strides;
  std::vector<std::int64_t> reduction_strides;
  std::int64_t              output_elements;
  std::int64_t              reduction_elements;
};

ReductionPlan make_plan(const holoflow::core::TDesc &desc, const ReductionSettings &settings) {
  const int        rank = static_cast<int>(desc.shape.size());
  std::vector<int> axes = settings.axis;
  if (axes.empty()) {
    axes.resize(static_cast<size_t>(rank));
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    for (auto &axis : axes) {
      if (axis < 0)
        axis += rank;
      if (axis < 0 || axis >= rank)
        throw std::invalid_argument("reduction axis out of range");
    }
    std::sort(axes.begin(), axes.end());
    if (std::adjacent_find(axes.begin(), axes.end()) != axes.end())
      throw std::invalid_argument("reduction axes must be unique");
  }

  std::vector<bool> reduced(static_cast<size_t>(rank), false);
  for (const auto axis : axes)
    reduced[static_cast<size_t>(axis)] = true;

  ReductionPlan plan;
  plan.axes = axes;
  if (settings.keepdims) {
    plan.output_shape = desc.shape;
    plan.output_to_input.resize(desc.shape.size());
    std::iota(plan.output_to_input.begin(), plan.output_to_input.end(), 0);
    for (const auto axis : axes)
      plan.output_shape[static_cast<size_t>(axis)] = 1;
  } else {
    for (int i = 0; i < rank; ++i) {
      if (!reduced[static_cast<size_t>(i)]) {
        plan.output_shape.push_back(desc.shape[static_cast<size_t>(i)]);
        plan.output_to_input.push_back(i);
      }
    }
  }

  std::vector<size_t> reduction_shape;
  for (const auto axis : axes)
    reduction_shape.push_back(desc.shape[static_cast<size_t>(axis)]);

  plan.output_elements    = product(plan.output_shape);
  plan.reduction_elements = product(reduction_shape);
  if (desc.strides.empty()) {
    plan.input_strides = utils::compact_strides_i64(desc.shape);
  } else {
    plan.input_strides.reserve(desc.strides.size());
    for (const auto stride : desc.strides) {
      if (stride % sizeof(float) != 0)
        throw std::invalid_argument("reduction requires element-aligned input strides");
      plan.input_strides.push_back(static_cast<std::int64_t>(stride / sizeof(float)));
    }
  }
  plan.output_strides    = utils::compact_strides_i64(plan.output_shape);
  plan.reduction_strides = utils::compact_strides_i64(reduction_shape);
  return plan;
}

template <int Mode>
__global__ void reduction_kernel(const float *input, float *output, std::int64_t output_elements,
                                 std::int64_t reduction_elements, int output_rank,
                                 int reduction_rank, const std::int64_t *output_strides,
                                 const int *output_to_input, const std::int64_t *input_strides,
                                 const int *reduction_axes, const std::int64_t *reduction_strides) {
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

  float sum = 0.0f;
  for (std::int64_t reduction_index = 0; reduction_index < reduction_elements; ++reduction_index) {
    auto coordinate_index = reduction_index;
    auto input_offset     = input_base;
    for (int i = 0; i < reduction_rank; ++i) {
      const auto coordinate = coordinate_index / reduction_strides[i];
      coordinate_index -= coordinate * reduction_strides[i];
      input_offset += coordinate * input_strides[reduction_axes[i]];
    }
    sum += input[input_offset];
  }

  if (Mode == 0) {
    output[output_index] = sum;
    return;
  }

  const float mean          = sum / static_cast<float>(reduction_elements);
  float       squared_error = 0.0f;
  for (std::int64_t reduction_index = 0; reduction_index < reduction_elements; ++reduction_index) {
    auto coordinate_index = reduction_index;
    auto input_offset     = input_base;
    for (int i = 0; i < reduction_rank; ++i) {
      const auto coordinate = coordinate_index / reduction_strides[i];
      coordinate_index -= coordinate * reduction_strides[i];
      input_offset += coordinate * input_strides[reduction_axes[i]];
    }
    const float delta = input[input_offset] - mean;
    squared_error += delta * delta;
  }
  output[output_index] = squared_error / static_cast<float>(reduction_elements);
  if (Mode == 1)
    output[output_index] = sqrtf(output[output_index]);
}

__global__ void argmin_kernel(const float *input, std::uint16_t *output,
                              std::int64_t output_elements, std::int64_t reduction_elements,
                              int output_rank, int reduction_rank,
                              const std::int64_t *output_strides, const int *output_to_input,
                              const std::int64_t *input_strides, const int *reduction_axes,
                              const std::int64_t *reduction_strides) {
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

  float         best_value = INFINITY;
  std::uint16_t best_index = 0;
  for (std::int64_t reduction_index = 0; reduction_index < reduction_elements; ++reduction_index) {
    auto coordinate_index = reduction_index;
    auto input_offset     = input_base;
    for (int i = 0; i < reduction_rank; ++i) {
      const auto coordinate = coordinate_index / reduction_strides[i];
      coordinate_index -= coordinate * reduction_strides[i];
      input_offset += coordinate * input_strides[reduction_axes[i]];
    }
    const float value = input[input_offset];
    if (value < best_value) {
      best_value = value;
      best_index = static_cast<std::uint16_t>(reduction_index);
    }
  }
  output[output_index] = best_index;
}

class Task final : public holoflow::core::ISyncTask {
public:
  Task(int mode, cudaStream_t stream, ReductionPlan plan)
      : mode_(mode), stream_(stream), plan_(std::move(plan)),
        output_strides_(upload(plan_.output_strides)), input_strides_(upload(plan_.input_strides)),
        reduction_strides_(upload(plan_.reduction_strides)),
        output_to_input_(upload(plan_.output_to_input)), reduction_axes_(upload(plan_.axes)) {}

  bool supports_cuda_graph() const noexcept override { return true; }
  void record_cuda_graph(holoflow::core::CudaGraphCtx &ctx) override {
    holoflow::core::SyncCtx call{ctx.inputs, ctx.outputs, nullptr, nullptr, nullptr};
    (void)execute(call);
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &context) override {
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((plan_.output_elements + block_size - 1) / block_size);

    if (mode_ == 3) {
      argmin_kernel<<<grid_size, block_size, 0, stream_>>>(
          reinterpret_cast<const float *>(context.inputs[0].data()),
          reinterpret_cast<std::uint16_t *>(context.outputs[0].data()), plan_.output_elements,
          plan_.reduction_elements, static_cast<int>(plan_.output_shape.size()),
          static_cast<int>(plan_.axes.size()), output_strides_.get(), output_to_input_.get(),
          input_strides_.get(), reduction_axes_.get(), reduction_strides_.get());
    } else if (mode_ == 0) {
      reduction_kernel<0><<<grid_size, block_size, 0, stream_>>>(
          reinterpret_cast<const float *>(context.inputs[0].data()),
          reinterpret_cast<float *>(context.outputs[0].data()), plan_.output_elements,
          plan_.reduction_elements, static_cast<int>(plan_.output_shape.size()),
          static_cast<int>(plan_.axes.size()), output_strides_.get(), output_to_input_.get(),
          input_strides_.get(), reduction_axes_.get(), reduction_strides_.get());
    } else if (mode_ == 1) {
      reduction_kernel<1><<<grid_size, block_size, 0, stream_>>>(
          reinterpret_cast<const float *>(context.inputs[0].data()),
          reinterpret_cast<float *>(context.outputs[0].data()), plan_.output_elements,
          plan_.reduction_elements, static_cast<int>(plan_.output_shape.size()),
          static_cast<int>(plan_.axes.size()), output_strides_.get(), output_to_input_.get(),
          input_strides_.get(), reduction_axes_.get(), reduction_strides_.get());
    } else {
      reduction_kernel<2><<<grid_size, block_size, 0, stream_>>>(
          reinterpret_cast<const float *>(context.inputs[0].data()),
          reinterpret_cast<float *>(context.outputs[0].data()), plan_.output_elements,
          plan_.reduction_elements, static_cast<int>(plan_.output_shape.size()),
          static_cast<int>(plan_.axes.size()), output_strides_.get(), output_to_input_.get(),
          input_strides_.get(), reduction_axes_.get(), reduction_strides_.get());
    }
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

private:
  template <typename T> curaii::unique_device_ptr<T> upload(const std::vector<T> &values) {
    auto device = curaii::make_unique_device_ptr<T>(values.size());
    if (!values.empty())
      CUDA_CHECK(cudaMemcpyAsync(device.get(), values.data(), values.size() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream_));
    return device;
  }

  int                                     mode_;
  cudaStream_t                            stream_;
  ReductionPlan                           plan_;
  curaii::unique_device_ptr<std::int64_t> output_strides_, input_strides_, reduction_strides_;
  curaii::unique_device_ptr<int>          output_to_input_, reduction_axes_;
};

template <class SettingsType>
holoflow::core::InferResult reduction_infer(std::span<const holoflow::core::TDesc> inputs,
                                            const nlohmann::json                  &json,
                                            holoflow::core::DType                  output_dtype) {
  if (inputs.size() != 1 || inputs[0].dtype != holoflow::core::DType::F32 ||
      inputs[0].mem_loc != holoflow::core::MemLoc::Device || inputs[0].num_elements() == 0)
    throw std::invalid_argument("reduction requires non-empty device F32 input");
  const auto settings = json.get<SettingsType>();
  const auto plan     = make_plan(inputs[0], {settings.axis, settings.keepdims});
  const auto output_desc =
      utils::make_contiguous_desc(plan.output_shape, output_dtype, holoflow::core::MemLoc::Device);
  return {{inputs[0]}, {output_desc}, {}, {false}, {false}, holoflow::core::TaskKind::Sync};
}

template <class SettingsType>
std::unique_ptr<holoflow::core::ISyncTask>
reduction_create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                 const holoflow::core::SyncCreateCtx &context, int mode) {
  const auto settings = json.get<SettingsType>();
  (void)reduction_infer<SettingsType>(
      inputs, json, mode == 3 ? holoflow::core::DType::U16 : holoflow::core::DType::F32);
  return std::make_unique<Task>(mode, context.stream,
                                make_plan(inputs[0], {settings.axis, settings.keepdims}));
}

} // namespace

void to_json(nlohmann::json &j, const SumSettings &s) { serialize_axes(j, {s.axis, s.keepdims}); }
void from_json(const nlohmann::json &j, SumSettings &s) {
  ReductionSettings settings;
  parse_axes(j, settings);
  s.axis     = std::move(settings.axis);
  s.keepdims = settings.keepdims;
}

holoflow::core::InferResult SumFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                              const nlohmann::json                  &json) const {
  return reduction_infer<SumSettings>(inputs, json, holoflow::core::DType::F32);
}
std::unique_ptr<holoflow::core::ISyncTask>
SumFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                   const holoflow::core::SyncCreateCtx &context) const {
  return reduction_create<SumSettings>(inputs, json, context, 0);
}
std::unique_ptr<holoflow::core::ISyncTask>
SumFactory::update(std::unique_ptr<holoflow::core::ISyncTask>,
                   std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                   const holoflow::core::SyncCreateCtx &context) const {
  return reduction_create<SumSettings>(inputs, json, context, 0);
}

holoflow::core::InferResult StdFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                              const nlohmann::json                  &json) const {
  return reduction_infer<StdSettings>(inputs, json, holoflow::core::DType::F32);
}
std::unique_ptr<holoflow::core::ISyncTask>
StdFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                   const holoflow::core::SyncCreateCtx &context) const {
  return reduction_create<StdSettings>(inputs, json, context, 1);
}
std::unique_ptr<holoflow::core::ISyncTask>
StdFactory::update(std::unique_ptr<holoflow::core::ISyncTask>,
                   std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                   const holoflow::core::SyncCreateCtx &context) const {
  return reduction_create<StdSettings>(inputs, json, context, 1);
}

holoflow::core::InferResult VarFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                              const nlohmann::json                  &json) const {
  return reduction_infer<VarSettings>(inputs, json, holoflow::core::DType::F32);
}
std::unique_ptr<holoflow::core::ISyncTask>
VarFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                   const holoflow::core::SyncCreateCtx &context) const {
  return reduction_create<VarSettings>(inputs, json, context, 2);
}
std::unique_ptr<holoflow::core::ISyncTask>
VarFactory::update(std::unique_ptr<holoflow::core::ISyncTask>,
                   std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                   const holoflow::core::SyncCreateCtx &context) const {
  return reduction_create<VarSettings>(inputs, json, context, 2);
}

holoflow::core::InferResult ArgminFactory::infer(std::span<const holoflow::core::TDesc> inputs,
                                                 const nlohmann::json &json) const {
  return reduction_infer<ArgminSettings>(inputs, json, holoflow::core::DType::U16);
}
std::unique_ptr<holoflow::core::ISyncTask>
ArgminFactory::create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                      const holoflow::core::SyncCreateCtx &context) const {
  return reduction_create<ArgminSettings>(inputs, json, context, 3);
}
std::unique_ptr<holoflow::core::ISyncTask>
ArgminFactory::update(std::unique_ptr<holoflow::core::ISyncTask>,
                      std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json,
                      const holoflow::core::SyncCreateCtx &context) const {
  return reduction_create<ArgminSettings>(inputs, json, context, 3);
}

} // namespace holonp
