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

#include "holonp/median.hh"
#include "holonp/percentile.hh"
#include "holonp/quantile.hh"
#include "utils/tensor_common.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <cub/block/block_reduce.cuh>

#include "curaii/cuda.hh"

namespace holonp {
namespace {

struct QuantilePlan {
  std::vector<int>          axes;
  std::vector<int>          output_to_input;
  std::vector<size_t>       output_shape;
  std::vector<std::int64_t> input_strides;
  std::vector<std::int64_t> output_strides;
  std::vector<std::int64_t> reduction_strides;
  std::int64_t              output_elements    = 0;
  std::int64_t              reduction_elements = 0;
};

void serialize_axes(nlohmann::json &j, const std::vector<int> &axis, bool keepdims) {
  j["keepdims"] = keepdims;
  if (axis.empty())
    j["axis"] = nullptr;
  else if (axis.size() == 1)
    j["axis"] = axis[0];
  else
    j["axis"] = axis;
}

void parse_axes(const nlohmann::json &j, std::vector<int> &axis, bool &keepdims) {
  axis.clear();
  if (j.is_null()) {
    keepdims = false;
    return;
  }
  keepdims = j.value("keepdims", false);
  if (!j.contains("axis") || j["axis"].is_null())
    return;
  if (j["axis"].is_number_integer())
    axis = {j["axis"].get<int>()};
  else if (j["axis"].is_array())
    j.at("axis").get_to(axis);
  else
    throw std::invalid_argument("quantile axis must be an integer, array, or null");
}

std::int64_t product(const std::vector<size_t> &shape) {
  std::int64_t result = 1;
  for (const auto dim : shape) {
    if (dim == 0 ||
        result > std::numeric_limits<std::int64_t>::max() / static_cast<std::int64_t>(dim))
      throw std::invalid_argument("quantile has empty or oversized dimensions");
    result *= static_cast<std::int64_t>(dim);
  }
  return result;
}

QuantilePlan make_plan(const holoflow::core::TDesc &desc, std::vector<int> axes, bool keepdims) {
  const int rank = static_cast<int>(desc.shape.size());
  if (axes.empty()) {
    axes.resize(static_cast<size_t>(rank));
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    for (auto &axis : axes) {
      if (axis < 0)
        axis += rank;
      if (axis < 0 || axis >= rank)
        throw std::invalid_argument("quantile axis out of range");
    }
    std::sort(axes.begin(), axes.end());
    if (std::adjacent_find(axes.begin(), axes.end()) != axes.end())
      throw std::invalid_argument("quantile axes must be unique");
  }

  std::vector<bool> reduced(static_cast<size_t>(rank), false);
  for (const auto axis : axes)
    reduced[static_cast<size_t>(axis)] = true;

  QuantilePlan plan;
  plan.axes = std::move(axes);
  if (keepdims) {
    plan.output_shape = desc.shape;
    plan.output_to_input.resize(desc.shape.size());
    std::iota(plan.output_to_input.begin(), plan.output_to_input.end(), 0);
    for (const auto axis : plan.axes)
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
  for (const auto axis : plan.axes)
    reduction_shape.push_back(desc.shape[static_cast<size_t>(axis)]);

  plan.output_elements    = product(plan.output_shape);
  plan.reduction_elements = product(reduction_shape);
  if (plan.output_elements > std::numeric_limits<int>::max())
    throw std::invalid_argument("quantile output has too many elements for a CUDA grid");
  if (desc.strides.empty()) {
    plan.input_strides = utils::compact_strides_i64(desc.shape);
  } else {
    plan.input_strides.reserve(desc.strides.size());
    for (const auto stride : desc.strides) {
      if (stride % sizeof(float) != 0)
        throw std::invalid_argument("quantile requires element-aligned input strides");
      plan.input_strides.push_back(static_cast<std::int64_t>(stride / sizeof(float)));
    }
  }
  plan.output_strides    = utils::compact_strides_i64(plan.output_shape);
  plan.reduction_strides = utils::compact_strides_i64(reduction_shape);
  return plan;
}

__device__ std::int64_t reduction_offset(std::int64_t reduction_index, std::int64_t input_base,
                                         int reduction_rank, const std::int64_t *reduction_strides,
                                         const int          *reduction_axes,
                                         const std::int64_t *input_strides) {
  auto offset = input_base;
  for (int i = 0; i < reduction_rank; ++i) {
    const auto coordinate = reduction_index / reduction_strides[i];
    reduction_index -= coordinate * reduction_strides[i];
    offset += coordinate * input_strides[reduction_axes[i]];
  }
  return offset;
}

__device__ std::uint32_t float_order_key(float value) {
  const auto bits = __float_as_uint(value);
  return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}

template <int BlockSize>
__device__ float
select_rank(const float *input, std::int64_t input_base, std::int64_t reduction_elements,
            int reduction_rank, const std::int64_t *reduction_strides, const int *reduction_axes,
            const std::int64_t *input_strides, std::int64_t target_rank,
            typename cub::BlockReduce<std::int64_t, BlockSize>::TempStorage &reduce_storage,
            std::uint32_t &prefix, std::uint32_t &prefix_mask, std::int64_t &rank_remaining) {
  if (threadIdx.x == 0) {
    prefix         = 0;
    prefix_mask    = 0;
    rank_remaining = target_rank;
  }
  __syncthreads();

  for (int bit = 31; bit >= 0; --bit) {
    const auto   bit_mask         = std::uint32_t{1} << bit;
    std::int64_t local_zero_count = 0;
    for (std::int64_t index = threadIdx.x; index < reduction_elements; index += BlockSize) {
      const auto input_index = reduction_offset(index, input_base, reduction_rank,
                                                reduction_strides, reduction_axes, input_strides);
      const auto key         = float_order_key(input[input_index]);
      if ((key & prefix_mask) == prefix && (key & bit_mask) == 0)
        ++local_zero_count;
    }

    const auto zero_count =
        cub::BlockReduce<std::int64_t, BlockSize>(reduce_storage).Sum(local_zero_count);
    if (threadIdx.x == 0) {
      if (rank_remaining >= zero_count) {
        prefix |= bit_mask;
        rank_remaining -= zero_count;
      }
      prefix_mask |= bit_mask;
    }
    __syncthreads();
  }

  const auto bits = (prefix & 0x80000000u) != 0 ? prefix ^ 0x80000000u : ~prefix;
  return __uint_as_float(bits);
}

constexpr int kQuantileBlockSize = 128;

__global__ void quantile_kernel(const float *input, float *output, std::int64_t output_elements,
                                std::int64_t reduction_elements, int output_rank,
                                int reduction_rank, const std::int64_t *output_strides,
                                const int *output_to_input, const std::int64_t *input_strides,
                                const int *reduction_axes, const std::int64_t *reduction_strides,
                                float q) {
  using BlockReduce = cub::BlockReduce<std::int64_t, kQuantileBlockSize>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ std::uint32_t prefix;
  __shared__ std::uint32_t prefix_mask;
  __shared__ std::int64_t rank_remaining;
  __shared__ std::int64_t has_nan;
  __shared__ std::int64_t input_base;

  if (blockIdx.x >= output_elements)
    return;

  if (threadIdx.x == 0) {
    auto output_coordinate = static_cast<std::int64_t>(blockIdx.x);
    input_base             = 0;
    for (int i = 0; i < output_rank; ++i) {
      const auto coordinate = output_coordinate / output_strides[i];
      output_coordinate -= coordinate * output_strides[i];
      input_base += coordinate * input_strides[output_to_input[i]];
    }
    has_nan = 0;
  }
  __syncthreads();

  std::int64_t local_nan_count = 0;
  for (std::int64_t index = threadIdx.x; index < reduction_elements; index += kQuantileBlockSize) {
    const auto input_index = reduction_offset(index, input_base, reduction_rank, reduction_strides,
                                              reduction_axes, input_strides);
    local_nan_count += isnan(input[input_index]);
  }
  const auto nan_count = BlockReduce(reduce_storage).Sum(local_nan_count);
  if (threadIdx.x == 0)
    has_nan = nan_count;
  __syncthreads();
  if (has_nan != 0) {
    if (threadIdx.x == 0)
      output[blockIdx.x] = NAN;
    return;
  }

  const float position   = q * static_cast<float>(reduction_elements - 1);
  const auto  lower_rank = static_cast<std::int64_t>(floorf(position));
  const auto  upper_rank = static_cast<std::int64_t>(ceilf(position));
  const float lower      = select_rank<kQuantileBlockSize>(
      input, input_base, reduction_elements, reduction_rank, reduction_strides, reduction_axes,
      input_strides, lower_rank, reduce_storage, prefix, prefix_mask, rank_remaining);
  const float upper = select_rank<kQuantileBlockSize>(
      input, input_base, reduction_elements, reduction_rank, reduction_strides, reduction_axes,
      input_strides, upper_rank, reduce_storage, prefix, prefix_mask, rank_remaining);
  if (threadIdx.x == 0)
    output[blockIdx.x] = lower + (upper - lower) * (position - static_cast<float>(lower_rank));
}

template <typename T> curaii::unique_device_ptr<T> upload_mapping(const std::vector<T> &values) {
  auto device = curaii::make_unique_device_ptr<T>(values.size());
  if (!values.empty())
    CUDA_CHECK(
        cudaMemcpy(device.get(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  return device;
}

class Task final : public holoflow::core::ISyncTask {
public:
  Task(cudaStream_t stream, QuantilePlan plan, float q)
      : stream_(stream), plan_(std::move(plan)), q_(q),
        output_strides_(upload_mapping(plan_.output_strides)),
        input_strides_(upload_mapping(plan_.input_strides)),
        reduction_strides_(upload_mapping(plan_.reduction_strides)),
        output_to_input_(upload_mapping(plan_.output_to_input)),
        reduction_axes_(upload_mapping(plan_.axes)) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &context) override {
    quantile_kernel<<<static_cast<unsigned int>(plan_.output_elements), kQuantileBlockSize, 0,
                      stream_>>>(
        reinterpret_cast<const float *>(context.inputs[0].data()),
        reinterpret_cast<float *>(context.outputs[0].data()), plan_.output_elements,
        plan_.reduction_elements, static_cast<int>(plan_.output_shape.size()),
        static_cast<int>(plan_.axes.size()), output_strides_.get(), output_to_input_.get(),
        input_strides_.get(), reduction_axes_.get(), reduction_strides_.get(), q_);
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

private:
  cudaStream_t                            stream_;
  QuantilePlan                            plan_;
  float                                   q_;
  curaii::unique_device_ptr<std::int64_t> output_strides_;
  curaii::unique_device_ptr<std::int64_t> input_strides_;
  curaii::unique_device_ptr<std::int64_t> reduction_strides_;
  curaii::unique_device_ptr<int>          output_to_input_;
  curaii::unique_device_ptr<int>          reduction_axes_;
};

holoflow::core::InferResult infer_common(std::span<const holoflow::core::TDesc> input_descs,
                                         const nlohmann::json &json, int mode,
                                         QuantilePlan *out_plan, float *out_q) {
  if (input_descs.size() != 1)
    throw std::invalid_argument("quantile expects exactly one input");
  const auto &input = input_descs[0];
  if (input.mem_loc != holoflow::core::MemLoc::Device || input.dtype != holoflow::core::DType::F32)
    throw std::invalid_argument("quantile requires a non-empty Device F32 input");
  if (input.num_elements() == 0)
    throw std::invalid_argument("quantile requires a non-empty input");

  std::vector<int> axes;
  bool             keepdims = false;
  float            q        = 0.5f;
  if (mode == 1) {
    const auto settings = json.get<MedianSettings>();
    axes                = settings.axis;
    keepdims            = settings.keepdims;
  } else if (mode == 0) {
    const auto settings = json.get<QuantileSettings>();
    axes                = settings.axis;
    keepdims            = settings.keepdims;
    q                   = settings.q;
  } else {
    const auto settings = json.get<PercentileSettings>();
    axes                = settings.axis;
    keepdims            = settings.keepdims;
    q                   = settings.q / 100.0f;
  }
  if (!std::isfinite(q) || q < 0.0f || q > 1.0f)
    throw std::invalid_argument("quantile q must be finite and in [0, 1]");

  auto plan = make_plan(input, std::move(axes), keepdims);
  if (out_plan != nullptr)
    *out_plan = plan;
  if (out_q != nullptr)
    *out_q = q;
  return {{input},
          {utils::make_contiguous_desc(plan.output_shape, holoflow::core::DType::F32,
                                       holoflow::core::MemLoc::Device)},
          {},
          {false},
          {false},
          holoflow::core::TaskKind::Sync};
}

} // namespace

void to_json(nlohmann::json &j, const QuantileSettings &settings) {
  j["q"] = settings.q;
  serialize_axes(j, settings.axis, settings.keepdims);
}

void from_json(const nlohmann::json &j, QuantileSettings &settings) {
  if (j.is_null()) {
    settings = {};
    return;
  }
  settings.q = j.value("q", 0.5f);
  parse_axes(j, settings.axis, settings.keepdims);
}

void to_json(nlohmann::json &j, const MedianSettings &settings) {
  serialize_axes(j, settings.axis, settings.keepdims);
}

void from_json(const nlohmann::json &j, MedianSettings &settings) {
  parse_axes(j, settings.axis, settings.keepdims);
}

void to_json(nlohmann::json &j, const PercentileSettings &settings) {
  j["q"] = settings.q;
  serialize_axes(j, settings.axis, settings.keepdims);
}

void from_json(const nlohmann::json &j, PercentileSettings &settings) {
  if (j.is_null()) {
    settings = {};
    return;
  }
  settings.q = j.value("q", 50.0f);
  parse_axes(j, settings.axis, settings.keepdims);
}

holoflow::core::InferResult
QuantileFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &json) const {
  return infer_common(input_descs, json, 0, nullptr, nullptr);
}

std::unique_ptr<holoflow::core::ISyncTask>
QuantileFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &json,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  QuantilePlan plan;
  float        q;
  (void)infer_common(input_descs, json, 0, &plan, &q);
  return std::make_unique<Task>(ctx.stream, std::move(plan), q);
}

std::unique_ptr<holoflow::core::ISyncTask> QuantileFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask>, std::span<const holoflow::core::TDesc> input_descs,
    const nlohmann::json &json, const holoflow::core::SyncCreateCtx &ctx) const {
  return create(input_descs, json, ctx);
}

holoflow::core::InferResult MedianFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json &json) const {
  return infer_common(input_descs, json, 1, nullptr, nullptr);
}

std::unique_ptr<holoflow::core::ISyncTask>
MedianFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json &json, const holoflow::core::SyncCreateCtx &ctx) const {
  QuantilePlan plan;
  float        q;
  (void)infer_common(input_descs, json, 1, &plan, &q);
  return std::make_unique<Task>(ctx.stream, std::move(plan), q);
}

std::unique_ptr<holoflow::core::ISyncTask>
MedianFactory::update(std::unique_ptr<holoflow::core::ISyncTask>,
                      std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json &json, const holoflow::core::SyncCreateCtx &ctx) const {
  return create(input_descs, json, ctx);
}

holoflow::core::InferResult
PercentileFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &json) const {
  return infer_common(input_descs, json, 2, nullptr, nullptr);
}

std::unique_ptr<holoflow::core::ISyncTask>
PercentileFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                          const nlohmann::json                  &json,
                          const holoflow::core::SyncCreateCtx   &ctx) const {
  QuantilePlan plan;
  float        q;
  (void)infer_common(input_descs, json, 2, &plan, &q);
  return std::make_unique<Task>(ctx.stream, std::move(plan), q);
}

std::unique_ptr<holoflow::core::ISyncTask> PercentileFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask>, std::span<const holoflow::core::TDesc> input_descs,
    const nlohmann::json &json, const holoflow::core::SyncCreateCtx &ctx) const {
  return create(input_descs, json, ctx);
}

} // namespace holonp
