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

#include "holonp/histogram.hh"
#include "utils/tensor_common.hh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "curaii/cuda.hh"

namespace holonp {
namespace {

void validate(std::span<const holoflow::core::TDesc> input_descs,
              const HistogramSettings               &settings) {
  if (input_descs.size() != 1)
    throw std::invalid_argument("histogram expects exactly one input");
  const auto &input = input_descs[0];
  if (input.mem_loc != holoflow::core::MemLoc::Device || input.dtype != holoflow::core::DType::F32)
    throw std::invalid_argument("histogram requires a Device F32 input");
  if (input.num_elements() == 0)
    throw std::invalid_argument("histogram requires a non-empty input");
  if (!utils::is_c_contiguous(input))
    throw std::invalid_argument("histogram requires a C-contiguous input");
  if (settings.bins <= 0)
    throw std::invalid_argument("histogram bins must be positive");
  if (!std::isfinite(settings.min) || !std::isfinite(settings.max) || settings.min >= settings.max)
    throw std::invalid_argument("histogram range must be finite and increasing");
  constexpr std::uint64_t max_exact_float_integer = std::uint64_t{1} << 24;
  if (input.num_elements() > max_exact_float_integer)
    throw std::invalid_argument(
        "histogram input exceeds the exact integer range of F32 count outputs");
}

__global__ void histogram_edges_kernel(float *edges, int bins, float min_value, float max_value) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index > bins)
    return;
  edges[index] =
      min_value + (max_value - min_value) * (static_cast<float>(index) / static_cast<float>(bins));
}

__global__ void histogram_count_kernel(const float *input, std::int64_t elements, float *counts,
                                       int bins, float min_value, float max_value) {
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const float value = input[index];
  if (!isfinite(value) || value < min_value || value > max_value)
    return;

  int bin = bins - 1;
  if (value < max_value)
    bin = static_cast<int>(((value - min_value) / (max_value - min_value)) * bins);
  if (bin >= 0 && bin < bins)
    atomicAdd(&counts[bin], 1.0f);
}

class Task final : public holoflow::core::ISyncTask {
public:
  Task(HistogramSettings settings, cudaStream_t stream) : settings_(settings), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &context) override {
    const auto    elements = static_cast<std::int64_t>(input_desc_.num_elements());
    constexpr int block    = 256;
    const int     grid     = static_cast<int>((elements + block - 1) / block);
    CUDA_CHECK(cudaMemsetAsync(context.outputs[0].data(), 0,
                               static_cast<size_t>(settings_.bins) * sizeof(float), stream_));
    histogram_count_kernel<<<grid, block, 0, stream_>>>(
        reinterpret_cast<const float *>(context.inputs[0].data()), elements,
        reinterpret_cast<float *>(context.outputs[0].data()), settings_.bins, settings_.min,
        settings_.max);
    CUDA_CHECK(cudaGetLastError());

    const int edge_grid = (settings_.bins + 1 + block - 1) / block;
    histogram_edges_kernel<<<edge_grid, block, 0, stream_>>>(
        reinterpret_cast<float *>(context.outputs[1].data()), settings_.bins, settings_.min,
        settings_.max);
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  void set_input_desc(holoflow::core::TDesc desc) { input_desc_ = std::move(desc); }

private:
  HistogramSettings     settings_;
  cudaStream_t          stream_;
  holoflow::core::TDesc input_desc_;
};

} // namespace

void to_json(nlohmann::json &j, const HistogramSettings &settings) {
  j = {{"bins", settings.bins}, {"min", settings.min}, {"max", settings.max}};
}

void from_json(const nlohmann::json &j, HistogramSettings &settings) {
  if (j.is_null()) {
    settings = {};
    return;
  }
  settings.bins = j.value("bins", 10);
  settings.min  = j.value("min", 0.0f);
  settings.max  = j.value("max", 1.0f);
}

holoflow::core::InferResult
HistogramFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &json) const {
  const auto settings = json.get<HistogramSettings>();
  validate(input_descs, settings);
  const auto bins = static_cast<size_t>(settings.bins);
  return {{input_descs[0]},
          {utils::make_contiguous_desc({bins}, holoflow::core::DType::F32,
                                       holoflow::core::MemLoc::Device),
           utils::make_contiguous_desc({bins + 1}, holoflow::core::DType::F32,
                                       holoflow::core::MemLoc::Device)},
          {},
          {false},
          {false, false},
          holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
HistogramFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &json,
                         const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto settings = json.get<HistogramSettings>();
  validate(input_descs, settings);
  auto task = std::make_unique<Task>(settings, ctx.stream);
  task->set_input_desc(input_descs[0]);
  return task;
}

std::unique_ptr<holoflow::core::ISyncTask> HistogramFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask>, std::span<const holoflow::core::TDesc> input_descs,
    const nlohmann::json &json, const holoflow::core::SyncCreateCtx &ctx) const {
  return create(input_descs, json, ctx);
}

} // namespace holonp
