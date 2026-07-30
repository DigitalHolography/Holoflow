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

#include "holotask/syncs/causal_sliding_average.hh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "curaii/cuda.hh"
#include "logger.hh"

namespace holotask::syncs {

void to_json(nlohmann::json &j, const CausalSlidingAverageSettings &s) {
  j = nlohmann::json{{"window_size", s.window_size}};
}

void from_json(const nlohmann::json &j, CausalSlidingAverageSettings &s) {
  j.at("window_size").get_to(s.window_size);
}

namespace {

bool is_contiguous(const holoflow::core::TDesc &desc) {
  holoflow::core::TDesc contiguous(desc.shape, desc.dtype, desc.mem_loc, desc.offset);
  return desc.strides == contiguous.strides;
}

__global__ void causal_sliding_average_kernel(const float *input, float *output, float *history,
                                              float *running_sum, size_t history_offset,
                                              size_t element_count, float divisor) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= element_count) {
    return;
  }

  const size_t history_idx = history_offset + idx;
  const float  old_value   = history[history_idx];
  const float  new_value   = input[idx];
  const float  new_sum     = running_sum[idx] + new_value - old_value;
  history[history_idx]     = new_value;
  running_sum[idx]         = new_sum;
  output[idx]              = new_sum / divisor;
}

class CausalSlidingAverage final : public holoflow::core::ISyncTask {
public:
  CausalSlidingAverage(CausalSlidingAverageSettings settings, holoflow::core::TDesc desc,
                       cudaStream_t stream)
      : settings_(settings), desc_(std::move(desc)), stream_(stream),
        history_(
            curaii::make_unique_device_ptr<float>(settings_.window_size * desc_.num_elements())),
        running_sum_(curaii::make_unique_device_ptr<float>(desc_.num_elements())) {
    CUDA_CHECK(cudaMemsetAsync(
        history_.get(), 0, settings_.window_size * desc_.num_elements() * sizeof(float), stream_));
    CUDA_CHECK(
        cudaMemsetAsync(running_sum_.get(), 0, desc_.num_elements() * sizeof(float), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const size_t element_count = desc_.num_elements();
    const size_t slot          = sample_count_ % settings_.window_size;
    const auto   divisor =
        static_cast<float>(std::min(sample_count_ + size_t{1}, settings_.window_size));

    constexpr int block_size = 256;
    const auto grid_size = static_cast<unsigned int>((element_count + block_size - 1) / block_size);
    causal_sliding_average_kernel<<<grid_size, block_size, 0, stream_>>>(
        reinterpret_cast<const float *>(ctx.inputs[0].data()),
        reinterpret_cast<float *>(ctx.outputs[0].data()), history_.get(), running_sum_.get(),
        slot * element_count, element_count, divisor);
    CUDA_CHECK(cudaGetLastError());
    ++sample_count_;
    return holoflow::core::OpResult::Ok;
  }

private:
  CausalSlidingAverageSettings     settings_;
  holoflow::core::TDesc            desc_;
  cudaStream_t                     stream_;
  curaii::unique_device_ptr<float> history_;
  curaii::unique_device_ptr<float> running_sum_;
  size_t                           sample_count_ = 0;
};

} // namespace

holoflow::core::InferResult
CausalSlidingAverageFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                   const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool condition, const std::string &message) {
    if (!condition) {
      logger()->error("[CausalSlidingAverageFactory::infer] error: {}", message);
      throw std::invalid_argument("CausalSlidingAverageFactory inference error: " + message);
    }
  };

  const auto settings = jsettings.get<CausalSlidingAverageSettings>();
  check(input_descs.size() == 1, "task must have exactly one input");
  const auto &input = input_descs[0];
  check(input.dtype == holoflow::core::DType::F32, "input dtype must be F32");
  check(input.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(input.rank() > 0, "input rank must be positive");
  check(is_contiguous(input), "input must be contiguous");
  check(settings.window_size > 0, "window_size must be positive");

  return {
      .input_descs   = {input},
      .output_descs  = {input},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
CausalSlidingAverageFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json                  &jsettings,
                                    const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  return std::make_unique<CausalSlidingAverage>(jsettings.get<CausalSlidingAverageSettings>(),
                                                input_descs[0], ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask> CausalSlidingAverageFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask>, std::span<const holoflow::core::TDesc> input_descs,
    const nlohmann::json &jsettings, const holoflow::core::SyncCreateCtx &ctx) const {
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
