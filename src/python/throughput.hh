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

#pragma once

#include "holoflow/core/tasks.hh"
#include <chrono>

namespace holoflow::python {

class ThroughputTask final : public core::ISyncTask {
public:
  using Clock = std::chrono::steady_clock;

  ThroughputTask(size_t frames_per_batch, std::chrono::milliseconds interval)
      : frames_per_batch_(frames_per_batch), interval_(interval) {}

  core::OpResult execute(core::SyncCtx &) override { return core::OpResult::Ok; }
  bool           supports_cuda_graph() const noexcept override { return true; }
  void           record_cuda_graph(core::CudaGraphCtx &) override {}

  void start(Clock::time_point time) {
    started_ = previous_ = time;
    previous_batches_    = 0;
  }

  Clock::time_point deadline() const { return previous_ + interval_; }

  nlohmann::json sample(uint64_t batches, Clock::time_point now) {
    const double seconds = std::chrono::duration<double>(now - previous_).count();
    const auto   frames  = (batches - previous_batches_) * frames_per_batch_;
    previous_            = now;
    previous_batches_    = batches;
    return {{"type", "throughput"},
            {"frames", frames},
            {"interval_seconds", seconds},
            {"fps", frames / seconds},
            {"total_frames", batches * frames_per_batch_}};
  }

  nlohmann::json summary(uint64_t batches, Clock::time_point now, bool completed) const {
    const double   seconds = std::chrono::duration<double>(now - started_).count();
    const auto     frames  = batches * frames_per_batch_;
    nlohmann::json result  = {{"type", "throughput_summary"},
                              {"total_frames", frames},
                              {"elapsed_seconds", seconds},
                              {"completed", completed}};
    if (completed && seconds > 0)
      result["average_fps"] = frames / seconds;
    return result;
  }

private:
  size_t                    frames_per_batch_;
  std::chrono::milliseconds interval_;
  Clock::time_point         started_, previous_;
  uint64_t                  previous_batches_ = 0;
};

class ThroughputFactory final : public core::ISyncTaskFactory {
public:
  core::InferResult infer(std::span<const core::TDesc> inputs,
                          const nlohmann::json        &settings) const override {
    if (inputs.size() != 1)
      throw std::invalid_argument("Throughput expects one tensor");
    (void)frames_per_batch(inputs[0], settings);
    const auto &interval = settings.at("interval_ms");
    if (!interval.is_number_integer() || interval.get<double>() < 1 ||
        interval.get<double>() > 2147483647)
      throw std::invalid_argument("Throughput interval_ms must be a positive 32-bit integer");
    return {.input_descs   = {inputs[0]},
            .output_descs  = {inputs[0]},
            .in_place      = {{0, 0}},
            .owned_inputs  = {false},
            .owned_outputs = {false},
            .kind          = core::TaskKind::Sync};
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const core::TDesc> inputs,
                                          const nlohmann::json        &settings,
                                          const core::SyncCreateCtx &) const override {
    (void)infer(inputs, settings);
    return std::make_unique<ThroughputTask>(
        frames_per_batch(inputs[0], settings),
        std::chrono::milliseconds(settings.at("interval_ms").get<int>()));
  }

private:
  static size_t frames_per_batch(const core::TDesc &input, const nlohmann::json &settings) {
    const auto &axis = settings.at("axis");
    if (axis.is_null())
      return 1;
    if (!axis.is_number_integer() || axis.get<double>() < 0 ||
        axis.get<double>() >= input.shape.size())
      throw std::invalid_argument("Invalid throughput frame axis");
    const auto frames = input.shape.at(axis.get<size_t>());
    if (frames == 0)
      throw std::invalid_argument("Throughput frame axis must be nonempty");
    return frames;
  }
};

} // namespace holoflow::python
