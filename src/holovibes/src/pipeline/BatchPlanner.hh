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

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace holovibes::pipeline {

// Pipeline memory shares sum to 100%: input 35, time-frequency 15, timing dual-reader 15,
// sliding average 15, processed display 10, recording/output 10. Shares apply to the entire
// corresponding device/host budget, so they are not renormalized over just the GPU input queue.
inline constexpr int kInputQueueSharePercent = 35;
inline constexpr int kTimeFrequencyQueueSharePercent = 15;
inline constexpr int kTimingDualReaderSharePercent = 15;
inline constexpr int kSlidingAverageQueueSharePercent = 15;
inline constexpr int kProcessedDisplayQueueSharePercent = 10;
inline constexpr int kRecordingOutputQueueSharePercent = 10;
inline constexpr std::array<int, 6> kDefaultMemorySharePlan{
    kInputQueueSharePercent, kTimeFrequencyQueueSharePercent, kTimingDualReaderSharePercent,
    kSlidingAverageQueueSharePercent, kProcessedDisplayQueueSharePercent, kRecordingOutputQueueSharePercent};
inline constexpr std::size_t kRamMinimumHeadroom = 256ULL * 1024 * 1024;
inline constexpr std::size_t kMinimumQueueDepth = 1;
inline constexpr std::size_t kMaximumQueueDepth = 4096;

// The target transfer chunk is 64 MiB. At 5 GB/s it represents 12.8 ms; for a frame size F,
// default frames/batch = clamp(round((5 GB/s * 12.8 ms) / F), 1, 4096). Thus each input batch
// approximates a 64 MiB chunk, the scale at which the assumed ~5 GB/s GPU input throughput is
// attainable without making the batching period depend on resolution.
inline constexpr std::size_t kTargetInputBandwidthBytesPerSecond = 5'000'000'000ULL;
inline constexpr std::size_t kDefaultBatchPeriodMicroseconds = 12'800;
inline constexpr std::size_t kTargetBatchBytes =
    kTargetInputBandwidthBytesPerSecond * kDefaultBatchPeriodMicroseconds / 1'000'000;

struct BatchPlannerMemory {
  std::size_t total_vram = 0;
  std::size_t used_vram = 0;
  std::size_t total_ram = 0;
  std::size_t used_ram = 0;
};

struct BatchQueueRequest {
  std::string name;
  bool device = true;
  std::size_t slot_bytes = 0;
  std::size_t requested_depth = 1;
  std::size_t minimum_depth = 1;
  std::size_t maximum_depth = kMaximumQueueDepth;
  std::size_t slots_per_depth = 1;
  std::size_t fixed_extra_bytes = 0;
  std::size_t share_index = 0;
  bool adjustable = true;
};

struct BatchQueuePlan {
  std::string name;
  bool device = true;
  std::size_t depth = 0;
  std::size_t allocated_bytes = 0;
  std::size_t share_percent = 0;
};

class BatchPlanner {
public:
  BatchPlanner(std::size_t frame_bytes, int target_frames_per_batch, int max_vram_percent,
               int max_ram_percent, const std::array<int, 6> &memory_share_plan,
               const BatchPlannerMemory &memory, bool automatic);

  [[nodiscard]] std::size_t target_frames_per_batch() const noexcept { return target_frames_; }
  [[nodiscard]] std::size_t budget(bool device) const noexcept;
  [[nodiscard]] std::size_t queue_budget(bool device, std::size_t share_index) const;
  [[nodiscard]] std::size_t depth_for(const BatchQueueRequest &request) const;
  void record(const BatchQueueRequest &request, std::size_t depth, std::size_t allocated_bytes);
  [[nodiscard]] std::vector<BatchQueuePlan> finish() const;
  [[nodiscard]] const std::array<int, 6> &shares() const noexcept { return shares_; }

private:
  std::size_t frame_bytes_;
  std::size_t target_frames_;
  int max_vram_percent_;
  int max_ram_percent_;
  std::array<int, 6> shares_;
  BatchPlannerMemory memory_;
  bool automatic_;
  std::vector<BatchQueuePlan> plans_;
};

} // namespace holovibes::pipeline
