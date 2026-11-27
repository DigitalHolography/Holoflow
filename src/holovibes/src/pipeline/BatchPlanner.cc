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

#include "pipeline/BatchPlanner.hh"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace holovibes::pipeline {

BatchPlanner::BatchPlanner(std::size_t frame_bytes, int target_frames_per_batch,
                           int max_vram_percent, int max_ram_percent,
                           const std::array<int, 6> &memory_share_plan,
                           const BatchPlannerMemory &memory, bool automatic)
    : frame_bytes_(frame_bytes), target_frames_(target_frames_per_batch > 0
                                                    ? static_cast<std::size_t>(target_frames_per_batch)
                                                    : std::clamp<std::size_t>(
                                                          frame_bytes == 0 ? 1 :
                                                              (kTargetBatchBytes + frame_bytes / 2) /
                                                                  frame_bytes,
                                                          1, kMaximumQueueDepth)),
      max_vram_percent_(max_vram_percent), max_ram_percent_(max_ram_percent),
      shares_(memory_share_plan), memory_(memory), automatic_(automatic) {
  if (max_vram_percent_ < 0 || max_vram_percent_ > 100 || max_ram_percent_ < 0 ||
      max_ram_percent_ > 100)
    throw std::invalid_argument("memory limits must be between 0 and 100 percent");
  if (std::any_of(shares_.begin(), shares_.end(), [](int share) { return share < 0; }) ||
      std::accumulate(shares_.begin(), shares_.end(), 0) > 100)
    throw std::invalid_argument("memory share plan must be non-negative and sum to at most 100");
  std::size_t remaining = static_cast<std::size_t>(
      100 - std::accumulate(shares_.begin(), shares_.end(), 0));
  for (std::size_t i = 0; i < remaining; ++i)
    ++shares_[i % shares_.size()];
}

std::size_t BatchPlanner::budget(bool device) const noexcept {
  const std::size_t total = device ? memory_.total_vram : memory_.total_ram;
  const std::size_t used = device ? memory_.used_vram : memory_.used_ram;
  const int percent = device ? max_vram_percent_ : max_ram_percent_;
  const std::size_t allowed = total / 100 * static_cast<std::size_t>(percent) +
                              total % 100 * static_cast<std::size_t>(percent) / 100;
  const std::size_t reserve = device ? 0 : kRamMinimumHeadroom;
  const std::size_t remaining = allowed > used ? allowed - used : 0;
  return remaining > reserve ? remaining - reserve : 0;
}

std::size_t BatchPlanner::queue_budget(bool device, std::size_t share_index) const {
  if (share_index >= shares_.size())
    throw std::out_of_range("memory share index is invalid");
  return budget(device) / 100 * static_cast<std::size_t>(shares_[share_index]) +
         budget(device) % 100 * static_cast<std::size_t>(shares_[share_index]) / 100;
}

std::size_t BatchPlanner::depth_for(const BatchQueueRequest &request) const {
  if (request.share_index >= shares_.size())
    throw std::out_of_range("memory share index is invalid for queue " + request.name);
  if (!automatic_)
    return request.requested_depth;
  const auto share_bytes = queue_budget(request.device, request.share_index);
  const auto slot_cost = request.slot_bytes * std::max<std::size_t>(1, request.slots_per_depth);
  const auto raw = slot_cost ? (share_bytes > request.fixed_extra_bytes
                                    ? (share_bytes - request.fixed_extra_bytes) / slot_cost
                                    : 0)
                             : request.maximum_depth;
  return std::clamp(raw, request.minimum_depth, request.maximum_depth);
}

void BatchPlanner::record(const BatchQueueRequest &request, std::size_t depth,
                          std::size_t allocated_bytes) {
  plans_.push_back({request.name, request.device, depth, allocated_bytes,
                    static_cast<std::size_t>(shares_.at(request.share_index))});
}

std::vector<BatchQueuePlan> BatchPlanner::finish() const {
  std::size_t used_vram = 0;
  std::size_t used_ram = 0;
  for (const auto &entry : plans_)
    (entry.device ? used_vram : used_ram) += entry.allocated_bytes;
  if (used_vram > budget(true) || used_ram > budget(false)) {
    std::ostringstream error;
    error << (automatic_ ? "automatic" : "manual") << " queue plan exceeds memory budget; frame_bytes="
          << frame_bytes_ << ", target_frames_per_batch=" << target_frames_ << ", VRAM requested="
          << used_vram << " budget=" << budget(true) << ", RAM requested=" << used_ram
          << " budget=" << budget(false) << "\nqueue | share | allocated bytes\n";
    for (const auto &entry : plans_)
      error << entry.name << " | " << entry.share_percent << "% | " << entry.allocated_bytes << '\n';
    if (!automatic_)
      throw std::runtime_error(error.str());
    throw std::runtime_error(error.str() + "automatic reduction cannot satisfy the minimum queue depths");
  }
  return plans_;
}

} // namespace holovibes::pipeline
