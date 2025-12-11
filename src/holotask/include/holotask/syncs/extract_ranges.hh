// Copyright 2025 Digital Holography Foundation
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

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include <vector>

namespace holotask::syncs {

struct Range {
  std::size_t start;
  std::size_t end; // exclusive, [start, end)
};

void to_json(nlohmann::json &j, const Range &range);
void from_json(const nlohmann::json &j, Range &range);

struct ExtractRangesSettings {
  std::vector<Range> x_ranges; // Ranges to extract along X axis (width/dim2)
  std::vector<Range> y_ranges; // Ranges to extract along Y axis (height/dim1)
  std::vector<Range> z_ranges; // Ranges to extract along Z axis (depth/dim0)
};

void to_json(nlohmann::json &j, const ExtractRangesSettings &settings);
void from_json(const nlohmann::json &j, ExtractRangesSettings &settings);

struct CopyOp {
  cudaPos    src_pos;
  cudaPos    dst_pos;
  cudaExtent extent;
};

class ExtractRanges : public holoflow::core::ISyncTask {
public:
  ExtractRanges(cudaStream_t stream, std::size_t input_width, std::size_t input_height,
                std::size_t output_width, std::size_t output_height,
                std::vector<CopyOp> &&copy_ops);

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  cudaStream_t stream_;
  std::size_t  input_width_;
  std::size_t  input_height_;
  std::size_t  output_width_;
  std::size_t  output_height_;

  std::vector<CopyOp> copy_ops_;
};

std::size_t compute_total_length(const std::vector<Range> &ranges);

bool validate_ranges(const std::vector<Range> &ranges, std::size_t max_size);

std::vector<CopyOp> precompute_copy_ops(const ExtractRangesSettings &settings);

class ExtractRangesFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs