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

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <vector>

#include "holoflow/core/tasks.hh"

namespace holovibes::tasks::syncs {

struct SliceSettings {
  struct AxisRange {
    std::optional<size_t> start;
    std::optional<size_t> end;
    std::optional<size_t> step;
  };

  std::vector<AxisRange> ranges;
};

void to_json(nlohmann::json &j, const SliceSettings::AxisRange &ar);
void from_json(const nlohmann::json &j, SliceSettings::AxisRange &ar);
void to_json(nlohmann::json &j, const SliceSettings &ss);
void from_json(const nlohmann::json &j, SliceSettings &ss);

class Slice : public holoflow::core::ISyncTask {
public:
  struct ResolvedRange {
    size_t start;
    size_t step;
    size_t out_size;
  };

  struct ResolvedSliceParams {
    ResolvedRange         dims[3];
    holoflow::core::TDesc output_desc;
  };

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Slice(const ResolvedSliceParams &params, cudaStream_t stream);

  friend class SliceFactory;

  ResolvedSliceParams params_;
  cudaStream_t        stream_;
};

class SliceFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs,
         const nlohmann::json                  &jsettings,
         const holoflow::core::SyncCreateCtx   &ctx) const override;

private:
  Slice::ResolvedSliceParams resolve_params(const SliceSettings         &settings,
                                            const holoflow::core::TDesc &input_desc) const;

  Slice::ResolvedRange resolve_range(const SliceSettings::AxisRange &range,
                                     size_t                          input_dim_size) const;
};

} // namespace holovibes::tasks::syncs