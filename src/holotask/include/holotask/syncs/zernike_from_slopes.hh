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

#include <nlohmann/json.hpp>
#include <vector>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

struct ZernikeFromSlopesSettings {
  std::vector<int> indexes;
  float            lambda = 0.0f;
  float            dx     = 0.0f;
  float            dy     = 0.0f;

  size_t subaperture_height = 0;
  size_t subaperture_width  = 0;

  size_t stride_y = 0;
  size_t stride_x = 0;

  size_t ny = 1;
  size_t nx = 1;

  bool skip_subapertures_outside_pupil = true;

  bool operator==(const ZernikeFromSlopesSettings &) const = default;
};

void to_json(nlohmann::json &j, const ZernikeFromSlopesSettings &s);
void from_json(const nlohmann::json &j, ZernikeFromSlopesSettings &s);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

class ZernikeFromSlopesFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs
