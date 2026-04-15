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

#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// STFDPhaseReference
// -------------------------------------------------------------------------------------------------

enum class STFDPhaseReference {
  LOCAL,
  GLOBAL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(STFDPhaseReference, {{STFDPhaseReference::LOCAL, "LOCAL"},
                                                  {STFDPhaseReference::GLOBAL, "GLOBAL"}})

// -------------------------------------------------------------------------------------------------
// ShortTimeFresnelDiffractionSettings
// -------------------------------------------------------------------------------------------------

struct ShortTimeFresnelDiffractionSettings {
  float              lambda;
  float              dx;
  float              dy;
  float              z;
  size_t             win_h;
  size_t             win_w;
  size_t             stride_y;
  size_t             stride_x;
  STFDPhaseReference phase_ref        = STFDPhaseReference::LOCAL;
  bool               skip_phase_shift = true;
  std::vector<int>   axes             = {-2, -1};

  bool operator==(const ShortTimeFresnelDiffractionSettings &) const = default;
};

void to_json(nlohmann::json &j, const ShortTimeFresnelDiffractionSettings &s);
void from_json(const nlohmann::json &j, ShortTimeFresnelDiffractionSettings &s);

// -------------------------------------------------------------------------------------------------
// ShortTimeFresnelDiffractionFactory
// -------------------------------------------------------------------------------------------------

class ShortTimeFresnelDiffractionFactory : public holoflow::core::ISyncTaskFactory {
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
