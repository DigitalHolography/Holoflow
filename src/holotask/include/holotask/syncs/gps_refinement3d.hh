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
#include <nlohmann/json.hpp>
#include <string>

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

struct GPSRefinement3DSettings {
  float delta_n_min = -0.5f;
  float delta_n_max = 0.5f;
  float kappa_min   = 0.0f;
  float kappa_max   = 100000.0f;
  int   iterations  = 10;

  bool operator==(const GPSRefinement3DSettings &) const = default;
};

void to_json(nlohmann::json &j, const GPSRefinement3DSettings &s);
void from_json(const nlohmann::json &j, GPSRefinement3DSettings &s);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

class GPSRefinement3DFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> inputs,
                                    const nlohmann::json                  &settings) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs
