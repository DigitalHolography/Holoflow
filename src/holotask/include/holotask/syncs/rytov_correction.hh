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

struct RytovCorrectionSettings {
  int   polynomial_degree = 5;
  int   sample_step       = 5;
  int   background_margin = 5;
  int   mask_threshold    = 45;
  bool  unwrap_phase      = true;
  float amplitude_floor   = 0.01f; ///< Minimum background amplitude relative to frame maximum.
  float damping_global    = 3e-4f; ///< Recovered MTD Volkov gradient regularization.
  float damping_local     = 0.1f;
  int   gradient_margin   = 2; ///< Suppress gradients near mirrored seams and borders.

  bool operator==(const RytovCorrectionSettings &) const = default;
};

void to_json(nlohmann::json &j, const RytovCorrectionSettings &s);
void from_json(const nlohmann::json &j, RytovCorrectionSettings &s);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

class RytovCorrectionFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> inputs,
                                    const nlohmann::json                  &settings) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs
