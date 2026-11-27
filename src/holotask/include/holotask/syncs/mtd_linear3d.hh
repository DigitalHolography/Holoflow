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

struct MTDLinear3DReconstructionSettings {
  int    dimension          = 512;
  int    hologram_size      = 1024;
  double wavelength         = 633e-9;
  double pixel_pitch_object = 55e-9;
  double background_index   = 1.49;
  double medium_index       = 1.49;
  float  tukey_alpha        = 0.1f;

  bool operator==(const MTDLinear3DReconstructionSettings &) const = default;
};

void to_json(nlohmann::json &j, const MTDLinear3DReconstructionSettings &s);
void from_json(const nlohmann::json &j, MTDLinear3DReconstructionSettings &s);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

class MTDLinear3DReconstructionFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> inputs,
                                    const nlohmann::json                  &settings) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs
