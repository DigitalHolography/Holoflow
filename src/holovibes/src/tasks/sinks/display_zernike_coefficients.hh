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

#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holovibes::ui {
class AutoFocusWidget;
}

namespace holovibes::tasks::sinks {

// -------------------------------------------------------------------------------------------------
// DisplayZernikeCoefficientsSettings
// -------------------------------------------------------------------------------------------------

struct DisplayZernikeCoefficientsSettings {
  std::vector<int> indexes;
  float            refresh_rate_hz = 30.0f;

  bool operator==(const DisplayZernikeCoefficientsSettings &) const = default;
};

void to_json(nlohmann::json &j, const DisplayZernikeCoefficientsSettings &settings);
void from_json(const nlohmann::json &j, DisplayZernikeCoefficientsSettings &settings);

// -------------------------------------------------------------------------------------------------
// DisplayZernikeCoefficientsFactory
// -------------------------------------------------------------------------------------------------

class DisplayZernikeCoefficientsFactory : public holoflow::core::ISyncTaskFactory {
public:
  explicit DisplayZernikeCoefficientsFactory(holovibes::ui::AutoFocusWidget *widget);

  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

private:
  holovibes::ui::AutoFocusWidget *widget_;
};

} // namespace holovibes::tasks::sinks
