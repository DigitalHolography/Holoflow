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

namespace holovibes::ui {
class ZernikeHistoryWidget;
}

namespace holovibes::tasks::sinks {

// -------------------------------------------------------------------------------------------------
// DisplaySignalHistorySettings
// -------------------------------------------------------------------------------------------------

struct DisplaySignalHistorySettings {
  std::vector<int> indexes;
  double           time_window_seconds = 8.0;

  // One interval separates two consecutive emitted, valid Zernike coefficient sets. It is
  // acquisition or pipeline time, not processing completion or GUI refresh time.
  double sample_time_seconds = 1.0 / 15.0;

  bool operator==(const DisplaySignalHistorySettings &) const = default;
};

void to_json(nlohmann::json &j, const DisplaySignalHistorySettings &settings);
void from_json(const nlohmann::json &j, DisplaySignalHistorySettings &settings);

// -------------------------------------------------------------------------------------------------
// DisplaySignalHistoryFactory
// -------------------------------------------------------------------------------------------------

class DisplaySignalHistoryFactory : public holoflow::core::ISyncTaskFactory {
public:
  explicit DisplaySignalHistoryFactory(holovibes::ui::ZernikeHistoryWidget *widget);

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
  holovibes::ui::ZernikeHistoryWidget *widget_;
};

} // namespace holovibes::tasks::sinks
