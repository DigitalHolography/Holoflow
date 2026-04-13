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

#include <optional>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

/// @brief Settings for the angular spectrum propagation task.
struct AngularSpectrumSettings {
  /// @brief Optional bandlimiting filter.
  struct Filter {
    int r_inner; ///< Inner radius in pixels.
    int r_outer; ///< Outer radius in pixels.
    int s_inner; ///< Inner slope in pixels.
    int s_outer; ///< Outer slope in pixels.

    bool operator==(const Filter &) const = default;
  };

  float                 lambda; ///< Wavelength [m].
  float                 dx;     ///< Pixel pitch, horizontal [m].
  float                 dy;     ///< Pixel pitch, vertical [m].
  float                 z;      ///< Propagation distance [m].
  std::optional<Filter> filter; ///< Optional bandlimiting filter.

  bool operator==(const AngularSpectrumSettings &) const = default;
};

void to_json(nlohmann::json &j, const AngularSpectrumSettings::Filter &f);
void from_json(const nlohmann::json &j, AngularSpectrumSettings::Filter &f);
void to_json(nlohmann::json &j, const AngularSpectrumSettings &as);
void from_json(const nlohmann::json &j, AngularSpectrumSettings &as);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

class AngularSpectrumFactory : public holoflow::core::ISyncTaskFactory {
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
