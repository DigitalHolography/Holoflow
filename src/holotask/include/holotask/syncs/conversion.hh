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

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

struct ConversionSettings {
  enum class Target {
    U8,
    U16,
    F32,
    CF32,
  };

  enum class Strategy { Real, Scaled, Modulus, Argument };

  Target   target;
  Strategy strategy;

  bool operator==(const ConversionSettings &) const = default;
};

void to_json(nlohmann::json &j, const ConversionSettings::Target &t);
void from_json(const nlohmann::json &j, ConversionSettings::Target &t);
void to_json(nlohmann::json &j, const ConversionSettings::Strategy &s);
void from_json(const nlohmann::json &j, ConversionSettings::Strategy &s);
void to_json(nlohmann::json &j, const ConversionSettings &cs);
void from_json(const nlohmann::json &j, ConversionSettings &cs);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

class ConversionFactory : public holoflow::core::ISyncTaskFactory {
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
