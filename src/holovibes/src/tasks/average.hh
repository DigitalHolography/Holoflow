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

namespace holovibes::tasks {

/// @brief Settings for the average task.
/// @details
/// JSON schema (informal):
/// @code{.json}
/// {
///   "axis": 0,
///   "start": 0,
///   "end": 100
/// }
/// @endcode
struct AverageSettings {
  int axis;  ///< Axis along which to average (0 for rows, 1 for columns).
  int start; ///< Starting index (inclusive).
  int end;   ///< Ending index (exclusive).
};

/// @name JSON serialization
/// @brief nlohmann::json adapters for @ref AverageSettings.
/// @{
void to_json(nlohmann::json &j, const AverageSettings &as);
void from_json(const nlohmann::json &j, AverageSettings &as);
/// @}

class Average : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Average(const AverageSettings &settings, cudaStream_t stream);

  friend class AverageFactory;

  AverageSettings settings_;
  cudaStream_t    stream_;
};

class AverageFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holovibes::tasks