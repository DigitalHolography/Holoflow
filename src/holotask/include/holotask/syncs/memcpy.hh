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

/// @brief Settings for the memcpy task.
/// @details
/// JSON schema (informal):
/// @code{.json}
/// {
///   "target": "Host|Device"
/// }
/// @endcode
struct MemcpySettings {
  /// @brief Target memory location for the copy operation.
  enum class Target {
    Host,  // Copy to host memory
    Device // Copy to device memory
  };

  Target target; // Destination memory location
};

/// @name JSON serialization
/// @brief nlohmann::json adapters for @ref MemcpySettings and @ref MemcpySettings::Target.
/// @{
void to_json(nlohmann::json &j, const MemcpySettings::Target &t);
void from_json(const nlohmann::json &j, MemcpySettings::Target &t);
void to_json(nlohmann::json &j, const MemcpySettings &ms);
void from_json(const nlohmann::json &j, MemcpySettings &ms);
/// @}

class Memcpy : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Memcpy(const MemcpySettings &settings, cudaStream_t stream);

  friend class MemcpyFactory;

  MemcpySettings settings_;
  cudaStream_t   stream_;
};

class MemcpyFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::syncs