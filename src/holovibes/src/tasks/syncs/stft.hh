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

#include "curaii/cufft.hh"
#include "holoflow/core/tasks.hh"

namespace holovibes::tasks::syncs {

/// @brief Settings for the short-time Fourier transform (STFT) task.
/// @details
/// JSON schema (informal):
/// @code{.json}
/// {
///   // Currently no settings.
/// }
/// @endcode
struct StftSettings {};

/// @name JSON serialization
/// @brief nlohmann::json adapters for @ref StftSettings.
/// @{
void to_json(nlohmann::json &j, const StftSettings &ss);
void from_json(const nlohmann::json &j, StftSettings &ss);
/// @}

/// @brief Synchronous task that performs STFT on the first axis of a
/// 3D float input tensor.
/// @details
/// Input tensor layout: `[batch_size, height, width]`
/// Output tensor layout: `[batch_size, height, width]`
class Stft : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Stft(const StftSettings &settings, curaii::CufftHandle &&plan);

  friend class StftFactory;

  StftSettings        settings_;
  curaii::CufftHandle plan_;
};

class StftFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holovibes::tasks::syncs