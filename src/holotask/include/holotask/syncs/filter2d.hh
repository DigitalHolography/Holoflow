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
#include <optional>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "curaii/nvrtc.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holotask::syncs {

/// @brief Settings for the 2D filter task.
/// @details
/// JSON schema (informal):
/// @code{.json}
/// {
///   "r_inner": 100,
///   "r_outer": 200,
///   "s_inner": 20,
///   "s_outer": 40
/// }
/// @endcode
struct Filter2DSettings {
  int r_inner; ///< Inner radius in pixels.
  int r_outer; ///< Outer radius in pixels.
  int s_inner; ///< Inner slope in pixels.
  int s_outer; ///< Outer slope in pixels.
};

/// @name JSON serialization
/// @brief nlohmann::json adapters for @ref Filter2DSettings.
/// @{
void to_json(nlohmann::json &j, const Filter2DSettings &f);
void from_json(const nlohmann::json &j, Filter2DSettings &f);
/// @}

class Filter2D : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Filter2D(const Filter2DSettings &settings, curaii::CufftHandle &&fwd_plan,
           curaii::CufftHandle &&inv_plan, DevPtr<cuFloatComplex> &&d_filter,
           DevPtr<void> &&d_caller_info, std::vector<char> &&lto);

  friend class Filter2DFactory;

  Filter2DSettings       settings_;
  curaii::CufftHandle    fwd_plan_;
  curaii::CufftHandle    inv_plan_;
  DevPtr<cuFloatComplex> d_filter_;
  DevPtr<void>           d_caller_info_;
  std::vector<char>      lto_;
};

class Filter2DFactory : public holoflow::core::ISyncTaskFactory {
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