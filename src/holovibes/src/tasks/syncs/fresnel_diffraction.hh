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

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "curaii/nvrtc.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holovibes::tasks {

/// @brief Settings for the Fresnel diffraction propagation task.
/// @details
/// JSON schema (informal):
/// @code{.json}
/// {
///   "lambda": 532e-9,
///   "dx": 3.45e-6,
///   "dy": 3.45e-6,
///   "z": 0.1
/// }
/// @endcode
struct FresnelDiffractionSettings {
  float lambda; ///< Wavelength in meters.
  float dx;     ///< Pixel pitch in meters.
  float dy;     ///< Pixel pitch in meters.
  float z;      ///< Propagation distance in meters.
};

/// @name JSON serialization
/// @brief nlohmann::json adapters for @ref FresnelDiffractionSettings.
/// @{
void to_json(nlohmann::json &j, const FresnelDiffractionSettings &fds);
void from_json(const nlohmann::json &j, FresnelDiffractionSettings &fds);
/// @}

class FresnelDiffraction : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  FresnelDiffraction(const FresnelDiffractionSettings &settings, curaii::CufftHandle &&fft_handle,
                     DevPtr<cuFloatComplex> &&d_lens, DevPtr<void> &&d_caller_info,
                     std::vector<char> &&lto);

  friend class FresnelDiffractionFactory;

  FresnelDiffractionSettings settings_;
  curaii::CufftHandle        fft_handle_;
  DevPtr<cuFloatComplex>     d_lens_;
  DevPtr<void>               d_caller_info_;
  std::vector<char>          lto_;
};

class FresnelDiffractionFactory : public holoflow::core::ISyncTaskFactory {
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

} // namespace holovibes::tasks