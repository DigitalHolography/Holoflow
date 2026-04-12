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

#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

struct FresnelDiffractionSettings {
  float            lambda;                      ///< Wavelength [m].
  float            dx;                          ///< Pixel pitch, horizontal [m].
  float            dy;                          ///< Pixel pitch, vertical [m].
  float            z;                           ///< Propagation distance [m].
  std::vector<int> axes             = {-2, -1}; ///< Spatial axes (H, W).
  bool             skip_phase_shift = true;     ///< Omit output-plane quadratic phase.

  bool operator==(const FresnelDiffractionSettings &) const = default;
};

void to_json(nlohmann::json &j, const FresnelDiffractionSettings &fds);
void from_json(const nlohmann::json &j, FresnelDiffractionSettings &fds);

// -------------------------------------------------------------------------------------------------
// Task
// -------------------------------------------------------------------------------------------------

class FresnelDiffraction : public holoflow::core::ISyncTask {
public:
  struct Impl;

  explicit FresnelDiffraction(std::unique_ptr<Impl> impl);
  ~FresnelDiffraction() override;

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc      &get_idesc() const;
  const FresnelDiffractionSettings &get_settings() const;

  void update_stream(cudaStream_t stream);

private:
  std::unique_ptr<Impl> impl_;
};

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

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

} // namespace holotask::syncs
