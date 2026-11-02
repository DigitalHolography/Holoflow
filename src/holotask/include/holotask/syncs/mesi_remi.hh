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

#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

enum class MesiEstimator : int { FixedLut = 0, Remi = 1, SpatialRemi = 2 };

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

struct MesiRemiSettings {
  std::vector<int> exposure_frames;
  double           frame_period_seconds = 1.0e-3;
  double           exposure_fraction    = 1.0;
  double           beta                 = 1.0;
  double           rho                  = 1.0;
  double           tau_min_frames       = 0.05;
  double           tau_max_frames       = 100.0;
  int              tau_samples          = 256;
  MesiEstimator    estimator            = MesiEstimator::FixedLut;
  // Zero selects the image-size default used by the sREMI reference implementation.
  int beta_smoothing_size = 0;

  bool operator==(const MesiRemiSettings &) const = default;
};

struct MesiModelTerms {
  double single_decay = 0.0;
  double double_decay = 0.0;
};

// Host-side model values used by the estimator and by ROI curve diagnostics.
MesiModelTerms mesi_model_terms(const MesiRemiSettings &settings, int exposure_frames,
                                double tau_frames);
double mesi_model_k2(const MesiRemiSettings &settings, int exposure_frames, double tau_frames);

void to_json(nlohmann::json &j, const MesiRemiSettings &s);
void from_json(const nlohmann::json &j, MesiRemiSettings &s);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

class MesiRemiFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::ExecutionUpdatePolicy execution_update_policy() const noexcept override {
    return holoflow::core::ExecutionUpdatePolicy::ExplicitInvalidation;
  }

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
