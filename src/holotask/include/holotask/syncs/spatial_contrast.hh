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

#include <cstdint>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

// Contrast from per-pixel temporal (sum, sum of squares), using the same clipped spatial window
// and population variance as SpatialContrast. Scratch contains one double2 per image pixel.
void enqueue_spatial_contrast_from_moments(const double2 *moments, double2 *scratch,
                                           float *output, int width, int height,
                                           int temporal_window_size, int window_size,
                                           cudaStream_t stream);

enum class SpatialContrastMethod : uint8_t { Kernel, FFT };

// -------------------------------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------------------------------

struct SpatialContrastSettings {
  int window_size = 7;
  int temporal_window_size = 1;
  SpatialContrastMethod method = SpatialContrastMethod::Kernel;

  bool operator==(const SpatialContrastSettings &) const = default;
};

void to_json(nlohmann::json &j, const SpatialContrastSettings &s);
void from_json(const nlohmann::json &j, SpatialContrastSettings &s);

// -------------------------------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------------------------------

class SpatialContrastFactory : public holoflow::core::ISyncTaskFactory {
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
