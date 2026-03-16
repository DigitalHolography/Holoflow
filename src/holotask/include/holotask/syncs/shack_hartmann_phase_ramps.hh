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

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <span>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holotask::syncs {

struct ShackHartmannPhaseRampsSettings {
  size_t sub_w;     ///< Width of one subaperture tile in pixels.
  size_t sub_h;     ///< Height of one subaperture tile in pixels.
  float  dx;        ///< Pixel pitch along x in meters.
  float  dy;        ///< Pixel pitch along y in meters.
  size_t nx_subabs; ///< Number of subapertures along x.
  size_t ny_subabs; ///< Number of subapertures along y.
  float  R;         ///< Radius of curvature in meters.
  float  lambda;    ///< Wavelength in meters.

  bool operator==(const ShackHartmannPhaseRampsSettings &other) const {
    return sub_w == other.sub_w && sub_h == other.sub_h && dx == other.dx && dy == other.dy &&
           nx_subabs == other.nx_subabs && ny_subabs == other.ny_subabs && R == other.R &&
           lambda == other.lambda;
  }
};

void to_json(nlohmann::json &j, const ShackHartmannPhaseRampsSettings &s);
void from_json(const nlohmann::json &j, ShackHartmannPhaseRampsSettings &s);

class ShackHartmannPhaseRamps : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const ShackHartmannPhaseRampsSettings &get_settings() const { return settings_; }
  void update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  ShackHartmannPhaseRamps(const ShackHartmannPhaseRampsSettings &settings,
                          DevPtr<cuFloatComplex> &&d_ramps, size_t count, cudaStream_t stream);

  friend class ShackHartmannPhaseRampsFactory;

  ShackHartmannPhaseRampsSettings settings_;
  DevPtr<cuFloatComplex>          d_ramps_;
  size_t                          count_;
  cudaStream_t                    stream_;
};

class ShackHartmannPhaseRampsFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc>     input_descs,
         const nlohmann::json                       &jsettings,
         const holoflow::core::SyncCreateCtx        &ctx) const override;
};

} // namespace holotask::syncs
