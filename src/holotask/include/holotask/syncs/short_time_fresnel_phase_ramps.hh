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

template <typename T> using DevPtrSTF = curaii::unique_device_ptr<T>;

namespace holotask::syncs {

// Computes the corrective complex phase mask for Short-Time Fresnel Diffraction in GLOBAL mode.
//
// For a sliding window grid over a field of size (field_h x field_w), the window at grid position
// (gx, gy) is centered at global physical coordinates (Xc, Yc). The corrective phase applied to
// local window coordinates (x', y') is:
//
//   E_corr(x', y') = exp(i * k/z * (Xc*x' + Yc*y'))   [linear ramp per window]
//                  * exp(i * k/(2z) * (Xc^2 + Yc^2))   [quadratic offset per window]
//
// where k = 2*pi/lambda.  This is equivalent to pre-applying a global quadratic lens centered at
// the origin to the full field before window extraction.
//
// Output shape: [ny_win, nx_win, win_h, win_w] as CF32 (broadcastable over batch dimensions).
struct ShortTimeFresnelPhaseSettings {
  size_t win_w;
  size_t win_h;
  float  dx;
  float  dy;
  size_t nx_win;
  size_t ny_win;
  size_t stride_x;
  size_t stride_y;
  size_t field_w; ///< Original field width  (for global coordinate centering).
  size_t field_h; ///< Original field height (for global coordinate centering).
  float  z;       ///< Propagation distance [m].
  float  lambda;  ///< Wavelength [m].

  bool operator==(const ShortTimeFresnelPhaseSettings &) const = default;
};

void to_json(nlohmann::json &j, const ShortTimeFresnelPhaseSettings &s);
void from_json(const nlohmann::json &j, ShortTimeFresnelPhaseSettings &s);

class ShortTimeFresnelPhaseRamps : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const ShortTimeFresnelPhaseSettings &get_settings() const { return settings_; }
  void                                 update_stream(cudaStream_t s) { stream_ = s; }

private:
  ShortTimeFresnelPhaseRamps(const ShortTimeFresnelPhaseSettings &settings,
                             DevPtrSTF<cuFloatComplex> &&d_ramps, size_t count,
                             cudaStream_t stream);

  friend class ShortTimeFresnelPhaseRampsFactory;

  ShortTimeFresnelPhaseSettings settings_;
  DevPtrSTF<cuFloatComplex>     d_ramps_;
  size_t                        count_;
  cudaStream_t                  stream_;
};

class ShortTimeFresnelPhaseRampsFactory : public holoflow::core::ISyncTaskFactory {
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
