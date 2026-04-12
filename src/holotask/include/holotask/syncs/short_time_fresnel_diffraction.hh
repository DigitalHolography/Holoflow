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

#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holotask::syncs {

// Which coordinate frame is used for the quadratic Fresnel lens applied to each window.
enum class STFDPhaseReference {
  // LOCAL — treat each window as an independent on-axis field.  The lens is the standard
  // quadratic lens centered on the window, identical to fresnel_diffraction applied per window.
  LOCAL,

  // GLOBAL — equivalent to applying a single global quadratic lens to the entire field BEFORE
  // sliding-window extraction.  For window at grid position (gx, gy), the combined phase reduces
  // to  π/(λz) * (x_global² + y_global²)  where (x_global, y_global) are the physical coordinates
  // of each pixel in the original field frame.  This is the correct model for plenoptic / tilted-
  // beam propagation (Shack-Hartmann generalisation with arbitrary overlap).
  GLOBAL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(STFDPhaseReference, {{STFDPhaseReference::LOCAL, "LOCAL"},
                                                  {STFDPhaseReference::GLOBAL, "GLOBAL"}})

// Sliding-window Fresnel diffraction (Short-Time Fresnel Transform).
//
// Input:  [..., H, W]  (CF32, device)
// Output: [..., ny_win, nx_win, win_h, win_w]  (CF32, device)
//
// where  ny_win = (H - win_h) / stride_y + 1
//        nx_win = (W - win_w) / stride_x + 1
//
// Implemented as a single fused CUDA operation:
//   • No intermediate "unfolded" buffer — windows are gathered directly from the original field
//     via a cuFFT JIT load callback.
//   • The callback simultaneously applies the appropriate quadratic lens (LOCAL or GLOBAL).
//   • Batch dimensions are folded into the cuFFT plan for maximum GPU occupancy.
struct ShortTimeFresnelDiffractionSettings {
  float              lambda;
  float              dx;
  float              dy;
  float              z;
  size_t             win_h;
  size_t             win_w;
  size_t             stride_y;
  size_t             stride_x;
  STFDPhaseReference phase_ref        = STFDPhaseReference::LOCAL;
  bool               skip_phase_shift = true;
  std::vector<int>   axes             = {-2, -1};

  bool operator==(const ShortTimeFresnelDiffractionSettings &) const = default;
};

void to_json(nlohmann::json &j, const ShortTimeFresnelDiffractionSettings &s);
void from_json(const nlohmann::json &j, ShortTimeFresnelDiffractionSettings &s);

class ShortTimeFresnelDiffraction : public holoflow::core::ISyncTask {
public:
  struct Impl;

  explicit ShortTimeFresnelDiffraction(std::unique_ptr<Impl> impl);
  ~ShortTimeFresnelDiffraction() override;

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

  const holoflow::core::TDesc               &get_idesc() const;
  const ShortTimeFresnelDiffractionSettings &get_settings() const;

  void update_stream(cudaStream_t stream);

private:
  std::unique_ptr<Impl> impl_;
};

class ShortTimeFresnelDiffractionFactory : public holoflow::core::ISyncTaskFactory {
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
