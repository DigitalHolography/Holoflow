// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holoflow/core/tasks.hh"
#include <nlohmann/json.hpp>

namespace holotask::syncs {

/// Holographic intensity refocusing in sensor propagation coordinates.
/// Input: one U8/U16 hologram [1,Y,X], host or device. Output: device F32 [Z,Y,X].
/// This is not a refractive-index tomogram or a measurement of anatomical depth.
struct FocusStackSettings {
  double wavelength_m    = 852e-9;
  double pixel_pitch_x_m = 20e-6;
  double pixel_pitch_y_m = 20e-6;
  double z_min_m         = 0.230;
  double z_max_m         = 0.240;
  int    planes          = 128;
  bool   padding         = true;
  bool   isolate_order   = false;
  int    carrier_x       = 0; ///< Offset from DC in the unpadded, shifted Fourier spectrum.
  int    carrier_y       = 0;
  int    order_radius    = 12;
  bool   operator==(const FocusStackSettings &) const = default;
};

void to_json(nlohmann::json &, const FocusStackSettings &);
void from_json(const nlohmann::json &, FocusStackSettings &);

class FocusStackFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,
                                    const nlohmann::json &) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override;
};
} // namespace holotask::syncs
