// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "holotask/syncs/rytov_correction.hh"
#include <cstdint>
#include <span>
#include <vector>

namespace holotask::syncs::mtd_detail {

struct BackgroundProjection {
  std::vector<int>    indexes;
  std::vector<double> weights; ///< coefficient-major least-squares projection.
};

BackgroundProjection background_projection(std::span<const uint8_t> mask, int side,
                                           const RytovCorrectionSettings &settings);

} // namespace holotask::syncs::mtd_detail
