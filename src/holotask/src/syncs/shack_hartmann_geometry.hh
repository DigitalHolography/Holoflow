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

#include <cstddef>
#include <vector>

namespace holotask::syncs::detail {

struct ShackHartmannGeometrySettings {
  size_t sy = 0;
  size_t sx = 0;

  size_t subaperture_height = 0;
  size_t subaperture_width  = 0;
  size_t stride_y           = 0;
  size_t stride_x           = 0;

  float dy = 0.0f;
  float dx = 0.0f;

  bool skip_subapertures_outside_pupil = true;
};

struct ShackHartmannSample {
  float x_m    = 0.0f;
  float y_m    = 0.0f;
  float x_n    = 0.0f;
  float y_n    = 0.0f;
  bool  active = false;
};

struct ShackHartmannGeometry {
  float                            pupil_radius_m = 0.0f;
  std::vector<ShackHartmannSample> samples;
};

ShackHartmannGeometry make_shack_hartmann_geometry(const ShackHartmannGeometrySettings &settings);

} // namespace holotask::syncs::detail
