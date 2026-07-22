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

#include "syncs/shack_hartmann_geometry.hh"

#include <algorithm>

namespace holotask::syncs::detail {

ShackHartmannGeometry make_shack_hartmann_geometry(const ShackHartmannGeometrySettings &settings) {
  const float pitch_x_m = static_cast<float>(settings.stride_x) * settings.dx;
  const float pitch_y_m = static_cast<float>(settings.stride_y) * settings.dy;

  const float aperture_width_m =
      static_cast<float>((settings.sx - 1) * settings.stride_x + settings.subaperture_width) *
      settings.dx;
  const float aperture_height_m =
      static_cast<float>((settings.sy - 1) * settings.stride_y + settings.subaperture_height) *
      settings.dy;

  ShackHartmannGeometry geometry;
  geometry.pupil_radius_m = 0.5f * std::min(aperture_width_m, aperture_height_m);
  geometry.samples.reserve(settings.sy * settings.sx);

  const float center_x = (static_cast<float>(settings.sx) - 1.0f) * 0.5f;
  const float center_y = (static_cast<float>(settings.sy) - 1.0f) * 0.5f;

  for (size_t sy = 0; sy < settings.sy; ++sy) {
    for (size_t sx = 0; sx < settings.sx; ++sx) {
      const float x_m = (static_cast<float>(sx) - center_x) * pitch_x_m;
      const float y_m = (static_cast<float>(sy) - center_y) * pitch_y_m;
      const float x_n = x_m / geometry.pupil_radius_m;
      const float y_n = y_m / geometry.pupil_radius_m;

      geometry.samples.push_back({
          .x_m    = x_m,
          .y_m    = y_m,
          .x_n    = x_n,
          .y_n    = y_n,
          .active = !settings.skip_subapertures_outside_pupil || x_n * x_n + y_n * y_n <= 1.0f,
      });
    }
  }

  return geometry;
}

} // namespace holotask::syncs::detail
