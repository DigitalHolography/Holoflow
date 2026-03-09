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

#include "holotask/syncs/zernike.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "logger.hh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace holotask::syncs {

void to_json(nlohmann::json &j, const ZernikeSettings &s) {
  j = nlohmann::json{
      {"indexes", s.indexes}, {"lambda", s.lambda}, {"dx", s.dx}, {"dy", s.dy}, {"z", s.z},
  };
}

void from_json(const nlohmann::json &j, ZernikeSettings &s) {
  s.indexes.clear();

  if (j.contains("indexes")) {
    j.at("indexes").get_to(s.indexes);
  } else if (j.contains("indices")) {
    j.at("indices").get_to(s.indexes);
  }

  j.at("lambda").get_to(s.lambda);
  j.at("dx").get_to(s.dx);
  j.at("dy").get_to(s.dy);
  j.at("z").get_to(s.z);
}

namespace {

inline void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[ZernikeFactory::infer] error: {}", msg);
    throw std::invalid_argument("ZernikeFactory inference error: " + msg);
  }
}

struct Shift {
  float dx_px = 0.0f; // Shift measured in propagated-plane pixels along x
  float dy_px = 0.0f; // Shift measured in propagated-plane pixels along y
};

/// Recover one shift per subaperture by locating the brightest spot and refining
/// its position with a small intensity-weighted centroid.
///
/// Input tensor layout is assumed to be:
///   [batch, nb_sub_y, nb_sub_x, win_h, win_w]
///
/// Returned shifts are expressed in propagated-plane pixels relative to the
/// subaperture center.
std::vector<Shift> recover_shifts(const holoflow::core::TView &view) {
  const auto &desc = view.desc;
  auto       *data = reinterpret_cast<std::uint8_t *>(view.storage->ptr + desc.offset);

  const auto nb_sub_y = static_cast<std::size_t>(desc.shape[1]);
  const auto nb_sub_x = static_cast<std::size_t>(desc.shape[2]);
  const auto win_h    = static_cast<std::size_t>(desc.shape[3]);
  const auto win_w    = static_cast<std::size_t>(desc.shape[4]);

  const float center_y = (static_cast<float>(win_h) - 1.0f) * 0.5f;
  const float center_x = (static_cast<float>(win_w) - 1.0f) * 0.5f;

  std::vector<Shift> shifts;
  shifts.reserve(nb_sub_y * nb_sub_x);

  constexpr int centroid_radius = 2;

  for (std::size_t sy = 0; sy < nb_sub_y; ++sy) {
    for (std::size_t sx = 0; sx < nb_sub_x; ++sx) {
      auto *subap_ptr = data + sy * desc.strides[1] + sx * desc.strides[2];

      int max_val = -1;
      int peak_y  = 0;
      int peak_x  = 0;

      for (int y = 0; y < static_cast<int>(win_h); ++y) {
        for (int x = 0; x < static_cast<int>(win_w); ++x) {
          const auto val = static_cast<int>(subap_ptr[y * desc.strides[3] + x]);
          if (val > max_val) {
            max_val = val;
            peak_y  = y;
            peak_x  = x;
          }
        }
      }

      float sum_val = 0.0f;
      float sum_y   = 0.0f;
      float sum_x   = 0.0f;

      for (int dy = -centroid_radius; dy <= centroid_radius; ++dy) {
        for (int dx = -centroid_radius; dx <= centroid_radius; ++dx) {
          const int iy = peak_y + dy;
          const int ix = peak_x + dx;

          if (iy < 0 || iy >= static_cast<int>(win_h) || ix < 0 || ix >= static_cast<int>(win_w)) {
            continue;
          }

          const float val = static_cast<float>(subap_ptr[iy * desc.strides[3] + ix]);
          sum_val += val;
          sum_y += val * static_cast<float>(iy);
          sum_x += val * static_cast<float>(ix);
        }
      }

      const float final_y = (sum_val > 0.0f) ? (sum_y / sum_val) : static_cast<float>(peak_y);
      const float final_x = (sum_val > 0.0f) ? (sum_x / sum_val) : static_cast<float>(peak_x);

      shifts.push_back({
          .dx_px = final_x - center_x,
          .dy_px = final_y - center_y,
      });
    }
  }

  return shifts;
}

/// Solve the 3x3 linear system M x = b.
/// Returns {0,0,0} if the matrix is numerically singular.
std::array<float, 3> solve3x3(const float M[3][3], const float b[3]) {
  const float det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

  if (std::abs(det) < 1e-9f) {
    return {0.0f, 0.0f, 0.0f};
  }

  const float inv_det = 1.0f / det;

  float inv[3][3];
  inv[0][0] = (M[1][1] * M[2][2] - M[1][2] * M[2][1]) * inv_det;
  inv[0][1] = (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * inv_det;
  inv[0][2] = (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * inv_det;

  inv[1][0] = (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * inv_det;
  inv[1][1] = (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * inv_det;
  inv[1][2] = (M[1][0] * M[0][2] - M[0][0] * M[1][2]) * inv_det;

  inv[2][0] = (M[1][0] * M[2][1] - M[1][1] * M[2][0]) * inv_det;
  inv[2][1] = (M[2][0] * M[0][1] - M[0][0] * M[2][1]) * inv_det;
  inv[2][2] = (M[0][0] * M[1][1] - M[1][0] * M[0][1]) * inv_det;

  std::array<float, 3> x{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      x[i] += inv[i][j] * b[j];
    }
  }

  return x;
}

/// Map subaperture index to normalized pupil coordinate in [-1, 1].
///
/// For a regular grid of subaperture centers:
///   index = 0        -> -1
///   index = count-1  -> +1
///
/// If count == 1, return 0 to avoid division by zero.
float normalized_coord(std::size_t index, std::size_t count) {
  if (count <= 1) {
    return 0.0f;
  }
  return 2.0f * static_cast<float>(index) / static_cast<float>(count - 1) - 1.0f;
}

} // namespace

Zernike::Zernike(const ZernikeSettings &settings) : settings_(settings) {}

holoflow::core::OpResult Zernike::execute(holoflow::core::SyncCtx &ctx) {
  const auto shifts = recover_shifts(ctx.inputs[0]);

  const auto &desc     = ctx.inputs[0].desc;
  const auto  nb_sub_y = static_cast<std::size_t>(desc.shape[1]);
  const auto  nb_sub_x = static_cast<std::size_t>(desc.shape[2]);
  const auto  win_h    = static_cast<std::size_t>(desc.shape[3]);
  const auto  win_w    = static_cast<std::size_t>(desc.shape[4]);

  // ---------------------------------------------------------------------------
  // Geometry of the pupil
  // ---------------------------------------------------------------------------
  //
  // We assume a circular pupil inscribed in the sampled support.
  //
  // Physical support size in the input plane:
  //   width  = nb_sub_x * win_w * dx
  //   height = nb_sub_y * win_h * dy
  //
  // Since the pupil is circular, its radius is the radius of the largest disk
  // fitting inside that support:
  //   R = min(width, height) / 2
  //
  // Zernike polynomials are defined on the unit disk:
  //   x_n = X / R
  //   y_n = Y / R
  //
  // Therefore, physical derivatives satisfy:
  //   d/dX = (1/R) d/dx_n
  //   d/dY = (1/R) d/dy_n
  //
  const float total_width_m  = static_cast<float>(nb_sub_x * win_w) * settings_.dx;
  const float total_height_m = static_cast<float>(nb_sub_y * win_h) * settings_.dy;
  const float pupil_radius_m = std::min(total_width_m, total_height_m) * 0.5f;

  // ---------------------------------------------------------------------------
  // Fresnel propagation sampling
  // ---------------------------------------------------------------------------
  //
  // Shifts are measured in the propagated subaperture image. With single-FFT
  // Fresnel propagation, the propagated sampling pitch is:
  //
  //   dx' = lambda * z / (win_w * dx)
  //   dy' = lambda * z / (win_h * dy)
  //
  // You stated that the special case win_w*dx == win_h*dy is guaranteed, so
  // dx' == dy'. We keep one common propagated pitch.
  //
  const float propagated_pitch_m =
      (settings_.lambda * settings_.z) / (static_cast<float>(win_w) * settings_.dx);

  // ---------------------------------------------------------------------------
  // Least-squares system
  // ---------------------------------------------------------------------------
  //
  // We estimate a4, a5, a6 from local slopes.
  //
  // Let W(X,Y) be the optical path difference (OPD) in meters:
  //
  //   W(X,Y) = a4 Z4(x_n,y_n) + a5 Z5(x_n,y_n) + a6 Z6(x_n,y_n)
  //
  // where (x_n, y_n) = (X/R, Y/R).
  //
  // Virtual Shack-Hartmann gives a displacement in the propagated plane.
  // A small displacement delta in that plane corresponds to a local tilt:
  //
  //   slope_x = dW/dX ≈ delta_x / z
  //   slope_y = dW/dY ≈ delta_y / z
  //
  // Since the measured displacement is in propagated-plane pixels:
  //
  //   delta_x = shift_x_px * propagated_pitch_m
  //   delta_y = shift_y_px * propagated_pitch_m
  //
  // Therefore:
  //
  //   slope_x ≈ shift_x_px * propagated_pitch_m / z
  //   slope_y ≈ shift_y_px * propagated_pitch_m / z
  //
  // We fit these slopes against the physical derivatives of the Zernike modes.
  //
  float GtG[3][3] = {};
  float Gts[3]    = {};

  const float sqrt3 = std::sqrt(3.0f);
  const float sqrt6 = std::sqrt(6.0f);

  for (std::size_t sy = 0; sy < nb_sub_y; ++sy) {
    for (std::size_t sx = 0; sx < nb_sub_x; ++sx) {
      const float x_n = normalized_coord(sx, nb_sub_x);
      const float y_n = normalized_coord(sy, nb_sub_y);

      // -----------------------------------------------------------------------
      // Noll-indexed Zernike derivatives on the unit disk
      // -----------------------------------------------------------------------
      //
      // Using the common convention:
      //
      //   Z4 = sqrt(3) * (2(x^2 + y^2) - 1)   defocus
      //   Z5 = sqrt(6) * (2xy)                astigmatism
      //   Z6 = sqrt(6) * (x^2 - y^2)          astigmatism
      //
      // Derivatives with respect to normalized coordinates:
      //
      //   dZ4/dx = 4 sqrt(3) x
      //   dZ4/dy = 4 sqrt(3) y
      //
      //   dZ5/dx = 2 sqrt(6) y
      //   dZ5/dy = 2 sqrt(6) x
      //
      //   dZ6/dx = 2 sqrt(6) x
      //   dZ6/dy = -2 sqrt(6) y
      //
      // Then convert to physical derivatives using:
      //
      //   dZ/dX = (1/R) dZ/dx
      //   dZ/dY = (1/R) dZ/dy
      //
      const float g_dx[3] = {
          (4.0f * sqrt3 * x_n) / pupil_radius_m,
          (2.0f * sqrt6 * y_n) / pupil_radius_m,
          (2.0f * sqrt6 * x_n) / pupil_radius_m,
      };

      const float g_dy[3] = {
          (4.0f * sqrt3 * y_n) / pupil_radius_m,
          (2.0f * sqrt6 * x_n) / pupil_radius_m,
          (-2.0f * sqrt6 * y_n) / pupil_radius_m,
      };

      const auto &shift = shifts[sy * nb_sub_x + sx];

      // Convert propagated-plane pixel shifts into physical slopes dW/dX, dW/dY.
      const float slope_x = (shift.dx_px * propagated_pitch_m) / settings_.z;
      const float slope_y = (shift.dy_px * propagated_pitch_m) / settings_.z;

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          GtG[i][j] += g_dx[i] * g_dx[j] + g_dy[i] * g_dy[j];
        }
        Gts[i] += g_dx[i] * slope_x + g_dy[i] * slope_y;
      }
    }
  }

  // The fitted coefficients are OPD amplitudes in meters.
  auto coefs_m = solve3x3(GtG, Gts);

  // Convert OPD meters to phase radians:
  //   phi = (2 pi / lambda) * W
  std::array<float, 3> coefs_rad{};
  for (int i = 0; i < 3; ++i) {
    coefs_rad[i] = coefs_m[i] * (2.0f * static_cast<float>(M_PI) / settings_.lambda);
  }

  // Output requested coefficients.
  auto *out_ptr = reinterpret_cast<float *>(ctx.outputs[0].data());
  for (std::size_t i = 0; i < settings_.indexes.size(); ++i) {
    const int idx = settings_.indexes[i];
    out_ptr[i]    = coefs_rad[idx - 4]; // a4 -> 0, a5 -> 1, a6 -> 2
  }

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
ZernikeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ZernikeSettings>();

  check(input_descs.size() == 1, "Zernike task must have exactly one input");

  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Host, "Input memory location must be Host");
  check(idesc.dtype == holoflow::core::DType::U8, "Input dtype must be U8");
  check(idesc.rank() == 5, "Input rank must be 5");

  check(settings.lambda > 0.0f, "Wavelength must be > 0");
  check(settings.dx > 0.0f, "Pixel pitch dx must be > 0");
  check(settings.dy > 0.0f, "Pixel pitch dy must be > 0");
  check(settings.z > 0.0f, "Propagation distance z must be > 0");

  check(!settings.indexes.empty(), "indexes must not be empty");
  for (int idx : settings.indexes) {
    check(idx == 4 || idx == 5 || idx == 6, "Only zernike indexes 4, 5, 6 are supported");
  }

  auto uniq = settings.indexes;
  std::sort(uniq.begin(), uniq.end());
  check(std::adjacent_find(uniq.begin(), uniq.end()) == uniq.end(), "indexes must be unique");

  holoflow::core::TDesc odesc({settings.indexes.size()}, holoflow::core::DType::F32,
                              holoflow::core::MemLoc::Host);

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
ZernikeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings,
                       const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)ctx;
  infer(input_descs, jsettings);

  const auto settings = jsettings.get<ZernikeSettings>();
  return std::unique_ptr<holoflow::core::ISyncTask>(new Zernike(settings));
}

} // namespace holotask::syncs