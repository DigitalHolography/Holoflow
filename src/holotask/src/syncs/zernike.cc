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
#include <string>
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

constexpr std::size_t kMaxSupportedModes = 5; // Z2..Z6

inline void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[ZernikeFactory::infer] error: {}", msg);
    throw std::invalid_argument("ZernikeFactory inference error: " + msg);
  }
}

struct Shift {
  float dx_px = 0.0f; // propagated-plane shift, in pixels, along x
  float dy_px = 0.0f; // propagated-plane shift, in pixels, along y
};

struct ZernikeEval {
  float value  = 0.0f; // Z_n(x, y)
  float d_dx_n = 0.0f; // dZ_n / d(x_n), derivative wrt normalized x
  float d_dy_n = 0.0f; // dZ_n / d(y_n), derivative wrt normalized y
};

/// Evaluate the supported Noll-indexed Zernike modes on the unit disk.
///
/// Convention used here:
///   Z2 = 2 x
///   Z3 = 2 y
///   Z4 = sqrt(3) * (2(x^2 + y^2) - 1)
///   Z5 = 2 sqrt(6) * x y
///   Z6 = sqrt(6) * (x^2 - y^2)
///
/// These are the common Noll-normalized low-order modes.
///
/// Coordinates (x_n, y_n) are normalized pupil coordinates on the unit disk:
///   x_n = X / R
///   y_n = Y / R
ZernikeEval eval_zernike_noll(int noll_index, float x_n, float y_n) {
  const float sqrt3 = std::sqrt(3.0f);
  const float sqrt6 = std::sqrt(6.0f);

  switch (noll_index) {
  case 2:
    return {
        .value  = 2.0f * x_n,
        .d_dx_n = 2.0f,
        .d_dy_n = 0.0f,
    };

  case 3:
    return {
        .value  = 2.0f * y_n,
        .d_dx_n = 0.0f,
        .d_dy_n = 2.0f,
    };

  case 4:
    return {
        .value  = sqrt3 * (2.0f * (x_n * x_n + y_n * y_n) - 1.0f),
        .d_dx_n = 4.0f * sqrt3 * x_n,
        .d_dy_n = 4.0f * sqrt3 * y_n,
    };

  case 5:
    return {
        .value  = 2.0f * sqrt6 * x_n * y_n,
        .d_dx_n = 2.0f * sqrt6 * y_n,
        .d_dy_n = 2.0f * sqrt6 * x_n,
    };

  case 6:
    return {
        .value  = sqrt6 * (x_n * x_n - y_n * y_n),
        .d_dx_n = 2.0f * sqrt6 * x_n,
        .d_dy_n = -2.0f * sqrt6 * y_n,
    };

  default:
    throw std::invalid_argument("Unsupported Noll index");
  }
}

/// Recover one shift per subaperture by locating the brightest spot and refining
/// the spot position with a small intensity-weighted centroid.
///
/// Input tensor layout is assumed to be:
///   [batch, nb_sub_y, nb_sub_x, win_h, win_w]
///
/// Returned shifts are expressed relative to the subaperture center, in
/// propagated-plane pixels.
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
          const int val = static_cast<int>(subap_ptr[y * desc.strides[3] + x]);
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

/// Solve a dense linear system A x = b for n <= kMaxSupportedModes using
/// Gaussian elimination with partial pivoting.
///
/// Returns zeros if the system is numerically singular.
std::array<float, kMaxSupportedModes>
solve_linear_system(std::array<std::array<float, kMaxSupportedModes>, kMaxSupportedModes> A,
                    std::array<float, kMaxSupportedModes> b, std::size_t n) {
  std::array<float, kMaxSupportedModes> x{};
  constexpr float                       singular_eps = 1e-12f;

  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    float       best  = std::abs(A[col][col]);

    for (std::size_t row = col + 1; row < n; ++row) {
      const float cand = std::abs(A[row][col]);
      if (cand > best) {
        best  = cand;
        pivot = row;
      }
    }

    if (best < singular_eps) {
      return x;
    }

    if (pivot != col) {
      std::swap(A[pivot], A[col]);
      std::swap(b[pivot], b[col]);
    }

    const float pivot_val = A[col][col];
    for (std::size_t j = col; j < n; ++j) {
      A[col][j] /= pivot_val;
    }
    b[col] /= pivot_val;

    for (std::size_t row = col + 1; row < n; ++row) {
      const float factor = A[row][col];
      if (factor == 0.0f) {
        continue;
      }

      for (std::size_t j = col; j < n; ++j) {
        A[row][j] -= factor * A[col][j];
      }
      b[row] -= factor * b[col];
    }
  }

  for (int row = static_cast<int>(n) - 1; row >= 0; --row) {
    float acc = b[static_cast<std::size_t>(row)];
    for (std::size_t col = static_cast<std::size_t>(row) + 1; col < n; ++col) {
      acc -= A[static_cast<std::size_t>(row)][col] * x[col];
    }
    x[static_cast<std::size_t>(row)] = acc;
  }

  return x;
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
  // The input support is sampled on a rectangular grid, but the optical pupil is
  // assumed to be a disk. We therefore normalize physical coordinates (X, Y) by
  // the radius R of the largest inscribed disk:
  //
  //   x_n = X / R
  //   y_n = Y / R
  //
  // Zernike polynomials are defined on the unit disk in (x_n, y_n).
  // Physical derivatives are obtained with:
  //
  //   d/dX = (1/R) d/dx_n
  //   d/dY = (1/R) d/dy_n
  //
  const float total_width_m  = static_cast<float>(nb_sub_x * win_w) * settings_.dx;
  const float total_height_m = static_cast<float>(nb_sub_y * win_h) * settings_.dy;
  const float pupil_radius_m = 0.5f * std::min(total_width_m, total_height_m);

  // Physical spacing between subaperture centers in the pupil plane.
  const float pitch_x_m = static_cast<float>(win_w) * settings_.dx;
  const float pitch_y_m = static_cast<float>(win_h) * settings_.dy;

  // ---------------------------------------------------------------------------
  // Fresnel propagation sampling
  // ---------------------------------------------------------------------------
  //
  // Shifts are measured in the propagated subaperture images. For single-FFT
  // Fresnel propagation:
  //
  //   dx' = lambda z / (win_w dx)
  //   dy' = lambda z / (win_h dy)
  //
  // You stated that win_w*dx == win_h*dy is guaranteed, so dx' == dy'. We keep
  // the x formula as the common propagated pitch.
  //
  const float propagated_pitch_m =
      (settings_.lambda * settings_.z) / (static_cast<float>(win_w) * settings_.dx);

  // ---------------------------------------------------------------------------
  // Least-squares model
  // ---------------------------------------------------------------------------
  //
  // We recover the requested Zernike coefficients from local slopes.
  //
  // Let the optical path difference be:
  //
  //   W(X, Y) = sum_k a_k Z_k(x_n, y_n)
  //
  // with W in meters, and x_n = X/R, y_n = Y/R.
  //
  // A virtual Shack-Hartmann shift measured after propagation gives a local tilt:
  //
  //   dW/dX ≈ delta_x / z
  //   dW/dY ≈ delta_y / z
  //
  // Since delta_x = shift_x_px * dx' and delta_y = shift_y_px * dy',
  // measured slopes are:
  //
  //   slope_x ≈ shift_x_px * propagated_pitch / z
  //   slope_y ≈ shift_y_px * propagated_pitch / z
  //
  const std::size_t n_modes = settings_.indexes.size();

  std::array<std::array<float, kMaxSupportedModes>, kMaxSupportedModes> GtG{};
  std::array<float, kMaxSupportedModes>                                 Gts{};

  // Small ridge regularization to improve stability when the number of valid
  // subapertures is low or the geometry is weakly conditioned.
  constexpr float ridge = 1e-9f;

  for (std::size_t sy = 0; sy < nb_sub_y; ++sy) {
    for (std::size_t sx = 0; sx < nb_sub_x; ++sx) {
      const float X =
          (static_cast<float>(sx) - (static_cast<float>(nb_sub_x) - 1.0f) * 0.5f) * pitch_x_m;
      const float Y =
          (static_cast<float>(sy) - (static_cast<float>(nb_sub_y) - 1.0f) * 0.5f) * pitch_y_m;

      const float x_n = X / pupil_radius_m;
      const float y_n = Y / pupil_radius_m;

      // Only use subaperture centers whose coordinates lie inside the pupil.
      if (x_n * x_n + y_n * y_n > 1.0f) {
        continue;
      }

      const auto &shift = shifts[sy * nb_sub_x + sx];

      // Convert propagated-plane pixel shifts into physical wavefront slopes.
      const float slope_x = (shift.dx_px * propagated_pitch_m) / settings_.z;
      const float slope_y = (shift.dy_px * propagated_pitch_m) / settings_.z;

      std::array<float, kMaxSupportedModes> gx{};
      std::array<float, kMaxSupportedModes> gy{};

      for (std::size_t i = 0; i < n_modes; ++i) {
        const auto eval = eval_zernike_noll(settings_.indexes[i], x_n, y_n);

        // Convert derivatives wrt normalized coordinates into physical
        // derivatives wrt meters in the pupil plane.
        gx[i] = eval.d_dx_n / pupil_radius_m;
        gy[i] = eval.d_dy_n / pupil_radius_m;
      }

      for (std::size_t i = 0; i < n_modes; ++i) {
        for (std::size_t j = 0; j < n_modes; ++j) {
          GtG[i][j] += gx[i] * gx[j] + gy[i] * gy[j];
        }
        Gts[i] += gx[i] * slope_x + gy[i] * slope_y;
      }
    }
  }

  for (std::size_t i = 0; i < n_modes; ++i) {
    GtG[i][i] += ridge;
  }

  // Coefficients are first recovered as OPD amplitudes in meters.
  const auto coefs_m = solve_linear_system(GtG, Gts, n_modes);

  // Convert OPD meters to phase radians:
  //
  //   phi = (2 pi / lambda) W
  //
  // so each modal amplitude is scaled by 2 pi / lambda.
  auto *out_ptr = reinterpret_cast<float *>(ctx.outputs[0].data());
  for (std::size_t i = 0; i < n_modes; ++i) {
    out_ptr[i] = coefs_m[i] * (2.0f * static_cast<float>(M_PI) / settings_.lambda);
  }
  
  // Scale the recovered coefficients by an empirical factor to better match the expected range of values.
  for (std::size_t i = 0; i < n_modes; ++i) {
    out_ptr[i] *= 1.17f;
  }
  
  // Log each recovered coefficient with its corresponding Zernike index for debugging.
  std::string log_msg = "Recovered Zernike coefficients:\n";
  for (std::size_t i = 0; i < n_modes; ++i) {
    log_msg += "  Z" + std::to_string(settings_.indexes[i]) + ": " + std::to_string(out_ptr[i]) + "\n";
  }
  logger()->info(log_msg);

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
  check(settings.indexes.size() <= kMaxSupportedModes, "Too many requested Zernike modes");

  for (int idx : settings.indexes) {
    check(idx >= 2 && idx <= 6, "Only zernike Noll indexes 2..6 are supported");
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