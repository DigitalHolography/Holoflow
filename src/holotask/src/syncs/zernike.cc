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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
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

constexpr std::size_t kMaxSupportedModes = 9; // Noll indices 2..10

inline void check(bool condition, const std::string &msg) {
  if (!condition) {
    logger()->error("[ZernikeFactory::infer] error: {}", msg);
    throw std::invalid_argument("ZernikeFactory inference error: " + msg);
  }
}

struct ShiftPx {
  float dx = 0.0f; // horizontal displacement in propagated-plane pixels
  float dy = 0.0f; // vertical displacement in propagated-plane pixels
};

struct ZernikeEval {
  float value  = 0.0f;
  float d_dx_n = 0.0f; // derivative wrt normalized x = X / R
  float d_dy_n = 0.0f; // derivative wrt normalized y = Y / R
};

/// Evaluate low-order Zernike modes using Noll indexing.
///
/// Supported modes:
///   Z2: x-tilt
///   Z3: y-tilt
///   Z4: defocus
///   Z5: oblique astigmatism
///   Z6: vertical astigmatism
///   Z7: vertical trefoil
///   Z8: vertical coma
///   Z9: horizontal coma
///   Z10: oblique trefoil
///
/// The coordinates (x_n, y_n) are normalized by the pupil radius:
///   x_n = X / R
///   y_n = Y / R
///
/// The formulas below use the common OSA/Noll normalized forms.
/// Reference:
///   https://en.wikipedia.org/wiki/Zernike_polynomials
ZernikeEval eval_zernike_noll(int noll_index, float x_n, float y_n) {
  const float sqrt3 = std::sqrt(3.0f);
  const float sqrt6 = std::sqrt(6.0f);
  const float sqrt8 = std::sqrt(8.0f);

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

  case 7:
    return {
        .value  = sqrt8 * y_n * (3.0f * x_n * x_n - y_n * y_n),
        .d_dx_n = 6.0f * sqrt8 * x_n * y_n,
        .d_dy_n = 3.0f * sqrt8 * (x_n * x_n - y_n * y_n),
    };

  case 8:
    return {
        .value  = sqrt8 * y_n * (3.0f * x_n * x_n + 3.0f * y_n * y_n - 2.0f),
        .d_dx_n = 6.0f * sqrt8 * x_n * y_n,
        .d_dy_n = sqrt8 * (3.0f * x_n * x_n + 9.0f * y_n * y_n - 2.0f),
    };

  case 9:
    return {
        .value  = sqrt8 * x_n * (3.0f * x_n * x_n + 3.0f * y_n * y_n - 2.0f),
        .d_dx_n = sqrt8 * (9.0f * x_n * x_n + 3.0f * y_n * y_n - 2.0f),
        .d_dy_n = 6.0f * sqrt8 * x_n * y_n,
    };

  case 10:
    return {
        .value  = sqrt8 * x_n * (x_n * x_n - 3.0f * y_n * y_n),
        .d_dx_n = 3.0f * sqrt8 * (x_n * x_n - y_n * y_n),
        .d_dy_n = -6.0f * sqrt8 * x_n * y_n,
    };

  default:
    throw std::invalid_argument("Unsupported Noll index");
  }
}

/// Read one float from a strided 2D tensor row/column view.
///
/// `base` points to the first sample of the 1D line.
/// `index` is the logical sample index along the line.
/// `stride_bytes` is the byte stride between two consecutive logical samples.
float load_strided_1d(const float *base, int index, std::ptrdiff_t stride_bytes) {
  const auto *ptr =
      reinterpret_cast<const float *>(reinterpret_cast<const std::uint8_t *>(base) +
                                      static_cast<std::ptrdiff_t>(index) * stride_bytes);
  return *ptr;
}

/// 3-point parabolic interpolation around a discrete maximum.
///
/// If samples are:
///   v_{-1} = f(i - 1)
///   v_0    = f(i)
///   v_{+1} = f(i + 1)
///
/// and `i` is the discrete peak, then the vertex of the parabola passing
/// through those three points lies at:
///
///   delta = 0.5 * (v_{-1} - v_{+1}) / (v_{-1} - 2 v_0 + v_{+1})
///
/// The returned position is `i + delta`.
///
/// This is a standard subpixel peak refinement method used when a signal is
/// locally smooth near its maximum.
///
/// Border peaks or nearly flat curvature fall back to the integer location.
float parabolic_peak_1d(const float *line, int peak_idx, int size, std::ptrdiff_t stride_bytes) {
  if (peak_idx <= 0 || peak_idx >= size - 1) {
    return static_cast<float>(peak_idx);
  }

  const float v_m = load_strided_1d(line, peak_idx - 1, stride_bytes);
  const float v_0 = load_strided_1d(line, peak_idx, stride_bytes);
  const float v_p = load_strided_1d(line, peak_idx + 1, stride_bytes);

  const float     denom = v_m - 2.0f * v_0 + v_p;
  constexpr float kEps  = 1e-9f;

  if (std::abs(denom) <= kEps) {
    return static_cast<float>(peak_idx);
  }

  const float delta = 0.5f * (v_m - v_p) / denom;
  return static_cast<float>(peak_idx) + delta;
}

/// Return the address of element (y, x) inside one subaperture image.
///
/// The subaperture tensor is assumed to have shape [win_h, win_w] and to be
/// addressed using desc.strides[3] and desc.strides[4].
const float *subap_value_ptr(const float *subap_base, int y, int x,
                             const holoflow::core::TDesc &desc) {
  return reinterpret_cast<const float *>(reinterpret_cast<const std::uint8_t *>(subap_base) +
                                         static_cast<std::ptrdiff_t>(y) * desc.strides[3] +
                                         static_cast<std::ptrdiff_t>(x) * desc.strides[4]);
}

/// Input tensor layout:
///   [batch, nb_sub_y, nb_sub_x, win_h, win_w]
///
/// For each propagated subaperture image, we locate the brightest point.
/// The position is then refined independently in x and y with 1D parabolic
/// interpolation around the discrete maximum.
///
/// Returned shifts are relative to the center of the propagated subaperture,
/// in propagated-plane pixels.
std::vector<ShiftPx> recover_spot_shifts(const holoflow::core::TView &view) {
  const auto &desc = view.desc;
  auto       *data = reinterpret_cast<float *>(view.storage->ptr + desc.offset);

  const auto nb_sub_y = static_cast<std::size_t>(desc.shape[1]);
  const auto nb_sub_x = static_cast<std::size_t>(desc.shape[2]);
  const auto win_h    = static_cast<std::size_t>(desc.shape[3]);
  const auto win_w    = static_cast<std::size_t>(desc.shape[4]);

  const float center_y = (static_cast<float>(win_h) - 1.0f) * 0.5f;
  const float center_x = (static_cast<float>(win_w) - 1.0f) * 0.5f;

  std::vector<ShiftPx> shifts;
  shifts.reserve(nb_sub_y * nb_sub_x);

  for (std::size_t sy = 0; sy < nb_sub_y; ++sy) {
    for (std::size_t sx = 0; sx < nb_sub_x; ++sx) {
      auto *subap_base =
          reinterpret_cast<float *>(reinterpret_cast<std::uint8_t *>(data) +
                                    static_cast<std::ptrdiff_t>(sy) * desc.strides[1] +
                                    static_cast<std::ptrdiff_t>(sx) * desc.strides[2]);

      float best_value = -std::numeric_limits<float>::infinity();
      int   peak_y     = 0;
      int   peak_x     = 0;

      for (int y = 0; y < static_cast<int>(win_h); ++y) {
        for (int x = 0; x < static_cast<int>(win_w); ++x) {
          const float value = *subap_value_ptr(subap_base, y, x, desc);
          if (value > best_value) {
            best_value = value;
            peak_y     = y;
            peak_x     = x;
          }
        }
      }

      // Column through the discrete peak: vary y, keep x fixed.
      const auto *peak_column =
          reinterpret_cast<const float *>(reinterpret_cast<const std::uint8_t *>(subap_base) +
                                          static_cast<std::ptrdiff_t>(peak_x) * desc.strides[4]);

      // Row through the discrete peak: vary x, keep y fixed.
      const auto *peak_row =
          reinterpret_cast<const float *>(reinterpret_cast<const std::uint8_t *>(subap_base) +
                                          static_cast<std::ptrdiff_t>(peak_y) * desc.strides[3]);

      const float refined_y =
          parabolic_peak_1d(peak_column, peak_y, static_cast<int>(win_h), desc.strides[3]);

      const float refined_x =
          parabolic_peak_1d(peak_row, peak_x, static_cast<int>(win_w), desc.strides[4]);

      shifts.push_back({
          .dx = refined_x - center_x,
          .dy = refined_y - center_y,
      });
    }
  }

  return shifts;
}

/// Solve a small dense linear system A x = b using Gaussian elimination with
/// partial pivoting.
///
/// This is used for the normal equations of the least-squares fit.
/// Since we only support 5 modes max, a fixed-size stack array is sufficient.
std::array<float, kMaxSupportedModes>
solve_linear_system(std::array<std::array<float, kMaxSupportedModes>, kMaxSupportedModes> A,
                    std::array<float, kMaxSupportedModes> b, std::size_t n) {
  std::array<float, kMaxSupportedModes> x{};
  constexpr float                       kSingularEps = 1e-12f;

  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    float       best  = std::abs(A[col][col]);

    for (std::size_t row = col + 1; row < n; ++row) {
      const float candidate = std::abs(A[row][col]);
      if (candidate > best) {
        best  = candidate;
        pivot = row;
      }
    }

    if (best < kSingularEps) {
      return x;
    }

    if (pivot != col) {
      std::swap(A[pivot], A[col]);
      std::swap(b[pivot], b[col]);
    }

    const float pivot_value = A[col][col];
    for (std::size_t j = col; j < n; ++j) {
      A[col][j] /= pivot_value;
    }
    b[col] /= pivot_value;

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
  const auto shifts = recover_spot_shifts(ctx.inputs[0]);

  const auto &desc     = ctx.inputs[0].desc;
  const auto  nb_sub_y = static_cast<std::size_t>(desc.shape[1]);
  const auto  nb_sub_x = static_cast<std::size_t>(desc.shape[2]);
  const auto  win_h    = static_cast<std::size_t>(desc.shape[3]);
  const auto  win_w    = static_cast<std::size_t>(desc.shape[4]);

  // Physical pitch between neighboring subaperture centers in the pupil plane.
  //
  // Each subaperture spans win_w × win_h pixels before propagation, so its
  // footprint in the original sampling plane is:
  //
  //   pitch_x = win_w * dx
  //   pitch_y = win_h * dy
  //
  // We use these pitches to assign one pupil-plane sample position (X, Y)
  // to each subaperture center.
  const float pitch_x_m = static_cast<float>(win_w) * settings_.dx;
  const float pitch_y_m = static_cast<float>(win_h) * settings_.dy;

  // Approximate pupil diameter from the full subaperture grid extent.
  //
  // This assumes the grid covers the pupil and that the pupil is circular.
  const float total_width_m  = static_cast<float>(nb_sub_x) * pitch_x_m;
  const float total_height_m = static_cast<float>(nb_sub_y) * pitch_y_m;
  const float pupil_radius_m = 0.5f * std::min(total_width_m, total_height_m);

  // Output sampling after 1-FFT Fresnel propagation.
  //
  // For the Fresnel transform implemented with a single FFT, the output-plane
  // sampling is:
  //
  //   dx_out = lambda * z / (N_x * dx_in)
  //   dy_out = lambda * z / (N_y * dy_in)
  //
  // where (dx_in, dy_in) are the input-plane pixel pitches and z is the
  // propagation distance.
  //
  // Reference:
  //   https://en.wikipedia.org/wiki/Fresnel_diffraction
  const float dx_out =
      (settings_.lambda * settings_.z) / (static_cast<float>(win_w) * settings_.dx);

  const float dy_out =
      (settings_.lambda * settings_.z) / (static_cast<float>(win_h) * settings_.dy);

  const std::size_t n_modes = settings_.indexes.size();

  // We solve a least-squares problem of the form:
  //
  //   s ≈ G a
  //
  // where:
  //   - s contains measured local slopes (∂W/∂x, ∂W/∂y),
  //   - a contains the unknown Zernike coefficients in meters,
  //   - G contains the derivatives of the chosen Zernike modes.
  //
  // We accumulate the normal equations:
  //
  //   (G^T G) a = G^T s
  std::array<std::array<float, kMaxSupportedModes>, kMaxSupportedModes> GtG{};
  std::array<float, kMaxSupportedModes>                                 Gts{};

  constexpr float ridge = 1e-9f;

  for (std::size_t sy = 0; sy < nb_sub_y; ++sy) {
    for (std::size_t sx = 0; sx < nb_sub_x; ++sx) {
      auto sx_f       = static_cast<float>(sx);
      auto sy_f       = static_cast<float>(sy);
      auto nb_sub_x_f = static_cast<float>(nb_sub_x);
      auto nb_sub_y_f = static_cast<float>(nb_sub_y);

      // Physical coordinates of the current subaperture center in the pupil plane.
      const float X = (sx_f - (nb_sub_x_f - 1.0f) * 0.5f) * pitch_x_m;
      const float Y = (sy_f - (nb_sub_y_f - 1.0f) * 0.5f) * pitch_y_m;

      // Normalized pupil coordinates.
      const float x_n = X / pupil_radius_m;
      const float y_n = Y / pupil_radius_m;

      // Ignore subapertures outside the inscribed circular pupil.
      if (x_n * x_n + y_n * y_n > 1.0f) {
        continue;
      }

      const auto &shift = shifts[sy * nb_sub_x + sx];

      // Convert spot displacement to wavefront slope.
      //
      // In geometric optics / Shack-Hartmann style reasoning, a local wavefront
      // tilt θ causes a lateral shift:
      //
      //   delta_x = z * θ_x
      //   delta_y = z * θ_y
      //
      // so:
      //
      //   θ_x = delta_x / z
      //   θ_y = delta_y / z
      //
      // Here delta_x and delta_y are measured in the propagated plane, hence:
      //
      //   delta_x = shift.dx * dx_out
      //   delta_y = shift.dy * dy_out
      //
      // The resulting slopes are dimensionless (meters / meter).
      float slope_x = (shift.dx * dx_out) / settings_.z;
      float slope_y = (shift.dy * dy_out) / settings_.z;

      // A phase ramp is induced on subapertures (execpt the center one) because there is a small
      // propagation angle.
      const float slope_ramp_x = X / settings_.z;
      const float slope_ramp_y = Y / settings_.z;
      slope_x += slope_ramp_x;
      slope_y += slope_ramp_y;

      std::array<float, kMaxSupportedModes> gx{};
      std::array<float, kMaxSupportedModes> gy{};

      for (std::size_t i = 0; i < n_modes; ++i) {
        const auto eval = eval_zernike_noll(settings_.indexes[i], x_n, y_n);

        // eval_* gives derivatives wrt normalized coordinates x_n, y_n.
        //
        // Since:
        //   x_n = X / R
        //   y_n = Y / R
        //
        // chain rule gives:
        //   dZ/dX = (dZ/dx_n) * (1/R)
        //   dZ/dY = (dZ/dy_n) * (1/R)
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

  // Small Tikhonov-style diagonal regularization to improve conditioning.
  for (std::size_t i = 0; i < n_modes; ++i) {
    GtG[i][i] += ridge;
  }

  // Coefficients are first recovered as optical path difference amplitudes in meters.
  const auto coefs_m = solve_linear_system(GtG, Gts, n_modes);

  // Convert OPD amplitude [m] to phase amplitude [rad]:
  //
  //   phi = (2π / λ) * W
  //
  // where W is the wavefront / OPD coefficient in meters.
  auto *out_ptr = reinterpret_cast<float *>(ctx.outputs[0].data());
  for (std::size_t i = 0; i < n_modes; ++i) {
    out_ptr[i] = coefs_m[i] * (2.0f * static_cast<float>(M_PI) / settings_.lambda);
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
  check(idesc.dtype == holoflow::core::DType::F32, "Input dtype must be F32");
  check(idesc.rank() == 5, "Input rank must be 5");
  check(idesc.shape[1] == idesc.shape[2], "Input subaperture grid must be square");
  check((idesc.shape[1] % 2) == 1, "Input subaperture grid size must be odd");

  check(settings.lambda > 0.0f, "Wavelength must be > 0");
  check(settings.dx > 0.0f, "Pixel pitch dx must be > 0");
  check(settings.dy > 0.0f, "Pixel pitch dy must be > 0");
  check(settings.z > 0.0f, "Propagation distance z must be > 0");

  check(!settings.indexes.empty(), "indexes must not be empty");
  check(settings.indexes.size() <= kMaxSupportedModes, "Too many requested Zernike modes");

  for (int idx : settings.indexes) {
    check(idx >= 2 && idx <= 10, "Only Noll indexes 2..10 are supported");
  }

  auto unique_indexes = settings.indexes;
  std::sort(unique_indexes.begin(), unique_indexes.end());
  check(std::adjacent_find(unique_indexes.begin(), unique_indexes.end()) == unique_indexes.end(),
        "indexes must be unique");

  holoflow::core::TDesc output_desc({settings.indexes.size()}, holoflow::core::DType::F32,
                                    holoflow::core::MemLoc::Host);

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {output_desc},
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
