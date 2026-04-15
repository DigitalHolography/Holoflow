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
      {"ny", s.ny},           {"nx", s.nx},
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

  // Default to 1 if not provided for backward compatibility
  s.ny = j.value("ny", 1);
  s.nx = j.value("nx", 1);
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

// Evaluate low-order Zernike modes using Noll indexing.
// The formulas below use the common OSA/Noll normalized forms where the
// polynomials are orthogonal over the unit disk: x_n^2 + y_n^2 <= 1.
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

float load_strided_1d(const float *base, int index, std::ptrdiff_t stride_bytes) {
  const auto *ptr =
      reinterpret_cast<const float *>(reinterpret_cast<const std::uint8_t *>(base) +
                                      static_cast<std::ptrdiff_t>(index) * stride_bytes);
  return *ptr;
}

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

const float *subap_value_ptr(const float *subap_base, int y, int x,
                             const holoflow::core::TDesc &desc) {
  return reinterpret_cast<const float *>(reinterpret_cast<const std::uint8_t *>(subap_base) +
                                         static_cast<std::ptrdiff_t>(y) * desc.strides[3] +
                                         static_cast<std::ptrdiff_t>(x) * desc.strides[4]);
}

std::vector<ShiftPx> recover_spot_shifts(const holoflow::core::TView &view) {
  const auto &desc = view.desc;
  auto       *data = reinterpret_cast<float *>(view.storage->ptr + desc.offset);

  const auto nb_sub_y = static_cast<std::size_t>(desc.shape[1]);
  const auto nb_sub_x = static_cast<std::size_t>(desc.shape[2]);
  const auto win_h    = static_cast<std::size_t>(desc.shape[3]);
  const auto win_w    = static_cast<std::size_t>(desc.shape[4]);

  const float center_y = (static_cast<float>(win_h / 2));
  const float center_x = (static_cast<float>(win_w / 2));

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

      const auto *peak_column =
          reinterpret_cast<const float *>(reinterpret_cast<const std::uint8_t *>(subap_base) +
                                          static_cast<std::ptrdiff_t>(peak_x) * desc.strides[4]);

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

// -------------------------------------------------------------------------------------------------
// Zernike task implementation
// -------------------------------------------------------------------------------------------------

class Zernike : public holoflow::core::ISyncTask {
public:
  explicit Zernike(ZernikeSettings settings, cudaStream_t stream)
      : settings_(std::move(settings)), stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto shifts = recover_spot_shifts(ctx.inputs[0]);

    for (std::size_t sy = 0; sy < static_cast<std::size_t>(ctx.inputs[0].desc.shape[1]); ++sy) {
      for (std::size_t sx = 0; sx < static_cast<std::size_t>(ctx.inputs[0].desc.shape[2]); ++sx) {
        const auto &shift = shifts[sy * static_cast<std::size_t>(ctx.inputs[0].desc.shape[2]) + sx];
        logger()->debug("Subaperture ({}, {}): Shift (dx: {:.3f} px, dy: {:.3f} px)", sx, sy,
                        shift.dx, shift.dy);
      }
    }

    const auto &desc     = ctx.inputs[0].desc;
    const auto  nb_sub_y = static_cast<std::size_t>(desc.shape[1]);
    const auto  nb_sub_x = static_cast<std::size_t>(desc.shape[2]);
    const auto  win_h    = static_cast<std::size_t>(desc.shape[3]);
    const auto  win_w    = static_cast<std::size_t>(desc.shape[4]);

    const float pitch_x_m = static_cast<float>(win_w) * settings_.dx;
    const float pitch_y_m = static_cast<float>(win_h) * settings_.dy;

    const float total_width_m  = static_cast<float>(nb_sub_x) * pitch_x_m;
    const float total_height_m = static_cast<float>(nb_sub_y) * pitch_y_m;

    const std::size_t n_modes     = settings_.indexes.size();
    const size_t      ny          = settings_.ny;
    const size_t      nx          = settings_.nx;
    const std::size_t num_regions = static_cast<std::size_t>(ny * nx);

    const float region_w       = total_width_m / static_cast<float>(nx);
    const float region_h       = total_height_m / static_cast<float>(ny);
    const float local_radius_m = 0.5f * std::min(region_w, region_h);

    const float global_pupil_radius_m = 0.5f * std::min(total_width_m, total_height_m);

    const float dx_out =
        (settings_.lambda * settings_.z) / (static_cast<float>(win_w) * settings_.dx);
    const float dy_out =
        (settings_.lambda * settings_.z) / (static_cast<float>(win_h) * settings_.dy);

    for (std::size_t ry = 0; ry < ny; ++ry) {
      for (std::size_t rx = 0; rx < nx; ++rx) {
        std::size_t center_sx = (rx * nb_sub_x / nx) + (nb_sub_x / nx) / 2;
        std::size_t center_sy = (ry * nb_sub_y / ny) + (nb_sub_y / ny) / 2;

        center_sx = std::min(center_sx, nb_sub_x - 1);
        center_sy = std::min(center_sy, nb_sub_y - 1);

        ShiftPx ref_shift = shifts[center_sy * nb_sub_x + center_sx];
        logger()->debug("Region ({}, {}): Reference subaperture at ({}, {}) with shift (dx: {:.3f} "
                        "px, dy: {:.3f} px)",
                        rx, ry, center_sx, center_sy, ref_shift.dx, ref_shift.dy);

        for (std::size_t sy = 0; sy < nb_sub_y; ++sy) {
          for (std::size_t sx = 0; sx < nb_sub_x; ++sx) {
            std::size_t current_rx = std::min(static_cast<std::size_t>(sx * nx / nb_sub_x),
                                              static_cast<std::size_t>(nx - 1));
            std::size_t current_ry = std::min(static_cast<std::size_t>(sy * ny / nb_sub_y),
                                              static_cast<std::size_t>(ny - 1));

            if (current_rx == rx && current_ry == ry) {
              shifts[sy * nb_sub_x + sx].dx -= ref_shift.dx;
              shifts[sy * nb_sub_x + sx].dy -= ref_shift.dy;
            }
          }
        }
      }
    }

    std::vector<std::array<std::array<float, kMaxSupportedModes>, kMaxSupportedModes>> GtG(
        num_regions);
    std::vector<std::array<float, kMaxSupportedModes>> Gts(num_regions);

    for (std::size_t r = 0; r < num_regions; ++r) {
      for (std::size_t i = 0; i < kMaxSupportedModes; ++i) {
        Gts[r][i] = 0.0f;
        for (std::size_t j = 0; j < kMaxSupportedModes; ++j) {
          GtG[r][i][j] = 0.0f;
        }
      }
    }

    constexpr float ridge          = 1e-9f;
    size_t          kept_subaps    = 0;
    size_t          skipped_subaps = 0;

    for (std::size_t sy = 0; sy < nb_sub_y; ++sy) {
      for (std::size_t sx = 0; sx < nb_sub_x; ++sx) {
        auto sx_f       = static_cast<float>(sx);
        auto sy_f       = static_cast<float>(sy);
        auto nb_sub_x_f = static_cast<float>(nb_sub_x);
        auto nb_sub_y_f = static_cast<float>(nb_sub_y);

        const float X = (sx_f - (nb_sub_x_f - 1.0f) * 0.5f) * pitch_x_m;
        const float Y = (sy_f - (nb_sub_y_f - 1.0f) * 0.5f) * pitch_y_m;

        const float x_n_global = X / global_pupil_radius_m;
        const float y_n_global = Y / global_pupil_radius_m;

        if (x_n_global * x_n_global + y_n_global * y_n_global > 1.0f) {
          skipped_subaps++;
          continue;
        }

        kept_subaps++;

        std::size_t rx         = std::min(static_cast<std::size_t>(sx * nx / nb_sub_x),
                                          static_cast<std::size_t>(nx - 1));
        std::size_t ry         = std::min(static_cast<std::size_t>(sy * ny / nb_sub_y),
                                          static_cast<std::size_t>(ny - 1));
        std::size_t region_idx = ry * nx + rx;

        const float local_center_X =
            (static_cast<float>(rx) + 0.5f) * region_w - (total_width_m * 0.5f);
        const float local_center_Y =
            (static_cast<float>(ry) + 0.5f) * region_h - (total_height_m * 0.5f);

        const float x_n_local = (X - local_center_X) / local_radius_m;
        const float y_n_local = (Y - local_center_Y) / local_radius_m;

        const auto &shift   = shifts[sy * nb_sub_x + sx];
        float       slope_x = (shift.dx * dx_out) / settings_.z;
        float       slope_y = (shift.dy * dy_out) / settings_.z;

        std::array<float, kMaxSupportedModes> gx{};
        std::array<float, kMaxSupportedModes> gy{};

        for (std::size_t i = 0; i < n_modes; ++i) {
          const auto eval = eval_zernike_noll(settings_.indexes[i], x_n_local, y_n_local);
          gx[i]           = eval.d_dx_n / local_radius_m;
          gy[i]           = eval.d_dy_n / local_radius_m;
        }

        for (std::size_t i = 0; i < n_modes; ++i) {
          for (std::size_t j = 0; j < n_modes; ++j) {
            GtG[region_idx][i][j] += gx[i] * gx[j] + gy[i] * gy[j];
          }
          Gts[region_idx][i] += gx[i] * slope_x + gy[i] * slope_y;
        }
      }
    }

    logger()->info("Kept {} subapertures, skipped {} outside the global pupil", kept_subaps,
                   skipped_subaps);

    auto *out_ptr = reinterpret_cast<float *>(ctx.outputs[0].data());
    for (std::size_t r = 0; r < num_regions; ++r) {
      for (std::size_t i = 0; i < n_modes; ++i) {
        GtG[r][i][i] += ridge;
      }

      const auto coefs_m = solve_linear_system(GtG[r], Gts[r], n_modes);
      for (std::size_t i = 0; i < n_modes; ++i) {
        out_ptr[r * n_modes + i] =
            coefs_m[i] * (2.0f * static_cast<float>(M_PI) / settings_.lambda);
      }
    }

    size_t center_region_idx = (ny / 2) * nx + (nx / 2);
    for (std::size_t r = 0; r < num_regions; ++r) {
      std::string coef_str;
      for (std::size_t i = 0; i < n_modes; ++i) {
        float center_coef = out_ptr[center_region_idx * n_modes + i];
        float coef_diff   = out_ptr[r * n_modes + i] - center_coef;
        coef_str += fmt::format("Z{}: {:.4e} rad, ", settings_.indexes[i], coef_diff);
      }
      logger()->info("Region ({}, {}): Coeff diff from center region: {}", r % nx, r / nx,
                     coef_str);
    }

    return holoflow::core::OpResult::Ok;
  }

  void                   update_stream(cudaStream_t stream) { stream_ = stream; }
  const ZernikeSettings &settings() const { return settings_; }

private:
  ZernikeSettings settings_;
  cudaStream_t    stream_;
};

// -------------------------------------------------------------------------------------------------
// ZernikeFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
ZernikeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<ZernikeSettings>();

  check(input_descs.size() == 1, "Zernike task must have exactly one input");

  const auto &idesc = input_descs[0];
  check(idesc.mem_loc == holoflow::core::MemLoc::Host, "Input memory location must be Host");
  check(idesc.dtype == holoflow::core::DType::F32, "Input dtype must be F32");
  check(idesc.rank() == 5, "Input rank must be 5");
  check(idesc.shape[0] == 1, "Only batch size 1 is supported");
  check(idesc.shape[1] == idesc.shape[2], "Input subaperture grid must be square");
  check((idesc.shape[1] % 2) == 1, "Input subaperture grid size must be odd");

  check(settings.lambda > 0.0f, "Wavelength must be > 0");
  check(settings.dx > 0.0f, "Pixel pitch dx must be > 0");
  check(settings.dy > 0.0f, "Pixel pitch dy must be > 0");
  check(settings.z > 0.0f, "Propagation distance z must be > 0");

  check(settings.ny > 0, "Grid size ny must be > 0");
  check(settings.nx > 0, "Grid size nx must be > 0");

  check(!settings.indexes.empty(), "indexes must not be empty");
  check(settings.indexes.size() <= kMaxSupportedModes, "Too many requested Zernike modes");

  for (int idx : settings.indexes) {
    check(idx >= 2 && idx <= 10, "Only Noll indexes 2..10 are supported");
  }

  auto unique_indexes = settings.indexes;
  std::sort(unique_indexes.begin(), unique_indexes.end());
  check(std::adjacent_find(unique_indexes.begin(), unique_indexes.end()) == unique_indexes.end(),
        "indexes must be unique");

  // Output shape: [ny, nx, n_modes]
  holoflow::core::TDesc output_desc({settings.ny, settings.nx, settings.indexes.size()},
                                    holoflow::core::DType::F32, holoflow::core::MemLoc::Host);

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
  (void)infer(input_descs, jsettings);

  const auto settings = jsettings.get<ZernikeSettings>();
  return std::make_unique<Zernike>(settings, ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
ZernikeFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                       std::span<const holoflow::core::TDesc>     input_descs,
                       const nlohmann::json                      &jsettings,
                       const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_zernike = dynamic_cast<Zernike *>(old_task.get());
  if (old_zernike == nullptr) {
    return create(input_descs, jsettings, ctx);
  }

  const auto settings = jsettings.get<ZernikeSettings>();
  if (settings == old_zernike->settings()) {
    old_zernike->update_stream(ctx.stream);
    return old_task;
  }

  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
