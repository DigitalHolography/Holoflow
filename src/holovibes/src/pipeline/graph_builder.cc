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

#include "graph_builder.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <spdlog/fmt/ranges.h>
#include <stdexcept>
#include <utility>

#include "bug.hh"
#include "logger.hh"
#include "settings_loader.hh"

namespace holovibes::pipeline {

namespace {

constexpr float kFlatfieldCutoffConstant = 0.187f;

holotask::syncs::FlatfieldSettings flatfield_settings_from_cutoff_period(float cutoff_period_m,
                                                                         float dy_m, float dx_m) {
  if (cutoff_period_m <= 0.0f || dy_m <= 0.0f || dx_m <= 0.0f) {
    throw std::invalid_argument("flatfield cutoff period and image pitches must be positive");
  }

  // The UI exposes a physical cutoff period, not the Gaussian sigma. The 0.187 factor follows
  // the 50% amplitude transition convention for the Gaussian high-pass
  // H_hp(f) = 1 - exp(-2*pi^2*sigma^2*f^2), so f50 = 0.187 / sigma_px and
  // sigma_px = 0.187 * period_px. This is a convention, not a hard cutoff.
  // Axis order follows image layout: y uses dy on axis -2, x uses dx on axis -1.
  return {
      .sigma_y = kFlatfieldCutoffConstant * cutoff_period_m / dy_m,
      .sigma_x = kFlatfieldCutoffConstant * cutoff_period_m / dx_m,
  };
}

std::pair<float, float> fresnel_1fft_output_pitch(float wavelength_m, float z_m, float dy_in_m,
                                                  float dx_in_m, size_t ny, size_t nx) {
  if (wavelength_m <= 0.0f || dy_in_m <= 0.0f || dx_in_m <= 0.0f || ny == 0 || nx == 0) {
    throw std::invalid_argument(
        "Fresnel output pitch requires positive wavelength, pitch, and shape");
  }

  const float z_abs = std::abs(z_m);
  if (z_abs <= 0.0f) {
    throw std::invalid_argument("Fresnel output pitch requires non-zero propagation distance");
  }
  return {
      wavelength_m * z_abs / (static_cast<float>(ny) * dy_in_m),
      wavelength_m * z_abs / (static_cast<float>(nx) * dx_in_m),
  };
}

std::pair<float, float> post_propagation_pitch(const Settings              &settings,
                                               const holoflow::core::TDesc &desc) {
  if (settings.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION) {
    const auto rank = desc.shape.size();
    return fresnel_1fft_output_pitch(settings.spacial_lambda, settings.spacial_z,
                                     settings.spacial_pixel_size, settings.spacial_pixel_size,
                                     desc.shape.at(rank - 2), desc.shape.at(rank - 1));
  }

  return {settings.spacial_pixel_size, settings.spacial_pixel_size};
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Initialization
// -------------------------------------------------------------------------------------------------

GraphBuilder::GraphBuilder(const Settings &settings, holoflow::core::Registry &registry)
    : GraphBuilderTasks(registry), s_(settings) {}

// -------------------------------------------------------------------------------------------------
// Top-level pipeline
// -------------------------------------------------------------------------------------------------

holoflow::core::GraphSpec GraphBuilder::build() {
  TDesc H     = build_acquisition();
  TDesc H_raw = H;

  if (s_.raw_view || s_.view_type == ViewType::RAW) {
    bool should_exit = build_raw_view(H);
    if (should_exit) {
      if (s_.recording_method == RecordingMethod::RAW) {
        build_raw_record(H);
      }
      return g_;
    }
  }

  H = build_preprocessing(H);

  if (s_.recording_method == RecordingMethod::RAW) {
    build_raw_record(H_raw);
  }

  TDesc FH = build_time_frequency_analysis(H);

  if (s_.filter_2d) {
    FH = build_spatial_filter(FH);
  }

  if (s_.autofocus_enabled) {
    if (s_.autofocus_nb_iter <= 0) {
      throw std::invalid_argument("autofocus_nb_iter must be positive");
    }

    ShackHartmannIterationState shack_hartmann_iteration_state;
    for (int pass = 0; pass < s_.autofocus_nb_iter; ++pass) {
      FH = build_shack_hartmann(FH, pass == s_.autofocus_nb_iter - 1,
                                shack_hartmann_iteration_state);
    }
  }

  TDesc FH_z = build_spatial_propagation(FH);

  build_xy_view(FH_z);

  if (s_.view_3d_cuts) {
    build_3d_cuts(FH_z);
  }

  return g_;
}

// -------------------------------------------------------------------------------------------------
// Pipeline stages
// -------------------------------------------------------------------------------------------------

GraphBuilder::TDesc GraphBuilder::build_acquisition() {
  auto cam_path = s_.camera_config_path.string();

  if (s_.import_source == ImportSource::HOLOFILE) {
    return holofile_read({
        .path        = s_.load_path.string(),
        .load_kind   = load_method_map_.at(s_.load_method),
        .start_frame = s_.load_begin,
        .end_frame   = s_.load_end,
        .batch_size  = s_.load_batch,
        .max_fps     = s_.load_fps_limit,
        .keep_cursor = false,
    });
  } else if (s_.import_source == ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO) {
    return ametek_s710_euresys_coaxlink_octo({cam_path});
  } else if (s_.import_source == ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP) {
    return ametek_s711_euresys_coaxlink_qsfp_plus({cam_path});
  }

  HOLOVIBES_UNREACHABLE();
}

void GraphBuilder::build_raw_record(const TDesc &H) {
  // auto Host = holotask::syncs::MemcpySettings::Target::Host;

  // auto H_rec = memcpy(H, {Host});
  // H_rec      = batched_queue(H_rec, {s_.recording_count, s_.time_window, s_.time_window});

  holofile_write(H, {
                        s_.recording_path.string(),
                        s_.recording_count,
                        settings_to_old_json(s_),
                        true,
                    });
}

bool GraphBuilder::build_raw_view(const TDesc &H) {
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  int64_t new_y = static_cast<int64_t>(H.shape.at(1));
  int64_t new_x = static_cast<int64_t>(H.shape.at(2));
  (void)new_y;
  (void)new_x;

  auto H_disp     = memcpy(H, {Host});
  auto H_view     = batched_queue(H_disp, {s_.cpu_out_size, 1, 1});
  auto H_reshaped = H_view;
  // auto H_reshaped = reshape(H_view, {{1, new_y, new_x}, true});

  if (s_.raw_view) {
    xy_raw_display(H_reshaped, {});
  }

  if (s_.view_type == ViewType::RAW) {
    xy_processed_display(H_reshaped, {});
    return true; // Signal to caller to exit pipeline early
  }

  return false;
}

GraphBuilder::TDesc GraphBuilder::build_preprocessing(TDesc H) {
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Device  = holotask::syncs::MemcpySettings::Target::Device;

  if (s_.load_method != LoadMethod::LOAD_IN_GPU) {
    H = memcpy(H, {Device});
    H = batched_queue(H, {s_.gpu_in_size, s_.time_window, s_.time_window});
  }

  return convert(H, {Target::F32, Strat::Real});
}

GraphBuilder::TDesc GraphBuilder::build_time_frequency_analysis(TDesc H) {
  // H enters as [T, Hy, Hx] (F32).
  // We first accumulate N_pre such windows into a batch, producing [N_pre, T, Hy, Hx].
  // Time-frequency analysis then operates along axis 1 (the T dimension).
  // The output is [N_pre, Nz, Hy, Hx], which feeds directly into the post-TFA queue.

  int     N_pre = 4;
  int64_t T     = static_cast<int64_t>(H.shape.at(0));
  int64_t Hy    = static_cast<int64_t>(H.shape.at(1));
  int64_t Hx    = static_cast<int64_t>(H.shape.at(2));

  H = reshape(H, {{1, T, Hy, Hx}, false});
  H = batched_queue(H, {N_pre * 2, N_pre, N_pre}); // → [N_pre, T, Hy, Hx]

  TDesc FH;
  if (s_.time_method == TimeMethod::RFFT) {
    FH = rfft(H, {1}); // axis 1 = T dimension

    // Optimization: slice relevant frequency components early
    if (!s_.view_3d_cuts) {
      FH = slice(FH, {{{}, holonp::SliceRange{s_.time_z_begin, s_.time_z_end}, {}, {}}});
      FH = copy(FH, {});
    }
  }

  else if (s_.time_method == TimeMethod::FFT) {
    FH = fft(H, {1}); // axis 1 = T dimension

    // Optimization: slice relevant components early, including the symmetric negative band
    if (!s_.view_3d_cuts) {
      auto N = T; // number of FFT points along the time axis

      // Positive frequencies: [s_.time_z_begin, s_.time_z_end)
      auto pos_range = holonp::SliceRange{s_.time_z_begin, s_.time_z_end};
      auto FH_pos    = slice(FH, {{{}, pos_range, {}, {}}});
      FH_pos         = copy(FH_pos, {});

      // Negative frequencies: [N - s_.time_z_end, N - s_.time_z_begin)
      auto neg_range = holonp::SliceRange{N - s_.time_z_end, N - s_.time_z_begin};
      auto FH_neg    = slice(FH, {{{}, neg_range, {}, {}}});
      FH_neg         = copy(FH_neg, {});

      FH = concatenate(std::array<TDesc, 2>{FH_pos, FH_neg}, {1}); // concat along freq axis
    }
  }

  else if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS) {
    // PCA natively supports arbitrary leading batch dimensions (rank >= 3).
    // Input H: [N_pre, T, Hy, Hx] — the feature axis is shape[-3] = T.
    // Output FH: [N_pre, Nz, Hy, Hx] where Nz = z1 - z0.
    int z0 = s_.view_3d_cuts ? 0 : s_.time_z_begin;
    int z1 = s_.view_3d_cuts ? static_cast<int>(T) : s_.time_z_end;
    FH     = pca(H, {z0, z1, 1});
  }

  else {
    throw std::logic_error{"Time method is currently not supported in GraphBuilder"};
  }

  // FH is [N_pre, Nz, Hy, Hx] — accumulate pp_accumulation such batches for post-processing.
  return batched_queue(FH, {s_.pp_accumulation * 2, s_.pp_accumulation, s_.pp_accumulation});
}

GraphBuilder::TDesc
GraphBuilder::build_shack_hartmann(TDesc FH, bool is_last_pass,
                                   ShackHartmannIterationState &iteration_state) {
  if (s_.autofocus_nb_subaps <= 0) {
    throw std::invalid_argument("autofocus_nb_subaps must be positive");
  }
  if ((s_.autofocus_nb_subaps % 2) == 0) {
    throw std::invalid_argument("autofocus_nb_subaps must be odd");
  }

  auto nb_subap = static_cast<size_t>(s_.autofocus_nb_subaps);
  auto lam      = s_.spacial_lambda;
  auto dx       = s_.spacial_pixel_size;
  auto dy       = s_.spacial_pixel_size;
  auto z_prop   = s_.spacial_z;

  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Host    = holotask::syncs::MemcpySettings::Target::Host;
  auto Device  = holotask::syncs::MemcpySettings::Target::Device;

  // 1. Spatial Cropping (keep this to ensure perfect divisibility)
  auto subap_w = FH.shape.at(3) / nb_subap;
  auto subap_h = FH.shape.at(2) / nb_subap;
  if (subap_w == 0 || subap_h == 0) {
    throw std::invalid_argument("autofocus_nb_subaps is too large for the current frame size");
  }

  const float pupil_radius_m = 0.5f * std::min(static_cast<float>(subap_w * nb_subap) * dx,
                                               static_cast<float>(subap_h * nb_subap) * dy);

  // 2. Sub-aperture Processing via Short-Time Fresnel
  // We use stride == subap size for non-overlapping Shack-Hartmann windows.
  auto FH_prop = short_time_fresnel_diffraction(
      FH, subap_w, subap_h,  // window dimensions
      subap_w, subap_h,      // strides (non-overlapping)
      lam, dx, dy, z_prop,   // reconstruction parameters
      PhaseReference::GLOBAL // applies the necessary off-axis phase correction
  );
  auto M0 = mean_abs(FH_prop, {{1}, false});
  M0      = mean(M0, {{0}, true});
  M0      = fftshift(M0, {{-2, -1}});

  if (s_.pp_flatfield) {
    const auto [flatfield_dy, flatfield_dx] =
        fresnel_1fft_output_pitch(lam, z_prop, dy, dx, subap_h, subap_w);
    M0 = flatfield(M0, flatfield_settings_from_cutoff_period(s_.pp_flatfield_cutoff_period_m,
                                                             flatfield_dy, flatfield_dx));
  }

  // Cross Correlation with Reference
  int64_t sy_ref = nb_subap / 2;
  int64_t sx_ref = nb_subap / 2;
  auto    M0_ref = slice(M0, {{{}, sy_ref, sx_ref, {}, {}}});
  auto    xcorr =
      cross_correlation2(M0, M0_ref,
                         {
                             {-2, -1},
                             holotask::syncs::FftNorm::Backward,
                             {0.5f, 0.5f, s_.pp_pctclip_radius, s_.pp_pctclip_radius, 0.0f},
                         });
  xcorr = fftshift(xcorr, {{-2, -1}});
  xcorr = normalize(xcorr, {{-2, -1}, 0.0f, 255.0f});

  if (is_last_pass) {
    // Shack-Hartmann Output Processing
    int64_t h          = static_cast<int64_t>(subap_h * nb_subap);
    int64_t w          = static_cast<int64_t>(subap_w * nb_subap);
    auto    M0_sh_disp = normalize(M0, {{-2, -1}, 0.0f, 255.0f});
    M0_sh_disp         = transpose(M0_sh_disp, {{0, 1, 3, 2, 4}});
    M0_sh_disp         = reshape(M0_sh_disp, {{1, h, w}});
    M0_sh_disp         = convert(M0_sh_disp, {Target::U8, Strat::Scaled});
    M0_sh_disp         = batched_queue(M0_sh_disp, {s_.cpu_out_size, 1, 1});
    shack_hartmann_display(M0_sh_disp, {});

    h                    = static_cast<int64_t>(xcorr.shape.at(3) * nb_subap);
    w                    = static_cast<int64_t>(xcorr.shape.at(4) * nb_subap);
    auto xcorr_flattened = convert(xcorr, {Target::U8, Strat::Scaled});
    xcorr_flattened      = transpose(xcorr_flattened, {{0, 1, 3, 2, 4}});
    xcorr_flattened      = reshape(xcorr_flattened, {{1, h, w}});
    xcorr_flattened      = batched_queue(xcorr_flattened, {s_.cpu_out_size, 1, 1});
    shack_hartmann_xcorr_display(xcorr_flattened, {});
  }

  // When no Zernike orders are specified, still display an empty phase map for consistency
  if (s_.autofocus_zernike_orders.empty()) {
    auto ny          = static_cast<size_t>(FH.shape.at(2));
    auto nx          = static_cast<size_t>(FH.shape.at(3));
    auto empty_phase = zeros({{1, ny, nx}, holoflow::core::DType::F32});
    FH               = correct_phase(FH, empty_phase, {});
    if (is_last_pass) {
      zernike_phase_display(empty_phase, {});
    }

    return FH;
  }

  // Zernike & Phase Correction
  int ny             = static_cast<int>(FH.shape.at(2));
  int nx             = static_cast<int>(FH.shape.at(3));
  xcorr              = cuda_stream_synchronize(xcorr, {});
  auto xcorr_zernike = memcpy(xcorr, {Host});

  holotask::syncs::ZernikeSettings zernike_settings{
      s_.autofocus_zernike_orders, lam, dx, dy, z_prop, 1, 1,
  };
  auto zernike_coeffs = zernike(xcorr_zernike, zernike_settings);
  zernike_coeffs      = slice(zernike_coeffs, {{0, 0, {}}});

  auto phase     = zernike_phase(zernike_coeffs, {s_.autofocus_zernike_orders, ny, nx});
  auto phase_gpu = memcpy(phase, {Device});
  FH             = correct_phase(FH, phase_gpu, {});

  auto zernike_coeffs_gpu = memcpy(zernike_coeffs, {Device});

  iteration_state.cumulative_coeffs_gpu =
      iteration_state.cumulative_coeffs_gpu.has_value()
          ? add(*iteration_state.cumulative_coeffs_gpu, zernike_coeffs_gpu, {})
          : zernike_coeffs_gpu;

  iteration_state.cumulative_phase_gpu =
      iteration_state.cumulative_phase_gpu.has_value()
          ? add(*iteration_state.cumulative_phase_gpu, phase_gpu, {})
          : phase_gpu;

  if (is_last_pass) {
    HOLOVIBES_CHECK(iteration_state.cumulative_coeffs_gpu.has_value());
    HOLOVIBES_CHECK(iteration_state.cumulative_phase_gpu.has_value());

    zernike_coefficients_display(*iteration_state.cumulative_coeffs_gpu,
                                 {s_.autofocus_zernike_orders});

    bool a4_included =
        std::find(s_.autofocus_zernike_orders.begin(), s_.autofocus_zernike_orders.end(), 4) !=
        s_.autofocus_zernike_orders.end();

    if (a4_included) {
      zernike_defocus_z_prop(*iteration_state.cumulative_coeffs_gpu,
                             {s_.autofocus_zernike_orders, lam, z_prop, pupil_radius_m});
    }

    auto phase_disp = copy(*iteration_state.cumulative_phase_gpu, {});
    phase_disp      = wrap2pi(phase_disp, {});
    phase_disp      = reshape(phase_disp, {{1, ny, nx}});
    phase_disp      = batched_queue(phase_disp, {s_.cpu_out_size, 1, 1});
    zernike_phase_display(phase_disp, {});
  }

  return FH;
}

GraphBuilder::TDesc GraphBuilder::short_time_fresnel_diffraction(
    const TDesc &field, size_t win_w, size_t win_h, size_t stride_x, size_t stride_y, float lam,
    float dx, float dy, float z_prop, PhaseReference phase_ref, bool skip_phase_shift) {
  return GraphBuilderTasks::short_time_fresnel_diffraction(
      field, {lam, dx, dy, z_prop, win_h, win_w, stride_y, stride_x, phase_ref, skip_phase_shift});
}

GraphBuilder::TDesc GraphBuilder::build_spatial_propagation(const TDesc &FH) {
  if (s_.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION) {
    return fresnel_diffraction(FH, {
                                       s_.spacial_lambda,
                                       s_.spacial_pixel_size,
                                       s_.spacial_pixel_size,
                                       s_.spacial_z,
                                       {-2, -1},
                                   });
  }

  else if (s_.spacial_method == SpacialMethod::ANGULAR_SPECTRUM) {
    if (s_.autofocus_enabled) {
      throw std::logic_error{"Angular Spectrum is not supported with Shack-Hartmann autofocus"};
    }

    return angular_spectrum(FH, {
                                    s_.spacial_lambda,
                                    s_.spacial_pixel_size,
                                    s_.spacial_pixel_size,
                                    s_.spacial_z,
                                    std::nullopt,
                                });
  }

  throw std::logic_error{"Spacial method is currently not supported in GraphBuilder"};
}

GraphBuilder::TDesc GraphBuilder::build_spatial_filter(const TDesc &FH_z) {
  HOLOVIBES_CHECK(s_.filter_2d);
  return filter_2d(FH_z, {
                             s_.filter_r_inner,
                             s_.filter_r_outer,
                             s_.filter_smooth_inner,
                             s_.filter_smooth_outer,
                         });
}

GraphBuilder::TDesc GraphBuilder::build_freq_weights() {
  auto N  = static_cast<double>(s_.time_window);
  auto fs = 37e3;
  auto df = fs / N;
  auto f0 = s_.time_z_begin * df;
  auto f1 = s_.time_z_end * df;

  TDesc freqs;
  if (s_.view_3d_cuts) {
    throw std::logic_error{"Frequency weights are not supported when 3D cuts are enabled"};
  }

  else if (s_.time_method == TimeMethod::FFT) {
    auto freqs_pos = arange({f0, f1, df, holoflow::core::DType::F32});
    auto freqs_neg = arange({f0 - fs, f1 - fs, df, holoflow::core::DType::F32});
    freqs          = concatenate(std::array<TDesc, 2>{freqs_pos, freqs_neg}, {0});
  }

  else if (s_.time_method == TimeMethod::RFFT ||
           s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS) {
    freqs = arange({f0, f1, df, holoflow::core::DType::F32});
  }

  return freqs;
}

void GraphBuilder::build_xy_view(const TDesc &FH_z) {
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Host    = holotask::syncs::MemcpySettings::Target::Host;

  TDesc result;

  if (s_.moment_type == MomentType::M0) {
    result = mean_abs(FH_z, {{-3}, true});
  }

  else if (s_.moment_type == MomentType::M1) {
    auto n_freq   = static_cast<int64_t>(FH_z.shape.at(1));
    auto abs_S    = abs(FH_z, {});
    auto freqs    = reshape(build_freq_weights(), {{1, n_freq, 1, 1}});
    auto weighted = multiply(freqs, abs_S, {});
    result        = mean(weighted, {{-3}, true});
  }

  else if (s_.moment_type == MomentType::M2) {
    auto n_freq   = static_cast<int64_t>(FH_z.shape.at(1));
    auto abs_S    = abs(FH_z, {});
    auto freqs    = reshape(build_freq_weights(), {{1, n_freq, 1, 1}});
    freqs         = multiply(freqs, freqs, {});
    auto weighted = multiply(freqs, abs_S, {});
    result        = mean(weighted, {{-3}, true});
  }

  if (s_.pp_fft_shift) {
    result = fftshift(result, {{-2, -1}});
  }

  if (s_.pp_flatfield) {
    const auto [flatfield_dy, flatfield_dx] = post_propagation_pitch(s_, FH_z);
    result = flatfield(result, flatfield_settings_from_cutoff_period(
                                   s_.pp_flatfield_cutoff_period_m, flatfield_dy, flatfield_dx));
  }

  if (s_.pp_registration) {
    throw std::logic_error{"Registration is currently not supported"};
  }

  result = mean(result, {{0}, false}); // [1, H, W]

  if (s_.pp_convolution) {
    throw std::logic_error{"Convolution is currently not supported"};
  }

  if (s_.pp_pctclip) {
    result = pct_clip(result, {s_.pp_pctclip_lower,
                               s_.pp_pctclip_upper,
                               {0.5f, 0.5f, s_.pp_pctclip_radius, s_.pp_pctclip_radius, 0.0f}});
  }

  result = convert(result, {Target::U8, Strat::Scaled});
  result = batched_queue(result, {s_.gpu_out_size, 1, 1});
  // result = memcpy(result, {Host});
  // result = batched_queue(result, {s_.cpu_out_size, 1, 1});
  xy_processed_display(result, {});

  if (s_.recording_method == RecordingMethod::PROCESSED) {
    auto result_rec = memcpy(result, {Host});
    result_rec      = batched_queue(result_rec, {s_.cpu_out_size, 1, 1});
    holofile_write(result_rec,
                   {s_.recording_path.string(), s_.recording_count, settings_to_old_json(s_)});
  }
}

void GraphBuilder::build_3d_cuts(const TDesc &FH_z) {
  (void)FH_z;
  throw std::logic_error{"3D cuts are currently not supported in GraphBuilder"};
}

} // namespace holovibes::pipeline
