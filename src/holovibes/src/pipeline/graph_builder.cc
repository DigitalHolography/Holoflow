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

#include <spdlog/fmt/ranges.h>

#include "bug.hh"
#include "logger.hh"
#include "settings_loader.hh"

namespace holovibes::pipeline {

// -------------------------------------------------------------------------------------------------
// Initialization
// -------------------------------------------------------------------------------------------------

GraphBuilder::GraphBuilder(const Settings &settings, holoflow::core::Registry &registry)
    : GraphBuilderTasks(registry), s_(settings) {}

// -------------------------------------------------------------------------------------------------
// Top-level pipeline
// -------------------------------------------------------------------------------------------------

holoflow::core::GraphSpec GraphBuilder::build() {
  TDesc H = build_acquisition();

  if (s_.recording_method == RecordingMethod::RAW) {
    build_raw_record(H);
  }

  if (s_.raw_view || s_.view_type == ViewType::RAW) {
    bool should_exit = build_raw_view(H);
    if (should_exit) {
      return g_;
    }
  }

  H = build_preprocessing(H);

  TDesc FH = build_time_frequency_analysis(H);

  if (s_.autofocus_enabled) {
    FH = build_shack_hartmann(FH);
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
        s_.load_path.string(),
        load_method_map_.at(s_.load_method),
        s_.load_begin,
        s_.load_end,
        s_.load_batch,
        false,
    });
  } else if (s_.import_source == ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO) {
    return ametek_s710_euresys_coaxlink_octo({cam_path});
  } else if (s_.import_source == ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP) {
    return ametek_s711_euresys_coaxlink_qsfp_plus({cam_path});
  }

  HOLOVIBES_UNREACHABLE();
}

void GraphBuilder::build_raw_record(const TDesc &H) {
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  auto H_rec = memcpy(H, {Host});
  H_rec      = batched_queue(H_rec, {s_.recording_count, s_.time_window, s_.time_window});

  holofile_write(H_rec, {s_.recording_path.string(), s_.recording_count, settings_to_old_json(s_)});
}

bool GraphBuilder::build_raw_view(const TDesc &H) {
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  int64_t new_y = static_cast<int64_t>(H.shape.at(1));
  int64_t new_x = static_cast<int64_t>(H.shape.at(2));

  auto H_disp     = memcpy(H, {Host});
  auto H_view     = batched_queue(H_disp, {s_.cpu_out_size, 1, 1});
  auto H_reshaped = reshape(H_view, {{1, new_y, new_x}, true});

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
  TDesc FH;
  if (s_.time_method == TimeMethod::SHORT_TIME_FOURIER) {
    FH = rfft(H, {0});

    // Optimization: slice relevant components early
    if (!s_.view_3d_cuts) {
      FH = slice(FH, {{holonp::SliceRange{s_.time_z_begin, s_.time_z_end}, {}, {}}});
      FH = copy(FH, {});
    }
  }

  else if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS) {
    // Optimization: slice relevant components early
    int z0 = s_.view_3d_cuts ? 0 : s_.time_z_begin;
    int z1 = s_.view_3d_cuts ? static_cast<int>(H.shape.at(0)) : s_.time_z_end;
    FH     = pca(H, {z0, z1, 1});
  }

  else {
    throw std::logic_error{"Time method is currently not supported in GraphBuilder"};
  }

  int64_t Nz = static_cast<int64_t>(FH.shape.at(0));
  int64_t Ny = static_cast<int64_t>(FH.shape.at(1));
  int64_t Nx = static_cast<int64_t>(FH.shape.at(2));

  FH = reshape(FH, {{1, Nz, Ny, Nx}, false});
  return batched_queue(FH, {s_.pp_accumulation * 2, s_.pp_accumulation, s_.pp_accumulation});
}

GraphBuilder::TDesc GraphBuilder::build_shack_hartmann(TDesc FH) {
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

  // Spatial Cropping & Fresnel Lens Application
  auto subap_w = FH.shape.at(3) / nb_subap;
  auto subap_h = FH.shape.at(2) / nb_subap;
  if (subap_w == 0 || subap_h == 0) {
    throw std::invalid_argument("autofocus_nb_subaps is too large for the current frame size");
  }
  auto valid_w = subap_w * nb_subap;
  auto valid_h = subap_h * nb_subap;

  holonp::SliceRange x_crop{0, valid_w};
  holonp::SliceRange y_crop{0, valid_h};

  auto FH_cropped = slice(FH, {{{}, {}, y_crop, x_crop}});
  auto n_freq     = FH_cropped.shape.at(1);

  // Organize into sub-aperture groups:
  // (accumulation, freq, subap_y, subap_x, subap_h, subap_w)
  auto FH_6d      = reshape(FH_cropped, {{
                                        (int64_t)s_.pp_accumulation,
                                        (int64_t)n_freq,
                                        (int64_t)nb_subap,
                                        (int64_t)subap_h,
                                        (int64_t)nb_subap,
                                        (int64_t)subap_w,
                                    },
                                         false});
  auto FH_grouped = transpose(FH_6d, {{0, 1, 2, 4, 3, 5}});
  FH_grouped      = ascontiguousarray(FH_grouped, {});

  // Sub-aperture Processing
  // Subapetures have an angle of propagation induced by parallax effects, which manifests as a
  // shift of the focal spot in the Shack-Hartmann image. To accurately recover this shift, we need
  // to apply a corrective phase ramp to each subaperture before propagation. This is
  // equivalent to applying a Fresnel lens that focuses at the expected focal plane, thus ensuring
  // the focal spot is well-defined and can be precisely localized via cross-correlation.
  auto ramps   = shack_hartmann_phase_ramps({
      subap_w,
      subap_h,
      dx,
      dy,
      nb_subap,
      nb_subap,
      z_prop,
      lam,
  });
  FH_grouped   = mul(FH_grouped, ramps, {});
  auto FH_prop = fresnel_diffraction(FH_grouped, {lam, dx, dy, z_prop, {-2, -1}});
  auto M0      = mean_abs(FH_prop, {{1}, false});
  M0           = mean(M0, {{0}, true});
  M0           = fftshift(M0, {{-2, -1}});

  // Cross Correlation with Reference
  int64_t sy_ref = nb_subap / 2;
  int64_t sx_ref = nb_subap / 2;
  auto    M0_ref = slice(M0, {{{}, sy_ref, sx_ref, {}, {}}});
  auto    xcorr =
      cross_correlation2(M0, M0_ref,
                         {
                             {-2, -1},
                             holonp::FftNorm::Backward,
                             {0.5f, 0.5f, s_.pp_pctclip_radius, s_.pp_pctclip_radius, 0.0f},
                         });

  // Shack-Hartmann Output Processing
  int64_t h          = static_cast<int64_t>(valid_h);
  int64_t w          = static_cast<int64_t>(valid_w);
  auto    M0_sh_disp = normalize(M0, {{-2, -1}, 0.0f, 255.0f});
  M0_sh_disp         = transpose(M0_sh_disp, {{0, 1, 3, 2, 4}});
  M0_sh_disp         = reshape(M0_sh_disp, {{1, h, w}});
  M0_sh_disp         = convert(M0_sh_disp, {Target::U8, Strat::Scaled});
  M0_sh_disp         = memcpy(M0_sh_disp, {Host});
  M0_sh_disp         = batched_queue(M0_sh_disp, {s_.cpu_out_size, 1, 1});
  shack_hartmann_display(M0_sh_disp, {});

  h                    = static_cast<int64_t>(xcorr.shape.at(3) * nb_subap);
  w                    = static_cast<int64_t>(xcorr.shape.at(4) * nb_subap);
  auto xcorr_flattened = fftshift(xcorr, {{-2, -1}});
  xcorr_flattened      = normalize(xcorr_flattened, {{-2, -1}, 0.0f, 255.0f});
  xcorr_flattened      = convert(xcorr_flattened, {Target::U8, Strat::Scaled});
  xcorr_flattened      = transpose(xcorr_flattened, {{0, 1, 3, 2, 4}});
  xcorr_flattened      = reshape(xcorr_flattened, {{1, h, w}});
  xcorr_flattened      = batched_queue(xcorr_flattened, {s_.cpu_out_size, 1, 1});
  shack_hartmann_xcorr_display(xcorr_flattened, {});

  // Zernike & Phase Correction
  int ny = static_cast<int>(FH.shape.at(2));
  int nx = static_cast<int>(FH.shape.at(3));

  if (!s_.autofocus_zernike_orders.empty()) {
    auto xcorr_zernike = fftshift(xcorr, {{-2, -1}});
    xcorr_zernike      = normalize(xcorr_zernike, {{-2, -1}, 0.0f, 255.0f});
    xcorr_zernike      = memcpy(xcorr_zernike, {Host});

    holotask::syncs::ZernikeSettings zernike_settings{
        s_.autofocus_zernike_orders, lam, dx, dy, z_prop, 1, 1,
    };
    auto zernike_coeffs = zernike(xcorr_zernike, zernike_settings);
    zernike_coeffs      = slice(zernike_coeffs, {{0, 0, {}}}); // Remove batch dimension
    zernike_coefficients_display(zernike_coeffs, {s_.autofocus_zernike_orders});

    auto phase     = zernike_phase(zernike_coeffs, {s_.autofocus_zernike_orders, ny, nx});
    auto phase_gpu = memcpy(phase, {Device});
    FH             = correct_phase(FH, phase_gpu, {});

    auto phase_disp = wrap2pi(phase_gpu, {});
    phase_disp      = reshape(phase_disp, {{1, ny, nx}});
    phase_disp      = batched_queue(phase_disp, {s_.cpu_out_size, 1, 1});
    zernike_phase_display(phase_disp, {});
  }

  // When no Zernike orders are specified, still display an empty phase map for consistency
  else {
    auto ny          = static_cast<size_t>(FH.shape.at(2));
    auto nx          = static_cast<size_t>(FH.shape.at(3));
    auto empty_phase = zeros({{1, ny, nx}, holoflow::core::DType::F32});
    FH               = correct_phase(FH, empty_phase, {});
    zernike_phase_display(empty_phase, {});
  }

  return FH;
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
    throw std::logic_error{"Angular Spectrum is currently not supported in GraphBuilder"};
  }

  throw std::logic_error{"Spacial method is currently not supported in GraphBuilder"};
}

void GraphBuilder::build_xy_view(const TDesc &FH_z) {
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Host    = holotask::syncs::MemcpySettings::Target::Host;

  auto M0 = mean_abs(FH_z, {{-3}, false});

  if (s_.pp_fft_shift) {
    M0 = fftshift(M0, {{-2, -1}});
  }

  if (s_.pp_registration) {
    throw std::logic_error{"Registration is currently not supported"};
  }

  auto M0_avg = mean(M0, {{0}, true});

  if (s_.pp_convolution) {
    throw std::logic_error{"Convolution is currently not supported"};
  }

  if (s_.pp_pctclip) {
    M0_avg = pct_clip(M0_avg, {s_.pp_pctclip_lower,
                               s_.pp_pctclip_upper,
                               {0.5f, 0.5f, s_.pp_pctclip_radius, s_.pp_pctclip_radius, 0.0f}});
  }

  M0_avg = convert(M0_avg, {Target::U8, Strat::Scaled});
  M0_avg = batched_queue(M0_avg, {s_.gpu_out_size, 1, 1});
  M0_avg = memcpy(M0_avg, {Host});
  M0_avg = batched_queue(M0_avg, {s_.cpu_out_size, 1, 1});
  xy_processed_display(M0_avg, {});

  if (s_.recording_method == RecordingMethod::PROCESSED) {
    auto M0_rec = memcpy(M0_avg, {Host});
    M0_rec      = batched_queue(M0_rec, {s_.cpu_out_size, 1, 1});
    holofile_write(M0_rec,
                   {s_.recording_path.string(), s_.recording_count, settings_to_old_json(s_)});
  }
}

void GraphBuilder::build_3d_cuts(const TDesc &FH_z) {
  (void)FH_z;
  throw std::logic_error{"3D cuts are currently not supported in GraphBuilder"};
}

} // namespace holovibes::pipeline
