// Copyright 2025 Digital Holography Foundation
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

#include "graph_builder_v2.hh"

#include <spdlog/fmt/ranges.h>

#include "bug.hh"
#include "logger.hh"
#include "settings_loader.hh"

namespace holovibes::pipeline {

// -------------------------------------------------------------------------------------------------
// GraphBuilder Implementation
// -------------------------------------------------------------------------------------------------

GraphBuilder_v2::GraphBuilder_v2(const Settings &settings, holoflow::core::Registry &registry)
    : s_(settings), reg_(registry) {}

holoflow::core::GraphSpec GraphBuilder_v2::build() {
  auto cam_path = s_.camera_config_path.string();

  auto lam    = s_.spacial_lambda;
  auto dx     = s_.spacial_pixel_size;
  auto dy     = s_.spacial_pixel_size;
  auto z_prop = s_.spacial_z;

  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Host    = holotask::syncs::MemcpySettings::Target::Host;
  auto Device  = holotask::syncs::MemcpySettings::Target::Device;

  // -------------------------------------------------------------------------------------------------
  // Acquisition (H - Hologram)
  // -------------------------------------------------------------------------------------------------

  TDesc H;

  if (s_.import_source == ImportSource::HOLOFILE) {
    std::tie(H) =
        unpack<1>(holofile_read({s_.load_path.string(), load_method_map_.at(s_.load_method),
                                 s_.load_begin, s_.load_end, s_.load_batch}));
  } else if (s_.import_source == ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO) {
    std::tie(H) = unpack<1>(ametek_s710_euresys_coaxlink_octo({cam_path}));
  } else if (s_.import_source == ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP) {
    std::tie(H) = unpack<1>(ametek_s711_euresys_coaxlink_qsfp_plus({cam_path}));
  } else {
    HOLOVIBES_UNREACHABLE();
  }

  // Raw record
  if (s_.recording_method == RecordingMethod::RAW) {
    TDesc H_rec;
    auto  count    = s_.recording_count;
    auto  batch    = s_.time_window;
    auto  path     = s_.recording_path.string();
    auto  settings = settings_to_old_json(s_);

    std::tie(H_rec) = unpack<1>(memcpy(H, {Host}));
    std::tie(H_rec) = unpack<1>(batched_queue(H_rec, {count, batch, batch}));
    holofile_write(H_rec, {path, count, settings});
  }

  // Raw view
  if (s_.raw_view || s_.view_type == ViewType::RAW) {
    int64_t new_z = 1;
    int64_t new_y = static_cast<int64_t>(H.shape.at(1));
    int64_t new_x = static_cast<int64_t>(H.shape.at(2));

    auto [H_disp]     = unpack<1>(memcpy(H, {Host}));
    auto [H_view]     = unpack<1>(batched_queue(H_disp, {s_.cpu_out_size, 1, 1}));
    auto [H_reshaped] = unpack<1>(reshape(H_view, {{new_z, new_y, new_x}, true}));

    if (s_.raw_view) {
      xy_raw_display(H_reshaped, {});
    }
    if (s_.view_type == ViewType::RAW) {
      xy_processed_display(H_reshaped, {});
      return g_;
    }
  }

  // GPU transfer and conversion to f32
  if (s_.load_method != LoadMethod::LOAD_IN_GPU) {
    std::tie(H) = unpack<1>(memcpy(H, {Device}));
    std::tie(H) = unpack<1>(batched_queue(H, {s_.gpu_in_size, s_.time_window, s_.time_window}));
  }
  std::tie(H) = unpack<1>(convert(H, {Target::F32, Strat::Real}));

  // -------------------------------------------------------------------------------------------------
  // Time-Frequency Analysis (H -> FH - Frequency Hologram)
  // -------------------------------------------------------------------------------------------------

  TDesc FH;

  if (s_.time_method == TimeMethod::SHORT_TIME_FOURIER) {
    std::tie(FH) = unpack<1>(rfft(H, {0}));

    // Optimization: If we don't need the full 3D volume for cuts, we can select only the relevant
    // z-slices here to reduce computation in the spacial propagation and subsequent nodes.
    if (!s_.view_3d_cuts) {
      holonp::SliceRange freq_slice{s_.time_z_begin, s_.time_z_end};
      std::tie(FH) = unpack<1>(slice(FH, {{freq_slice, {}, {}}}));
    }
  }

  else if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS) {
    auto z0 = 0;
    auto z1 = static_cast<int>(H.shape.at(0));

    // Optimization: If we don't need the full 3D volume for cuts, we can select only the relevant
    // components here to reduce computation in the spacial propagation and subsequent nodes.
    if (!s_.view_3d_cuts) {
      z0 = s_.time_z_begin;
      z1 = s_.time_z_end;
    }

    std::tie(FH) = unpack<1>(pca(H, {z0, z1}));
  }

  else {
    throw std::logic_error{"No time method is currently not supported in GraphBuilder_v2"};
  }

  int     o_batches = s_.pp_accumulation;
  int64_t Nz        = static_cast<int64_t>(FH.shape.at(0));
  int64_t Ny        = static_cast<int64_t>(FH.shape.at(1));
  int64_t Nx        = static_cast<int64_t>(FH.shape.at(2));
  std::tie(FH)      = unpack<1>(reshape(FH, {{1, Nz, Ny, Nx}, false}));
  std::tie(FH)      = unpack<1>(batched_queue(FH, {o_batches * 2, o_batches, o_batches}));

  // Some time methods (e.g. PCA) may yield a F32 output. Convert to CF32 for consistency in
  // downstream processing and display.
  // if (FH.dtype == holoflow::core::DType::F32) {
  //   std::tie(FH) = unpack<1>(convert(FH, {Target::CF32, Strat::Real}));
  // }

  // -------------------------------------------------------------------------------------------------
  // Spacial Propagation (FH -> FH_z - Propagated Hologram)
  // -------------------------------------------------------------------------------------------------

  TDesc FH_z;

  if (s_.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION) {
    std::tie(FH_z) = unpack<1>(fresnel_diffraction(FH, {lam, dx, dy, z_prop, {-2, -1}}));
  }

  else if (s_.spacial_method == SpacialMethod::ANGULAR_SPECTRUM) {
    throw std::logic_error{"Angular Spectrum is currently not supported in GraphBuilder_v2"};
  }

  else {
    throw std::logic_error{"No spacial method is currently not supported in GraphBuilder_v2"};
  }

  if (s_.filter_2d) {
    throw std::logic_error{"2D filtering is currently not supported in GraphBuilder_v2"};
  }

  // -------------------------------------------------------------------------------------------------
  // XY View Processing (FH_z -> M0 - Processed XY View)
  // -------------------------------------------------------------------------------------------------
  {
    auto [M0] = unpack<1>(mean_abs(FH_z, {{-3}, false}));

    if (s_.pp_fft_shift) {
      std::tie(M0) = unpack<1>(fftshift(M0, {{-2, -1}}));
    }

    if (s_.pp_registration) {
      throw std::logic_error{"Registration is currently not supported in GraphBuilder_v2"};
      std::tie(M0) = unpack<1>(registration(M0, {s_.pp_registration_radius}));
    }

    // auto [M0_avg] = unpack<1>(slide_avg(M0, {128, (size_t)s_.pp_accumulation}));
    auto [M0_avg] = unpack<1>(mean(M0, {{0}, true}));

    if (s_.pp_convolution) {
      throw std::logic_error{"Convolution is currently not supported in GraphBuilder_v2"};
      std::tie(M0_avg) =
          unpack<1>(convolution(M0_avg, {s_.pp_convolution_path, s_.pp_convolution_divide}));
    }

    if (s_.pp_pctclip) {
      // auto Ny = M0_avg.shape.at(1);
      // auto Nx = M0_avg.shape.at(2);

      auto cx          = 0.5f;
      auto cy          = 0.5f;
      auto rx          = s_.pp_pctclip_radius;
      auto ry          = s_.pp_pctclip_radius;
      auto angle       = 0.0f;
      std::tie(M0_avg) = unpack<1>(
          pct_clip(M0_avg, {s_.pp_pctclip_lower, s_.pp_pctclip_upper, {cx, cy, rx, ry, angle}}));
    }

    std::tie(M0_avg) = unpack<1>(convert(M0_avg, {Target::U8, Strat::Scaled}));
    std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.gpu_out_size, 1, 1}));
    std::tie(M0_avg) = unpack<1>(memcpy(M0_avg, {Host}));
    std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.cpu_out_size, 1, 1}));
    xy_processed_display(M0_avg, {});

    // XY processed recording
    if (s_.recording_method == RecordingMethod::PROCESSED) {
      auto path     = s_.recording_path;
      auto count    = s_.recording_count;
      auto settings = settings_to_old_json(s_);

      auto [M0_rec]    = unpack<1>(memcpy(M0_avg, {Host}));
      std::tie(M0_rec) = unpack<1>(batched_queue(M0_rec, {s_.cpu_out_size, 1, 1}));
      holofile_write(M0_rec, {path.string(), count, settings});
    }
  }

  if (s_.view_3d_cuts) {
    throw std::logic_error{"3D cuts are currently not supported in GraphBuilder_v2"};
    auto [S] = unpack<1>(abs(FH_z, {}));
    // -------------------------------------------------------------------------------------------------
    // XZ View Processing (S -> M0 - Processed XZ View)
    // -------------------------------------------------------------------------------------------------
    {
      std::vector<size_t> crop_origin = {0, 10, 0};
      std::vector<size_t> crop_shape  = {1, S.shape.at(1) - 20, S.shape.at(0)};
      int64_t             new_z       = 1;
      int64_t             new_y       = static_cast<int64_t>(S.shape.at(1));
      int64_t             new_x       = static_cast<int64_t>(S.shape.at(0));

      auto [M0]        = unpack<1>(average(S, {1, s_.time_y_begin, s_.time_y_end}));
      std::tie(M0)     = unpack<1>(reshape(M0, {{new_z, new_y, new_x}, true}));
      auto [M0_avg]    = unpack<1>(slide_avg(M0, {128, (size_t)s_.pp_accumulation}));
      std::tie(M0_avg) = unpack<1>(crop(M0_avg, {crop_origin, crop_shape}));
      std::tie(M0_avg) = unpack<1>(convert(M0_avg, {Target::U8, Strat::Scaled}));
      std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.gpu_out_size, 1, 1}));
      std::tie(M0_avg) = unpack<1>(memcpy(M0_avg, {Host}));
      std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.cpu_out_size, 1, 1}));
      xz_processed_display(M0_avg, {});
    }

    // -------------------------------------------------------------------------------------------------
    // YZ View Processing (S -> M0 - Processed YZ View)
    // -------------------------------------------------------------------------------------------------
    {
      std::vector<size_t> crop_origin = {0, 10, 0};
      std::vector<size_t> crop_shape  = {1, S.shape.at(1) - 20, S.shape.at(0)};
      int64_t             new_z       = 1;
      int64_t             new_y       = static_cast<int64_t>(S.shape.at(1));
      int64_t             new_x       = static_cast<int64_t>(S.shape.at(0));

      auto [M0]        = unpack<1>(average(S, {2, s_.time_x_begin, s_.time_x_end}));
      std::tie(M0)     = unpack<1>(reshape(M0, {{new_z, new_y, new_x}, true}));
      auto [M0_avg]    = unpack<1>(slide_avg(M0, {128, (size_t)s_.pp_accumulation}));
      std::tie(M0_avg) = unpack<1>(crop(M0_avg, {crop_origin, crop_shape}));
      std::tie(M0_avg) = unpack<1>(convert(M0_avg, {Target::U8, Strat::Scaled}));
      std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.gpu_out_size, 1, 1}));
      std::tie(M0_avg) = unpack<1>(memcpy(M0_avg, {Host}));
      std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.cpu_out_size, 1, 1}));
      yz_processed_display(M0_avg, {});
    }
  }

  // -------------------------------------------------------------------------------------------------
  // Shack-Hartmann View Processing (Vectorized Spatial with Cropping)
  // -------------------------------------------------------------------------------------------------

  if (true) {
    auto nb_subap = 5ULL;

    // -------------------------------------------------------------------------------------------------
    // Spatial Cropping
    // -------------------------------------------------------------------------------------------------

    auto               subap_w = FH.shape.at(3) / nb_subap;
    auto               subap_h = FH.shape.at(2) / nb_subap;
    auto               valid_w = subap_w * nb_subap;
    auto               valid_h = subap_h * nb_subap;
    holonp::SliceRange crop_y{0, valid_h};
    holonp::SliceRange crop_x{0, valid_w};

    auto [FH_cropped] = unpack<1>(slice(FH, {{{}, {}, crop_y, crop_x}}));
    auto n_freq       = FH_cropped.shape.at(1);

    // -------------------------------------------------------------------------------------------------
    // Fresnel Lens Application
    // -------------------------------------------------------------------------------------------------

    auto [z_prop_t] = unpack<1>(asarray({z_prop}));
    auto [Qin]      = unpack<1>(fresnel_qin(z_prop_t, {lam, dx, dy, valid_w, valid_h}));
    auto [FH_Qin]   = unpack<1>(mul(FH_cropped, Qin, {}));

    // -------------------------------------------------------------------------------------------------
    // Sub-aperture Processing
    // -------------------------------------------------------------------------------------------------

    // Reshape to isolate tiles
    // Shape: (Batches, Freq, Grid_Y, Tile_Y, Grid_X, Tile_X)
    auto [FH_6d] =
        unpack<1>(reshape(FH_Qin, {{(int64_t)o_batches, (int64_t)n_freq, (int64_t)nb_subap,
                                    (int64_t)subap_h, (int64_t)nb_subap, (int64_t)subap_w}}));

    // Transpose to group grids
    // Shape: (Batches, Freq, Grid_Y, Grid_X, Tile_Y, Tile_X)
    auto [FH_grouped] = unpack<1>(transpose(FH_6d, {{0, 1, 2, 4, 3, 5}}));

    // 2D FFT on the last two axes (Tile_Y, Tile_X)
    auto [FH_prop] = unpack<1>(fft2(FH_grouped, {{-2, -1}}));

    // Intensity & Averaging
    auto [M0_blocked]    = unpack<1>(mean_abs(FH_prop, {{1}, false}));
    std::tie(M0_blocked) = unpack<1>(mean(M0_blocked, {{0}, true}));
    std::tie(M0_blocked) = unpack<1>(fftshift(M0_blocked, {{-2, -1}}));

    // -------------------------------------------------------------------------------------------------
    // Cross Correlation with Reference (Center Tile)
    // -------------------------------------------------------------------------------------------------

    // int64_t sy_ref = nb_subap / 2;
    // int64_t sx_ref = nb_subap / 2;
    // auto [M0_ref]  = unpack<1>(slice(M0_blocked, {{{}, sy_ref, sx_ref, {}, {}}}));

    // auto [F_ref]       = unpack<1>(rfft2(M0_ref, {{-2, -1}}));
    // auto [F_mov]       = unpack<1>(rfft2(M0_blocked, {{-2, -1}}));
    // auto [F_ref_conj]  = unpack<1>(conj(F_ref, {}));
    // auto [F_xcorr]     = unpack<1>(mul(F_mov, F_ref_conj, {}));
    // auto [F_xcorr_abs] = unpack<1>(abs(F_xcorr, {}));
    // std::tie(F_xcorr)  = unpack<1>(div(F_xcorr, F_xcorr_abs, {}));
    // auto [xcorr]       = unpack<1>(irfft2(F_xcorr, {{-2, -1}}));

    // -------------------------------------------------------------------------------------------------
    // Output
    // -------------------------------------------------------------------------------------------------

    int64_t h = (int64_t)valid_h;
    int64_t w = (int64_t)valid_w;

    auto rescale = [&](const TDesc &input, const std::vector<int> &axis, float a, float b) {
      auto [a_t]       = unpack<1>(asarray({a}));
      auto [b_t]       = unpack<1>(asarray({b}));
      auto [mn]        = unpack<1>(min(input, {axis, true}));
      auto [mx]        = unpack<1>(max(input, {axis, true}));
      auto [b_m_a]     = unpack<1>(sub(b_t, a_t, {}));
      auto [denom]     = unpack<1>(sub(mx, mn, {}));
      auto [zero]      = unpack<1>(asarray({0.0f}));
      auto [mask_zero] = unpack<1>(equal(denom, zero, {}));
      auto [scale]     = unpack<1>(div(b_m_a, denom, {}));
      std::tie(scale)  = unpack<1>(where(mask_zero, zero, scale, {}));
      auto [x_m_mn]    = unpack<1>(sub(input, mn, {}));
      auto [scaled]    = unpack<1>(mul(x_m_mn, scale, {}));
      return unpack<1>(add(scaled, a_t, {}));
    };

    // auto [M0_scaled]     = rescale(M0_blocked, {-2, -1}, 0.0f, 255.0f);
    auto M0_scaled = M0_blocked;
    logger()->info("M0_blocked shape: {}", M0_blocked.shape);
    logger()->info("M0_blocked strides: {}", M0_blocked.strides);
    auto [M0_ordered]    = unpack<1>(transpose(M0_scaled, {{0, 1, 3, 2, 4}}));
    auto [M0_sh_disp]    = unpack<1>(reshape(M0_ordered, {{1, h, w}}));
    std::tie(M0_sh_disp) = unpack<1>(convert(M0_sh_disp, {Target::U8, Strat::Scaled}));
    std::tie(M0_sh_disp) = unpack<1>(memcpy(M0_sh_disp, {Host}));
    std::tie(M0_sh_disp) = unpack<1>(batched_queue(M0_sh_disp, {s_.cpu_out_size, 1, 1}));
    shack_hartmann_display(M0_sh_disp, {});

    // auto [xcorr_shift]   = unpack<1>(fftshift(xcorr, {{-2, -1}}));
    // auto [xcorr_scaled]  = rescale(xcorr_shift, {-2, -1}, 0.0f, 255.0f);
    // auto [xcorr_ordered] = unpack<1>(transpose(xcorr_scaled, {{0, 1, 3, 2, 4}}));
    // auto [xcorr_disp]    = unpack<1>(reshape(xcorr_ordered, {{1, h, w}}));
    // std::tie(xcorr_disp) = unpack<1>(convert(xcorr_disp, {Target::U8, Strat::Scaled}));
    // std::tie(xcorr_disp) = unpack<1>(memcpy(xcorr_disp, {Host}));
    // std::tie(xcorr_disp) = unpack<1>(batched_queue(xcorr_disp, {s_.cpu_out_size, 1, 1}));
    // shack_hartmann_xcorr_display(xcorr_disp, {});
  }

  return g_;
}

// -------------------------------------------------------------------------------------------------
// Task Wrappers Implementations
// -------------------------------------------------------------------------------------------------

#define DEFINE_SOURCE_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                    \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(SettingsType s) {                   \
    return make_source_sync_node(node_name_str, kind_str, kind_str, s);                            \
  }

#define DEFINE_UNARY_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                     \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(const TDesc &X, SettingsType s) {   \
    return make_unary_sync_node(node_name_str, kind_str, kind_str, X, s);                          \
  }

#define DEFINE_NARY_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                      \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(std::span<const TDesc> Xs,          \
                                                               SettingsType           s) {                   \
    return make_nary_sync_node(node_name_str, kind_str, kind_str, Xs, s);                          \
  }

#define DEFINE_UNARY_ASYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                    \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(const TDesc &X, SettingsType s) {   \
    return make_unary_async_node(node_name_str, kind_str, kind_str, X, s);                         \
  }

// clang-format off
DEFINE_SOURCE_SYNC_NODE(holofile_read,                          "source",                              "Holofile",                        holotask::sources::HolofileSettings)
DEFINE_SOURCE_SYNC_NODE(empty,                                 "empty",                               "Empty",                            holonp::EmptySettings)
DEFINE_SOURCE_SYNC_NODE(zeros,                                 "zeros",                               "Zeros",                            holonp::ZerosSettings)
DEFINE_SOURCE_SYNC_NODE(asarray,                               "asarray",                             "AsArray",                          holonp::AsArraySettings)
DEFINE_SOURCE_SYNC_NODE(ametek_s710_euresys_coaxlink_octo,      "ametek_s710_euresys_coaxlink_octo",   "AmetekS710EuresysCoaxlinkOcto",   holotask::sources::AmetekS710EuresysCoaxlinkOctoSettings)
DEFINE_SOURCE_SYNC_NODE(ametek_s711_euresys_coaxlink_qsfp_plus, "ametek_s711_euresys_coaxlink_qsfp_+", "AmetekS711EuresysCoaxlinkQSFP+",  holotask::sources::AmetekS711EuresysCoaxlinkQSFPSettings)
DEFINE_UNARY_SYNC_NODE (fresnel_qin,                            "fresnel_qin",                         "FresnelQin",                      holotask::sources::FresnelQinSettings)
DEFINE_UNARY_SYNC_NODE (memcpy,                                 "memcpy",                              "Memcpy",                          holotask::syncs::MemcpySettings)
DEFINE_UNARY_SYNC_NODE (convert,                                "conversion",                          "Conversion",                      holotask::syncs::ConversionSettings)
DEFINE_UNARY_SYNC_NODE (pca,                                    "pca",                                 "Pca",                             holotask::syncs::PcaSettings)
DEFINE_UNARY_SYNC_NODE (stft,                                   "stft",                                "Stft",                            holotask::syncs::StftSettings)
DEFINE_UNARY_SYNC_NODE (filter_2d,                              "filter_2d",                           "Filter2D",                        holotask::syncs::Filter2DSettings)
DEFINE_UNARY_SYNC_NODE (fresnel_diffraction,                    "fresnel_diffraction",                 "FresnelDiffraction",              holotask::syncs::FresnelDiffractionSettings)
DEFINE_UNARY_SYNC_NODE (angular_spectrum,                       "angular_spectrum",                    "AngularSpectrum",                 holotask::syncs::AngularSpectrumSettings)
DEFINE_UNARY_SYNC_NODE (reshape,                                "reshape",                             "Reshape",                         holonp::ReshapeSettings)
DEFINE_UNARY_SYNC_NODE (average,                                "average",                             "Average",                         holotask::syncs::AverageSettings)
DEFINE_UNARY_SYNC_NODE (convolution,                            "convolution",                         "Convolution",                     holotask::syncs::ConvolutionSettings)
DEFINE_UNARY_SYNC_NODE (crop,                                   "crop",                                "Crop",                            holotask::syncs::CropSettings)
DEFINE_UNARY_SYNC_NODE (fft_shift,                              "fft_shift",                           "FFTShift",                        holotask::syncs::FFTShiftSettings)
DEFINE_UNARY_SYNC_NODE (pct_clip,                               "pct_clip",                            "PctClip",                         holotask::syncs::PctClipSettings)
DEFINE_UNARY_SYNC_NODE (registration,                           "registration",                        "Registration",                    holotask::syncs::RegistrationSettings)
DEFINE_UNARY_SYNC_NODE (rotation,                               "rotation",                            "Rotation",                        holotask::syncs::RotationSettings)
DEFINE_UNARY_SYNC_NODE (xy_raw_display,                         "xy_raw_display",                      "DisplayTensorXYRaw",              tasks::sinks::DisplayTensorSettings)
DEFINE_UNARY_SYNC_NODE (xy_processed_display,                   "xy_processed_display",                "DisplayTensorXY",                 tasks::sinks::DisplayTensorSettings)
DEFINE_UNARY_SYNC_NODE (xz_processed_display,                   "xz_processed_display",                "DisplayTensorXZ",                 tasks::sinks::DisplayTensorSettings)
DEFINE_UNARY_SYNC_NODE (yz_processed_display,                   "yz_processed_display",                "DisplayTensorYZ",                 tasks::sinks::DisplayTensorSettings)
DEFINE_UNARY_SYNC_NODE (shack_hartmann_display,                 "shack_hartmann_display",              "DisplayTensorShackHartmann",      tasks::sinks::DisplayTensorSettings)
DEFINE_UNARY_SYNC_NODE (shack_hartmann_xcorr_display,           "shack_hartmann_xcorr_display",        "DisplayTensorShackHartmannXcorr", tasks::sinks::DisplayTensorSettings)
DEFINE_NARY_SYNC_NODE  (concatenate,                            "concatenate",                         "Concatenate",                     holonp::ConcatenateSettings)
DEFINE_UNARY_SYNC_NODE (transpose,                              "transpose",                           "Transpose",                       holonp::TransposeSettings)
DEFINE_UNARY_SYNC_NODE (conj,                                   "conj",                                "Conj",                            holonp::ConjSettings)
DEFINE_UNARY_SYNC_NODE (rfft,                                   "rfft",                                "RFFT",                            holonp::RFFTSettings)
DEFINE_UNARY_SYNC_NODE (rfft2,                                  "rfft2",                               "RFFT2",                           holonp::RFFT2Settings)
DEFINE_UNARY_SYNC_NODE (irfft2,                                 "irfft2",                              "IRFFT2",                          holonp::IRFFT2Settings)
DEFINE_UNARY_SYNC_NODE (slice,                                  "slice",                               "Slice",                           holonp::SliceSettings)
DEFINE_UNARY_SYNC_NODE (fft,                                    "fft",                                 "FFT",                             holonp::FFTSettings)
DEFINE_UNARY_SYNC_NODE (fft2,                                   "fft2",                                "FFT2",                            holonp::FFT2Settings)
DEFINE_UNARY_SYNC_NODE (fftshift,                               "fftshift",                            "FFTShiftNp",                      holonp::FFTShiftSettings)
DEFINE_UNARY_SYNC_NODE (abs,                                    "abs",                                 "Abs",                             holonp::AbsSettings)
DEFINE_UNARY_SYNC_NODE (mean,                                   "mean",                                "Mean",                            holonp::MeanSettings)
DEFINE_UNARY_SYNC_NODE (mean_abs,                               "mean_abs",                            "MeanAbs",                         holonp::MeanAbsSettings)
DEFINE_UNARY_SYNC_NODE (min,                                    "min",                                 "Min",                             holonp::MinSettings)
DEFINE_UNARY_SYNC_NODE (max,                                    "max",                                 "Max",                             holonp::MaxSettings)
DEFINE_UNARY_ASYNC_NODE(batched_queue,                          "batch_queue",                         "BatchQueue",                      holotask::asyncs::BatchQueueSettings)
DEFINE_UNARY_ASYNC_NODE(slide_avg,                              "slide_avg",                           "SlidingAverage",                  holotask::asyncs::SlidingAverageSettings)
// clang-format on

std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::mul(const TDesc &A, const TDesc &B,
                                                         holonp::MulSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return make_nary_sync_node("mul", "Mul", "Mul", std::span<const TDesc>{inputs}, s);
}

std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::add(const TDesc &A, const TDesc &B,
                                                         holonp::AddSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return make_nary_sync_node("add", "Add", "Add", std::span<const TDesc>{inputs}, s);
}

std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::div(const TDesc &A, const TDesc &B,
                                                         holonp::DivSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return make_nary_sync_node("div", "Div", "Div", std::span<const TDesc>{inputs}, s);
}

std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::sub(const TDesc &A, const TDesc &B,
                                                         holonp::SubSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return make_nary_sync_node("sub", "Sub", "Sub", std::span<const TDesc>{inputs}, s);
}

std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::equal(const TDesc &A, const TDesc &B,
                                                           holonp::EqualSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return make_nary_sync_node("equal", "Equal", "Equal", std::span<const TDesc>{inputs}, s);
}

std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::where(const TDesc &Cond, const TDesc &X, const TDesc &Y, holonp::WhereSettings s) {
  std::array<TDesc, 3> inputs{Cond, X, Y};
  return make_nary_sync_node("where", "Where", "Where", std::span<const TDesc>{inputs}, s);
}

std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::assign(const TDesc &X, const TDesc &Y,
                                                            holonp::AssignSettings s) {
  std::array<TDesc, 2> inputs{X, Y};
  return make_nary_sync_node("assign", "Assign", "Assign", std::span<const TDesc>{inputs}, s);
}

std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::holofile_write(const TDesc &X, holotask::sinks::HolofileSettings s) {
  auto node_name = "record";
  auto kind      = "HolofileWriter";
  auto reg_key   = "HolofileWriter";
  auto debug     = false;

  HOLOVIBES_CHECK(X.producer.has_value());
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name},
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, 0}, g_);

  auto      &factory     = reg_.get_sync(std::string{reg_key});
  const auto core_inputs = to_core_descs(std::span{&X, 1});
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

// -------------------------------------------------------------------------------------------------
// Various Helpers
// -------------------------------------------------------------------------------------------------

holoflow::core::TDesc GraphBuilder_v2::TDesc::as_core() const {
  holoflow::core::TDesc t{};
  t.shape   = shape;
  t.dtype   = dtype;
  t.mem_loc = mem_loc;
  t.strides = strides;
  t.offset  = offset;
  return t;
}

GraphBuilder_v2::TDesc GraphBuilder_v2::TDesc::from_core(const holoflow::core::TDesc &base) {
  TDesc t;
  static_cast<holoflow::core::TDesc &>(t) = base;
  return t;
}

std::vector<holoflow::core::TDesc> GraphBuilder_v2::to_core_descs(std::span<const TDesc> src) {
  std::vector<holoflow::core::TDesc> out;
  out.reserve(src.size());
  for (const auto &t : src) {
    out.push_back(t.as_core());
  }
  return out;
}

template <class InferResult>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::wrap_infer_outputs(std::string_view node_id, V vertex, const InferResult &infer) {
  std::vector<TDesc> ys;
  ys.reserve(infer.output_descs.size());
  int out_idx = 0;
  for (const auto &base : infer.output_descs) {
    auto y     = TDesc::from_core(base);
    y.producer = TDesc::Producer{
        .node_id = std::string{node_id},
        .out_idx = out_idx,
        .vertex  = vertex,
    };
    ys.push_back(std::move(y));
    ++out_idx;
  }
  return ys;
}

template <typename SettingsT>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::make_source_sync_node(std::string_view node_name, std::string_view kind,
                                       std::string_view reg_key, const SettingsT &s, bool debug) {
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);

  auto      &factory = reg_.get_sync(std::string{reg_key});
  const auto infer   = factory.infer(std::span<const holoflow::core::TDesc>{}, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

template <typename SettingsT>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::make_unary_sync_node(std::string_view node_name, std::string_view kind,
                                      std::string_view reg_key, const TDesc &X, const SettingsT &s,
                                      bool debug) {
  HOLOVIBES_CHECK(X.producer.has_value());
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, 0}, g_);

  auto      &factory     = reg_.get_sync(std::string{reg_key});
  const auto core_inputs = to_core_descs(std::span{&X, 1});
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

template <typename SettingsT>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::make_nary_sync_node(std::string_view node_name, std::string_view kind,
                                     std::string_view reg_key, std::span<const TDesc> inputs,
                                     const SettingsT &s, bool debug) {
  HOLOVIBES_CHECK(!inputs.empty());
  for (const auto &X : inputs) {
    HOLOVIBES_CHECK(X.producer.has_value());
  }
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &X = inputs[i];
    boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, static_cast<int>(i)}, g_);
  }

  auto      &factory     = reg_.get_sync(std::string{reg_key});
  const auto core_inputs = to_core_descs(inputs);
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

template <typename SettingsT>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::make_unary_async_node(std::string_view node_name, std::string_view kind,
                                       std::string_view reg_key, const TDesc &X, const SettingsT &s,
                                       bool debug) {
  HOLOVIBES_CHECK(X.producer.has_value());
  HOLOVIBES_CHECK(reg_.is_async_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, 0}, g_);

  auto      &factory     = reg_.get_async(std::string{reg_key});
  const auto core_inputs = to_core_descs(std::span{&X, 1});
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

} // namespace holovibes::pipeline
