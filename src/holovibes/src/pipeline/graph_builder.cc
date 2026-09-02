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
#include <map>
#include <optional>
#include <spdlog/fmt/ranges.h>
#include <stdexcept>
#include <utility>

#include "bug.hh"
#include "graph_builder_tasks.hh"
#include "logger.hh"
#include "pipeline/settings.hh"
#include "settings_loader.hh"

namespace holovibes::pipeline {

using PhaseReference = holotask::syncs::STFDPhaseReference;

// -------------------------------------------------------------------------------------------------
// Implementation
// -------------------------------------------------------------------------------------------------

class GraphBuilder::Impl : public GraphBuilderTasks {
public:
  Impl(const Settings &settings, holoflow::core::Registry &registry);

  holoflow::core::GraphSpec build();

private:
  struct ShackHartmannGeometry {
    size_t frame_width;
    size_t frame_height;
    size_t nb_subapertures;
    size_t subaperture_width;
    size_t subaperture_height;

    float wavelength;
    float pixel_pitch_x;
    float pixel_pitch_y;
    float propagation_distance;
    float pupil_radius_m;
  };

  struct ShackHartmannSlopeOutput {
    TDesc                slopes;
    std::optional<TDesc> xcorr;
  };

  struct AberrationCorrectionState {
    std::optional<TDesc> cumulative_coeffs_gpu;
    std::optional<TDesc> cumulative_phase_gpu;
  };

  // clang-format off
  TDesc build_acquisition();
  TDesc short_time_fresnel_diffraction(const TDesc &field, size_t win_w, size_t win_h, size_t stride_x, size_t stride_y, float lam, float dx, float dy, float z_prop, PhaseReference phase_ref, bool skip_phase_shift = true);
  void build_raw_record(const TDesc &H);
  bool build_raw_view(const TDesc &H);
  TDesc build_preprocessing(TDesc H);
  TDesc build_time_frequency_analysis(TDesc H);
  TDesc build_aberration_correction(const TDesc &FH_current, const TDesc &FH_delayed);
  TDesc build_aberration_correction_pass(const TDesc &FH_current, const TDesc &FH_delayed, bool is_last_pass, AberrationCorrectionState &state);
  ShackHartmannGeometry shack_hartmann_geometry(const TDesc &FH) const;
  TDesc build_shack_hartmann_sensor(const TDesc &FH, const ShackHartmannGeometry &geometry);
  ShackHartmannSlopeOutput build_shack_hartmann_slopes(const TDesc &sensor_images, const ShackHartmannGeometry &geometry, bool output_xcorr);
  void build_shack_hartmann_view(const TDesc &sensor_images, const ShackHartmannGeometry &geometry);
  void build_shack_hartmann_xcorr_view(const TDesc &xcorr, const ShackHartmannGeometry &geometry);
  TDesc build_zernike_correction(const TDesc &FH, const TDesc &slopes, const ShackHartmannGeometry &geometry, bool is_last_pass, AberrationCorrectionState &state);
  void build_zernike_outputs(const AberrationCorrectionState &state, const ShackHartmannGeometry &geometry);
  TDesc build_spatial_propagation(const TDesc &FH);
  TDesc build_spatial_filter(const TDesc &FH_z);
  void build_xy_view(const TDesc &FH_z);
  void build_3d_cuts(const TDesc &FH_z);
  TDesc build_freq_weights();
  // clang-format on

  Settings s_;

  std::map<LoadMethod, holotask::sources::HolofileSettings::LoadKind> load_method_map_{
      {LoadMethod::READ_LIVE, holotask::sources::HolofileSettings::LoadKind::Live},
      {LoadMethod::LOAD_IN_CPU, holotask::sources::HolofileSettings::LoadKind::CPUCached},
      {LoadMethod::LOAD_IN_GPU, holotask::sources::HolofileSettings::LoadKind::GPUCached},
  };
};

// -----------------------------------------------------------------------------
// Helpers declarations
// -----------------------------------------------------------------------------

namespace {

holotask::syncs::FlatfieldSettings flatfield_settings_from_cutoff_period(float cutoff_period_m,
                                                                         float dy_m, float dx_m);

std::pair<float, float> fresnel_1fft_output_pitch(float wavelength_m, float z_m, float dy_in_m,
                                                  float dx_in_m, size_t ny, size_t nx);

std::pair<float, float> post_propagation_pitch(const Settings              &settings,
                                               const holoflow::core::TDesc &desc);

} // namespace

// -------------------------------------------------------------------------------------------------
// Initialization
// -------------------------------------------------------------------------------------------------

GraphBuilder::GraphBuilder(const Settings &settings, holoflow::core::Registry &registry)
    : impl_(std::make_unique<Impl>(settings, registry)) {}

GraphBuilder::~GraphBuilder() = default;

GraphBuilder::GraphBuilder(GraphBuilder &&) noexcept            = default;
GraphBuilder &GraphBuilder::operator=(GraphBuilder &&) noexcept = default;

holoflow::core::GraphSpec GraphBuilder::build() { return impl_->build(); }

GraphBuilder::Impl::Impl(const Settings &settings, holoflow::core::Registry &registry)
    : GraphBuilderTasks(registry), s_(settings) {}

// -------------------------------------------------------------------------------------------------
// Top-level pipeline
// -------------------------------------------------------------------------------------------------

holoflow::core::GraphSpec GraphBuilder::Impl::build() {
  if (!std::isfinite(s_.signal_plot_time_window_seconds) ||
      s_.signal_plot_time_window_seconds <= 0.0) {
    throw std::invalid_argument("signal_plot_time_window_seconds must be positive and finite");
  }
  const double signal_plot_sample_time_seconds = s_.signal_plot_sample_time_seconds();
  if (!std::isfinite(signal_plot_sample_time_seconds) || signal_plot_sample_time_seconds <= 0.0) {
    throw std::invalid_argument("derived signal plot sample time must be positive and finite");
  }

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

  bool filter_2d_standalone = s_.spacial_method != SpacialMethod::ANGULAR_SPECTRUM;
  if (s_.filter_2d && filter_2d_standalone) {
    FH = build_spatial_filter(FH);
  }

  if (s_.pp_accumulation <= 0) {
    throw std::invalid_argument("pp_accumulation must be positive");
  }

  // Batch queue with two readers. FH_delayed is used for Shack-Hartmann correction, FH_current is
  // used for propagation.
  auto        target_capacity = std::max<size_t>(2, 2 * FH.shape.at(0));
  auto        window_size     = static_cast<size_t>(s_.pp_accumulation);
  const auto  timing_outputs  = dual_reader_batch_queue(FH, {target_capacity, window_size});
  TDesc       FH_current      = timing_outputs.at(0);
  const TDesc FH_delayed      = timing_outputs.at(1);

  if (s_.autofocus_enabled) {
    FH_current = build_aberration_correction(FH_current, FH_delayed);
  }

  TDesc FH_z = build_spatial_propagation(FH_current);

  build_xy_view(FH_z);

  if (s_.view_3d_cuts) {
    build_3d_cuts(FH_z);
  }

  return g_;
}

// -------------------------------------------------------------------------------------------------
// Pipeline stages
// -------------------------------------------------------------------------------------------------

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_acquisition() {
  auto cam_path = s_.camera_config_path.string();

  if (s_.import_source == ImportSource::HOLOFILE) {
    return holofile_read({.path        = s_.load_path.string(),
                          .load_kind   = load_method_map_.at(s_.load_method),
                          .start_frame = s_.load_begin,
                          .end_frame   = s_.load_end,
                          .batch_size  = s_.load_batch,
                          .max_fps     = s_.load_fps_limit,
                          .keep_cursor = false});
  }

  if (s_.import_source == ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO) {
    return ametek_s710_euresys_coaxlink_octo({cam_path});
  }

  if (s_.import_source == ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP) {
    return ametek_s711_euresys_coaxlink_qsfp_plus({cam_path});
  }

  HOLOVIBES_UNREACHABLE();
}

void GraphBuilder::Impl::build_raw_record(const TDesc &H) {
  auto path          = s_.recording_path.string();
  auto count         = s_.recording_count;
  auto settings_json = settings_to_old_json(s_);
  holofile_write(H, {path, count, settings_json, true});
}

bool GraphBuilder::Impl::build_raw_view(const TDesc &H) {
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  auto H_disp = memcpy(H, {Host});
  auto H_view = batched_queue(H_disp, {s_.cpu_out_size, 1, 1});

  if (s_.raw_view) {
    xy_raw_display(H_view, {});
  }

  if (s_.view_type == ViewType::RAW) {
    xy_processed_display(H_view, {});
    return true;
  }

  return false;
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_preprocessing(TDesc H) {
  using MemLoc = holoflow::core::MemLoc;
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Device  = holotask::syncs::MemcpySettings::Target::Device;

  if (H.mem_loc != MemLoc::Device) {
    H = memcpy(H, {Device});
    H = batched_queue(H, {s_.gpu_in_size, s_.time_window, s_.time_window});
  }

  return convert(H, {Target::F32, Strat::Real});
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_time_frequency_analysis(TDesc H) {
  // H enters as [T, Hy, Hx] (F32).
  // We first accumulate N_pre such windows into a batch, producing [N_pre, T, Hy, Hx].
  // Time-frequency analysis then operates along axis 1 (the T dimension).
  // The output is [N_pre, Nz, Hy, Hx], which feeds directly into the post-TFA queue.

  int     N_pre = 8;
  int64_t T     = static_cast<int64_t>(H.shape.at(0));
  int64_t Hy    = static_cast<int64_t>(H.shape.at(1));
  int64_t Hx    = static_cast<int64_t>(H.shape.at(2));

  H = reshape(H, {{1, T, Hy, Hx}, false});
  H = batched_queue(H, {N_pre * 2, N_pre, N_pre}); // [N_pre, T, Hy, Hx]

  if (s_.time_method == TimeMethod::RFFT) {
    auto FH = rfft(H, {1}); // axis 1 = T dimension

    if (s_.view_3d_cuts) {
      return FH;
    }

    // Optimization: slice relevant components early. The rest is not used therefore this saves
    // further computations.
    FH = slice(FH, {{{}, holonp::SliceRange{s_.time_z_begin, s_.time_z_end}, {}, {}}});
    FH = copy(FH, {});
    return FH;
  }

  if (s_.time_method == TimeMethod::FFT) {
    auto FH = fft(H, {1});

    if (s_.view_3d_cuts) {
      return FH;
    }

    // Optimization: slice relevant components early, including the symmetric negative band. The
    // rest is not used therefore this saves further computations.
    auto pos_range = holonp::SliceRange{s_.time_z_begin, s_.time_z_end};
    auto FH_pos    = slice(FH, {{{}, pos_range, {}, {}}});

    auto neg_range = holonp::SliceRange{T - s_.time_z_end, T - s_.time_z_begin};
    auto FH_neg    = slice(FH, {{{}, neg_range, {}, {}}});

    FH = concatenate(std::array<TDesc, 2>{FH_pos, FH_neg}, {1});
    return FH;
  }

  if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS) {
    // PCA natively supports arbitrary leading batch dimensions (rank >= 3).
    // Input H: [N_pre, T, Hy, Hx] — the feature axis is shape[-3] = T.
    // Output FH: [N_pre, Nz, Hy, Hx] where Nz = z1 - z0.
    int  z0 = s_.view_3d_cuts ? 0 : s_.time_z_begin;
    int  z1 = s_.view_3d_cuts ? static_cast<int>(T) : s_.time_z_end;
    auto FH = pca(H, {z0, z1});
    return FH;
  }

  HOLOVIBES_UNREACHABLE();
}

// -------------------------------------------------------------------------------------------------
// Aberration correction
// -------------------------------------------------------------------------------------------------

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_aberration_correction(const TDesc &FH_current,
                                                                          const TDesc &FH_delayed) {
  if (s_.autofocus_nb_iter <= 0) {
    throw std::invalid_argument("autofocus_nb_iter must be positive");
  }
  if (s_.pp_accumulation > 1 && s_.autofocus_nb_iter > 1) {
    throw std::invalid_argument(
        "sliding Shack-Hartmann correction only supports one autofocus iteration");
  }

  auto                      corrected = FH_current;
  AberrationCorrectionState state;

  for (int pass = 0; pass < s_.autofocus_nb_iter; ++pass) {
    const bool   is_last_pass = pass == s_.autofocus_nb_iter - 1;
    const TDesc &delayed      = pass == 0 ? FH_delayed : corrected;
    corrected = build_aberration_correction_pass(corrected, delayed, is_last_pass, state);
  }

  // Decouple the final propagation so it can overlap the next Shack-Hartmann iteration.
  return batched_queue(corrected, {2, 1, 1});
}

GraphBuilder::Impl::TDesc
GraphBuilder::Impl::build_aberration_correction_pass(const TDesc &FH_current,
                                                     const TDesc &FH_delayed, bool is_last_pass,
                                                     AberrationCorrectionState &state) {
  auto geometry      = shack_hartmann_geometry(FH_current);
  auto sensor_images = build_shack_hartmann_sensor(FH_current, geometry);

  bool output_xcorr = is_last_pass && s_.view_shack_hartmann_xcorr;
  auto slope_output = build_shack_hartmann_slopes(sensor_images, geometry, output_xcorr);

  if (is_last_pass && s_.view_shack_hartmann) {
    build_shack_hartmann_view(sensor_images, geometry);
  }

  if (slope_output.xcorr.has_value()) {
    build_shack_hartmann_xcorr_view(*slope_output.xcorr, geometry);
  }

  return build_zernike_correction(FH_delayed, slope_output.slopes, geometry, is_last_pass, state);
}

GraphBuilder::Impl::ShackHartmannGeometry
GraphBuilder::Impl::shack_hartmann_geometry(const TDesc &FH) const {
  if (s_.autofocus_nb_subaps <= 0) {
    throw std::invalid_argument("autofocus_nb_subaps must be positive");
  }
  if ((s_.autofocus_nb_subaps % 2) == 0) {
    throw std::invalid_argument("autofocus_nb_subaps must be odd");
  }

  const auto frame_width        = FH.shape.at(3);
  const auto frame_height       = FH.shape.at(2);
  const auto nb_subapertures    = static_cast<size_t>(s_.autofocus_nb_subaps);
  const auto subaperture_width  = frame_width / nb_subapertures;
  const auto subaperture_height = frame_height / nb_subapertures;

  if (subaperture_width == 0 || subaperture_height == 0) {
    throw std::invalid_argument("autofocus_nb_subaps is too large for the current frame size");
  }

  const auto  wavelength           = s_.spacial_lambda;
  const auto  pixel_pitch_x        = s_.spacial_pixel_size;
  const auto  pixel_pitch_y        = s_.spacial_pixel_size;
  const auto  propagation_distance = s_.spacial_z;
  const auto  width_m  = static_cast<float>(subaperture_width * nb_subapertures) * pixel_pitch_x;
  const auto  height_m = static_cast<float>(subaperture_height * nb_subapertures) * pixel_pitch_y;
  const float pupil_radius_m = 0.5f * std::min(width_m, height_m);

  return {.frame_width          = frame_width,
          .frame_height         = frame_height,
          .nb_subapertures      = nb_subapertures,
          .subaperture_width    = subaperture_width,
          .subaperture_height   = subaperture_height,
          .wavelength           = wavelength,
          .pixel_pitch_x        = pixel_pitch_x,
          .pixel_pitch_y        = pixel_pitch_y,
          .propagation_distance = propagation_distance,
          .pupil_radius_m       = pupil_radius_m};
}

GraphBuilder::Impl::TDesc
GraphBuilder::Impl::build_shack_hartmann_sensor(const TDesc                 &FH,
                                                const ShackHartmannGeometry &geometry) {
  auto propagated = short_time_fresnel_diffraction(
      FH, geometry.subaperture_width, geometry.subaperture_height, geometry.subaperture_width,
      geometry.subaperture_height, geometry.wavelength, geometry.pixel_pitch_x,
      geometry.pixel_pitch_y, geometry.propagation_distance, PhaseReference::GLOBAL);

  auto sensor_images = mean(propagated, {{1}, false});
  sensor_images      = causal_slide_avg(sensor_images, {static_cast<size_t>(s_.pp_accumulation)});
  sensor_images      = fftshift(sensor_images, {{-2, -1}});

  if (!s_.pp_flatfield) {
    return sensor_images;
  }

  const auto [dy, dx] = fresnel_1fft_output_pitch(
      geometry.wavelength, geometry.propagation_distance, geometry.pixel_pitch_y,
      geometry.pixel_pitch_x, geometry.subaperture_height, geometry.subaperture_width);

  const auto cutoff   = s_.pp_flatfield_cutoff_period_m;
  const auto settings = flatfield_settings_from_cutoff_period(cutoff, dy, dx);
  sensor_images       = flatfield(sensor_images, settings);
  return sensor_images;
}

GraphBuilder::Impl::ShackHartmannSlopeOutput GraphBuilder::Impl::build_shack_hartmann_slopes(
    const TDesc &sensor_images, const ShackHartmannGeometry &geometry, bool output_xcorr) {
  using SlopeMode      = holotask::syncs::ShackHartmannSlopeMode;
  auto FullPairwise    = SlopeMode::FullPairwise;
  auto SingleReference = SlopeMode::SingleReference;

  const auto mode       = s_.autofocus_use_graph_laplacian ? FullPairwise : SingleReference;
  const bool emit_xcorr = output_xcorr && mode == SingleReference;

  auto outputs = shack_hartmann_slopes(
      sensor_images,
      {.mode               = mode,
       .lambda             = geometry.wavelength,
       .dx                 = geometry.pixel_pitch_x,
       .dy                 = geometry.pixel_pitch_y,
       .z                  = geometry.propagation_distance,
       .subaperture_height = geometry.subaperture_height,
       .subaperture_width  = geometry.subaperture_width,
       .stride_y           = geometry.subaperture_height,
       .stride_x           = geometry.subaperture_width,
       .correlation_roi    = {0.5f, 0.5f, s_.pp_pctclip_radius, s_.pp_pctclip_radius, 0.0f},
       .skip_subapertures_outside_pupil = s_.autofocus_skip_subapertures_outside_pupil,
       .output_xcorr_maps               = emit_xcorr});

  ShackHartmannSlopeOutput result{.slopes = outputs.at(0)};
  if (emit_xcorr) {
    result.xcorr = outputs.at(1);
  }

  return result;
}

void GraphBuilder::Impl::build_shack_hartmann_view(const TDesc                 &sensor_images,
                                                   const ShackHartmannGeometry &geometry) {
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;

  const auto height = static_cast<int64_t>(geometry.subaperture_height * geometry.nb_subapertures);
  const auto width  = static_cast<int64_t>(geometry.subaperture_width * geometry.nb_subapertures);

  auto display = normalize(sensor_images, {{-2, -1}, 0.0f, 255.0f});
  display      = transpose(display, {{0, 1, 3, 2, 4}});
  display      = reshape(display, {{1, height, width}});
  display      = convert(display, {Target::U8, Strat::Scaled});
  display      = batched_queue(display, {s_.cpu_out_size, 1, 1});
  shack_hartmann_display(display, {});
}

void GraphBuilder::Impl::build_shack_hartmann_xcorr_view(const TDesc                 &xcorr,
                                                         const ShackHartmannGeometry &geometry) {
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;

  auto display = fftshift(xcorr, {{-2, -1}});
  display      = normalize(display, {{-2, -1}, 0.0f, 255.0f});

  const auto height = static_cast<int64_t>(display.shape.at(3) * geometry.nb_subapertures);
  const auto width  = static_cast<int64_t>(display.shape.at(4) * geometry.nb_subapertures);

  display = convert(display, {Target::U8, Strat::Scaled});
  display = transpose(display, {{0, 1, 3, 2, 4}});
  display = reshape(display, {{1, height, width}});
  display = batched_queue(display, {s_.cpu_out_size, 1, 1});
  shack_hartmann_xcorr_display(display, {});
}

GraphBuilder::Impl::TDesc
GraphBuilder::Impl::build_zernike_correction(const TDesc &FH, const TDesc &slopes,
                                             const ShackHartmannGeometry &geometry,
                                             bool is_last_pass, AberrationCorrectionState &state) {
  auto F32    = holoflow::core::DType::F32;
  auto Device = holoflow::core::MemLoc::Device;

  if (s_.autofocus_zernike_orders.empty()) {
    auto phase     = zeros({{1, geometry.frame_height, geometry.frame_width}, F32});
    auto corrected = correct_phase(FH, phase, {});

    if (is_last_pass && s_.view_zernike_phase) {
      zernike_phase_display(phase, {});
    }

    return corrected;
  }

  holotask::syncs::ZernikeFromSlopesSettings settings{
      .indexes                         = s_.autofocus_zernike_orders,
      .lambda                          = geometry.wavelength,
      .dx                              = geometry.pixel_pitch_x,
      .dy                              = geometry.pixel_pitch_y,
      .subaperture_height              = geometry.subaperture_height,
      .subaperture_width               = geometry.subaperture_width,
      .stride_y                        = geometry.subaperture_height,
      .stride_x                        = geometry.subaperture_width,
      .ny                              = 1,
      .nx                              = 1,
      .skip_subapertures_outside_pupil = s_.autofocus_skip_subapertures_outside_pupil};

  auto coeffs    = zernike_from_slopes(slopes, settings);
  coeffs         = slice(coeffs, {{0, 0, {}}});
  auto height    = static_cast<int>(geometry.frame_height);
  auto width     = static_cast<int>(geometry.frame_width);
  auto phase     = zernike_phase(coeffs, {s_.autofocus_zernike_orders, height, width, Device});
  auto corrected = correct_phase(FH, phase, {});

  const auto accumulate = [this](auto &cumulative, const auto &value) {
    cumulative = cumulative ? add(*cumulative, value, {}) : value;
  };

  accumulate(state.cumulative_coeffs_gpu, coeffs);
  accumulate(state.cumulative_phase_gpu, phase);

  if (is_last_pass) {
    build_zernike_outputs(state, geometry);
  }

  return corrected;
}

void GraphBuilder::Impl::build_zernike_outputs(const AberrationCorrectionState &state,
                                               const ShackHartmannGeometry     &geometry) {
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  HOLOVIBES_CHECK(state.cumulative_coeffs_gpu.has_value());
  HOLOVIBES_CHECK(state.cumulative_phase_gpu.has_value());

  const auto &coeffs = *state.cumulative_coeffs_gpu;
  const auto &phase  = *state.cumulative_phase_gpu;

  zernike_coefficients_display(coeffs, {s_.autofocus_zernike_orders});

  if (s_.view_zernike_metrics) {
    auto coeffs_host = memcpy(coeffs, {Host});
    zernike_history_display(coeffs_host, {
                                             s_.autofocus_zernike_orders,
                                             s_.signal_plot_time_window_seconds,
                                             s_.signal_plot_sample_time_seconds(),
                                             static_cast<size_t>(s_.pp_accumulation - 1),
                                         });
  }

  const bool defocus_included =
      std::ranges::find(s_.autofocus_zernike_orders, 4) != s_.autofocus_zernike_orders.end();

  if (defocus_included) {
    zernike_defocus_z_prop(coeffs, {s_.autofocus_zernike_orders, geometry.wavelength,
                                    geometry.propagation_distance, geometry.pupil_radius_m});
  }

  if (s_.view_zernike_phase) {
    auto height  = static_cast<int64_t>(geometry.frame_height);
    auto width   = static_cast<int64_t>(geometry.frame_width);
    auto display = copy(phase, {});
    display      = wrap2pi(display, {});
    display      = reshape(display, {{1, height, width}});
    display      = batched_queue(display, {s_.cpu_out_size, 1, 1});
    zernike_phase_display(display, {});
  }
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::short_time_fresnel_diffraction(
    const TDesc &field, size_t win_w, size_t win_h, size_t stride_x, size_t stride_y, float lam,
    float dx, float dy, float z_prop, PhaseReference phase_ref, bool skip_phase_shift) {
  return GraphBuilderTasks::short_time_fresnel_diffraction(field,
                                                           {.lambda           = lam,
                                                            .dx               = dx,
                                                            .dy               = dy,
                                                            .z                = z_prop,
                                                            .win_h            = win_h,
                                                            .win_w            = win_w,
                                                            .stride_y         = stride_y,
                                                            .stride_x         = stride_x,
                                                            .phase_ref        = phase_ref,
                                                            .skip_phase_shift = skip_phase_shift,
                                                            .output_magnitude = true});
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_spatial_propagation(const TDesc &FH) {
  using Padding = holotask::syncs::AngularSpectrumSettings::Padding;
  using Filter  = holotask::syncs::AngularSpectrumSettings::Filter;

  if (s_.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION) {
    return fresnel_diffraction(FH, {.lambda           = s_.spacial_lambda,
                                    .dx               = s_.spacial_pixel_size,
                                    .dy               = s_.spacial_pixel_size,
                                    .z                = s_.spacial_z,
                                    .axes             = {-2, -1},
                                    .output_magnitude = true});
  }

  if (s_.spacial_method == SpacialMethod::ANGULAR_SPECTRUM) {
    if (s_.autofocus_enabled) {
      throw std::logic_error{"Angular Spectrum is not supported with Shack-Hartmann autofocus"};
    }

    std::optional<Padding> padding;
    if (s_.asp_padding_enabled) {
      padding = Padding{.width = s_.asp_padded_width, .height = s_.asp_padded_height};
    }

    std::optional<Filter> filter;
    if (s_.filter_2d) {
      filter = Filter{.r_inner = s_.filter_r_inner,
                      .r_outer = s_.filter_r_outer,
                      .s_inner = s_.filter_smooth_inner,
                      .s_outer = s_.filter_smooth_outer};
    }

    return angular_spectrum(FH, {.lambda  = s_.spacial_lambda,
                                 .dx      = s_.spacial_pixel_size,
                                 .dy      = s_.spacial_pixel_size,
                                 .z       = s_.spacial_z,
                                 .filter  = filter,
                                 .padding = padding});
  }

  HOLOVIBES_UNREACHABLE();
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_spatial_filter(const TDesc &FH_z) {
  return filter_2d(FH_z, {
                             s_.filter_r_inner,
                             s_.filter_r_outer,
                             s_.filter_smooth_inner,
                             s_.filter_smooth_outer,
                         });
}

GraphBuilder::Impl::TDesc GraphBuilder::Impl::build_freq_weights() {
  auto N  = static_cast<double>(s_.time_window);
  auto fs = 37e3;
  auto df = fs / N;
  auto f0 = s_.time_z_begin * df;
  auto f1 = s_.time_z_end * df;

  if (s_.view_3d_cuts) {
    throw std::logic_error{"Frequency weights are not supported when 3D cuts are enabled"};
  }

  if (s_.time_method == TimeMethod::FFT) {
    auto freqs_pos = arange({f0, f1, df, holoflow::core::DType::F32});
    auto freqs_neg = arange({f0 - fs, f1 - fs, df, holoflow::core::DType::F32});
    auto freqs     = concatenate(std::array<TDesc, 2>{freqs_pos, freqs_neg}, {0});
    return freqs;
  }

  if (s_.time_method == TimeMethod::RFFT ||
      s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS) {
    auto freqs = arange({f0, f1, df, holoflow::core::DType::F32});
    return freqs;
  }

  HOLOVIBES_UNREACHABLE();
}

void GraphBuilder::Impl::build_xy_view(const TDesc &FH_z) {
  using Target                 = holotask::syncs::ConversionSettings::Target;
  using Strat                  = holotask::syncs::ConversionSettings::Strategy;
  using SlidingAverageSettings = holotask::asyncs::SlidingAverageSettings;
  using PctClipSettings        = holotask::syncs::PctClipSettings;
  using Ellipse                = PctClipSettings::Ellipse;
  using holotask::sinks::HolofileSettings;
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  TDesc      result;
  const bool is_magnitude = FH_z.dtype == holoflow::core::DType::F32;

  if (s_.moment_type == MomentType::M0) {
    result = is_magnitude ? mean(FH_z, {{-3}, true}) : mean_abs(FH_z, {{-3}, true});
  }

  else if (s_.moment_type == MomentType::M1) {
    auto n_freq   = static_cast<int64_t>(FH_z.shape.at(1));
    auto abs_S    = is_magnitude ? FH_z : abs(FH_z, {});
    auto freqs    = reshape(build_freq_weights(), {{1, n_freq, 1, 1}});
    auto weighted = multiply(freqs, abs_S, {});
    result        = mean(weighted, {{-3}, true});
  }

  else if (s_.moment_type == MomentType::M2) {
    auto n_freq   = static_cast<int64_t>(FH_z.shape.at(1));
    auto abs_S    = is_magnitude ? FH_z : abs(FH_z, {});
    auto freqs    = reshape(build_freq_weights(), {{1, n_freq, 1, 1}});
    freqs         = multiply(freqs, freqs, {});
    auto weighted = multiply(freqs, abs_S, {});
    result        = mean(weighted, {{-3}, true});
  }

  else {
    HOLOVIBES_UNREACHABLE();
  }

  if (s_.pp_fft_shift) {
    result = fftshift(result, {{-2, -1}});
  }

  if (s_.pp_flatfield) {
    auto cutoff         = s_.pp_flatfield_cutoff_period_m;
    const auto [dy, dx] = post_propagation_pitch(s_, FH_z);
    result              = flatfield(result, flatfield_settings_from_cutoff_period(cutoff, dy, dx));
  }

  if (s_.pp_registration) {
    throw std::logic_error{"Registration is currently not supported"};
  }

  result = mean(result, {{0}, false}); // [1, H, W]

  // auto target_capacity = static_cast<size_t>(std::max(1, s_.gpu_out_size));
  auto target_capacity = 8ULL;
  auto window_size     = static_cast<size_t>(s_.pp_accumulation);
  auto discard_first   = s_.autofocus_enabled ? window_size - 1 : 0;
  auto slide_settings  = SlidingAverageSettings{target_capacity, window_size, discard_first};
  result               = slide_avg(result, slide_settings);

  if (s_.pp_convolution) {
    throw std::logic_error{"Convolution is currently not supported"};
  }

  if (s_.pp_pctclip) {
    Ellipse         roi{0.5f, 0.5f, s_.pp_pctclip_radius, s_.pp_pctclip_radius, 0.0f};
    PctClipSettings pct_clip_settings{s_.pp_pctclip_lower, s_.pp_pctclip_upper, roi};
    result = pct_clip(result, pct_clip_settings);
  }

  result = convert(result, {Target::U8, Strat::Scaled});
  result = batched_queue(result, {s_.gpu_out_size, 1, 1});
  xy_processed_display(result, {});

  if (s_.recording_method == RecordingMethod::PROCESSED) {
    auto result_rec = memcpy(result, {Host});
    result_rec      = batched_queue(result_rec, {s_.cpu_out_size, 1, 1});

    auto path              = s_.recording_path.string();
    auto count             = s_.recording_count;
    auto settings_json     = settings_to_old_json(s_);
    auto holofile_settings = HolofileSettings{path, count, settings_json};
    holofile_write(result_rec, holofile_settings);
  }
}

void GraphBuilder::Impl::build_3d_cuts(const TDesc &FH_z) {
  (void)FH_z;
  throw std::logic_error{"3D cuts are currently not supported in GraphBuilder"};
}

// -----------------------------------------------------------------------------
// Helpers definitions
// -----------------------------------------------------------------------------

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

} // namespace holovibes::pipeline
