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

#pragma once

#include <filesystem>
#include <optional>
#include <variant>

namespace holovibes::pipeline {

enum class ImportSource {
  AMETEK_S710_EURESYS_COAXLINK_OCTO,
  AMETEK_S711_EURESYS_COAXLINK_QSFP,
  HOLOFILE,
};

enum class LoadMethod {
  READ_LIVE,
  LOAD_IN_CPU,
  LOAD_IN_GPU,
};

enum class SpacialMethod {
  NONE,
  FRESNEL_DIFFRACTION,
  ANGULAR_SPECTRUM,
};

enum class TimeMethod {
  NONE,
  PRINCIPAL_COMPONENT_ANALYSIS,
  RFFT,
  FFT,
};

enum class RecordingMethod {
  NONE,
  RAW,
  PROCESSED,
};

enum class ViewType {
  PROCESSED,
  RAW,
};

enum class MomentType {
  M0,
  M1,
  M2,
};

struct Settings {
  // Advanced
  int cpu_in_size;
  int gpu_in_size;
  int cpu_rec_size;
  int cpu_out_size;
  int gpu_out_size;
  int time_accumulation;

  // Import
  ImportSource          import_source;
  std::filesystem::path load_path;
  LoadMethod            load_method;
  int                   load_begin;
  int                   load_end;
  int                   load_batch;
  std::optional<int>    load_fps_limit;
  double                input_sampling_frequency_hz = 1.0e6 / 27.0;
  std::filesystem::path camera_config_path;

  // Spacial Propagation
  SpacialMethod spacial_method;
  float         spacial_lambda;
  float         spacial_z;
  float         spacial_pixel_size;

  // Spatial Filter
  bool filter_2d;
  int  filter_r_inner;
  int  filter_r_outer;
  int  filter_smooth_inner;
  int  filter_smooth_outer;

  // Temporal Filter
  TimeMethod time_method;
  int        time_window;
  int        time_stride;

  // Cuts views
  int time_x_begin;
  int time_x_end;
  int time_y_begin;
  int time_y_end;
  int time_z_begin;
  int time_z_end;

  // View
  bool       view_3d_cuts;
  bool       raw_view;
  ViewType   view_type;
  MomentType moment_type;
  bool       view_raw_spectrum;
  bool       view_processed_spectrum;
  bool       view_zernike_metrics;
  bool       view_zernike_phase;
  bool       view_shack_hartmann;
  bool       view_shack_hartmann_xcorr;

  // Post-processing
  int  pp_fps;
  bool pp_fft_shift;
  bool pp_flatfield;
  // Physical cutoff period used to derive Gaussian sigmas at the current image pitch.
  float       pp_flatfield_cutoff_period_m;
  int         pp_accumulation;
  bool        pp_convolution;
  std::string pp_convolution_path;
  bool        pp_convolution_divide;
  bool        pp_pctclip;
  float       pp_pctclip_lower;
  float       pp_pctclip_upper;
  float       pp_pctclip_radius;
  bool        pp_registration;
  float       pp_registration_radius;

  // Recording
  RecordingMethod       recording_method;
  std::filesystem::path recording_path;
  int                   recording_count;

  // Auto-focus
  bool             autofocus_enabled;
  int              autofocus_nb_subaps;
  int              autofocus_nb_iter;
  std::vector<int> autofocus_zernike_orders;
  bool             autofocus_skip_subapertures_outside_pupil = true;
  bool             autofocus_use_graph_laplacian             = false;

  // Zernike coefficient history.
  double signal_plot_time_window_seconds = 8.0;

  [[nodiscard]] double signal_plot_sample_time_seconds() const {
    return static_cast<double>(time_stride) * static_cast<double>(pp_accumulation) /
           input_sampling_frequency_hz;
  }
};

} // namespace holovibes::pipeline
