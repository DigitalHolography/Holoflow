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
  SHORT_TIME_FOURIER,
};

struct Settings {
  // Advanced
  int cpu_in_size;
  int gpu_in_size;
  int cpu_rec_size;
  int cpu_out_size;
  int gpu_out_size;

  // Import
  ImportSource          import_source;
  std::filesystem::path load_path;
  LoadMethod            load_method;
  int                   load_begin;
  int                   load_end;
  int                   load_batch;
  std::filesystem::path camera_config_path;

  // Spacial
  SpacialMethod spacial_method;
  float         spacial_lambda;
  float         spacial_z;
  float         spacial_pixel_size;

  // Filter
  bool filter_2d;
  int  filter_r_inner;
  int  filter_r_outer;
  int  filter_smooth_inner;
  int  filter_smooth_outer;

  // Temporal
  TimeMethod time_method;
  int        time_window;
  int        time_stride;
  int        time_x_begin;
  int        time_x_end;
  int        time_y_begin;
  int        time_y_end;
  int        time_z_begin;
  int        time_z_end;

  // View
  bool view_3d_cuts;

  // Post-processing
  int         pp_fps;
  bool        pp_fft_shift;
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
};

} // namespace holovibes::pipeline