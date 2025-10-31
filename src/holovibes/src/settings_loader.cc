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

#include "settings_loader.hh"

namespace holovibes::pipeline {

// Convert new Settings to old JSON format
nlohmann::json settings_to_old_json(const Settings &settings) {
  nlohmann::json j;

  // Advanced section
  j["compute_settings"]["advanced"]["buffer_size"]["input"]  = settings.gpu_in_size;
  j["compute_settings"]["advanced"]["buffer_size"]["output"] = settings.gpu_out_size;
  j["compute_settings"]["advanced"]["buffer_size"]["record"] = settings.cpu_rec_size;
  // FIXME: buffer_size.file - no mapping found
  //   j["compute_settings"]["advanced"]["buffer_size"]["file"] = 1024;
  // FIXME: buffer_size.time_transformation_cuts - no mapping found
  //   j["compute_settings"]["advanced"]["buffer_size"]["time_transformation_cuts"] = 512;

  // FIXME: contrast.frame_index_offset - no mapping found
  //   j["compute_settings"]["advanced"]["contrast"]["frame_index_offset"] = 2;
  // FIXME: contrast.lower - no mapping found
  //   j["compute_settings"]["advanced"]["contrast"]["lower"] = 0.02;
  // FIXME: contrast.upper - no mapping found
  //   j["compute_settings"]["advanced"]["contrast"]["upper"] = 99.98;

  // FIXME: make sure it s good
  j["compute_settings"]["advanced"]["filter2d_smooth"]["high"] = settings.filter_smooth_outer;
  j["compute_settings"]["advanced"]["filter2d_smooth"]["low"]  = settings.filter_smooth_inner;

  // FIXME: nb_frames_to_record - closest is recording_count but context differs
  j["compute_settings"]["advanced"]["nb_frames_to_record"] = settings.recording_count;
  // FIXME: raw_bitshift - no mapping found
  //   j["compute_settings"]["advanced"]["raw_bitshift"] = 0;
  // FIXME: renorm_constant - no mapping found
  //   j["compute_settings"]["advanced"]["renorm_constant"] = 5;

  // FIXME: entire color_composite_image section - no mapping found
  //   j["compute_settings"]["color_composite_image"]["auto_weight"] = false;
  //   j["compute_settings"]["color_composite_image"]["mode"]        = "RGB";
  // ... (HSV/RGB subsections omitted for brevity, all FIXMEs)

  // Image rendering section
  // FIXME: batch_size - closest is load_batch but context differs
  j["compute_settings"]["image_rendering"]["batch_size"] = settings.load_batch;

  j["compute_settings"]["image_rendering"]["convolution"]["divide"] =
      settings.pp_convolution_divide;
  j["compute_settings"]["image_rendering"]["convolution"]["type"] =
      settings.pp_convolution ? settings.pp_convolution_path : "";

  j["compute_settings"]["image_rendering"]["filter2d"]["enabled"]      = settings.filter_2d;
  j["compute_settings"]["image_rendering"]["filter2d"]["inner_radius"] = settings.filter_r_inner;
  j["compute_settings"]["image_rendering"]["filter2d"]["outer_radius"] = settings.filter_r_outer;

  // FIXME: image_mode - no mapping found, inferring from spacial_method
  std::string image_mode = "HOLOGRAM";
  if (settings.spacial_method != SpacialMethod::NONE) {
    image_mode = "HOLOGRAM";
  }
  j["compute_settings"]["image_rendering"]["image_mode"] = image_mode;

  // FIXME: input_filter.type - no mapping found
  //   j["compute_settings"]["image_rendering"]["input_filter"]["type"] = "None";

  j["compute_settings"]["image_rendering"]["lambda"]               = settings.spacial_lambda;
  j["compute_settings"]["image_rendering"]["propagation_distance"] = settings.spacial_z;

  // Map spacial method
  std::string space_transform;
  switch (settings.spacial_method) {
  case SpacialMethod::FRESNEL_DIFFRACTION:
    space_transform = "FRESNELTR";
    break;
  case SpacialMethod::ANGULAR_SPECTRUM:
    space_transform = "ANGULAR";
    break;
  default:
    space_transform = "NONE";
  }
  j["compute_settings"]["image_rendering"]["space_transformation"] = space_transform;

  // Map time method
  std::string time_transform;
  switch (settings.time_method) {
  case TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS:
    time_transform = "PCA";
    break;
  case TimeMethod::SHORT_TIME_FOURIER:
    time_transform = "STFT";
    break;
  default:
    time_transform = "NONE";
  }
  j["compute_settings"]["image_rendering"]["time_transformation"]        = time_transform;
  j["compute_settings"]["image_rendering"]["time_transformation_size"]   = settings.time_window;
  j["compute_settings"]["image_rendering"]["time_transformation_stride"] = settings.time_stride;

  //   j["compute_settings"]["version"] = "v1";

  // View section
  j["compute_settings"]["view"]["fft_shift"] = settings.pp_fft_shift;
  // FIXME: image_type - no mapping found
  //   j["compute_settings"]["view"]["image_type"] = "MODULUS";

  j["compute_settings"]["view"]["registration"]["registration_enabled"] = settings.pp_registration;
  j["compute_settings"]["view"]["registration"]["registration_zone"] =
      settings.pp_registration_radius;

  // FIXME: renorm - closest might be pp_pctclip but not clear
  //   j["compute_settings"]["view"]["renorm"] = true;

  // FIXME: reticle section - no mapping found
  //   j["compute_settings"]["view"]["reticle"]["display_enabled"] = false;
  //   j["compute_settings"]["view"]["reticle"]["scale"]           = 0.5;

  // Window sections
  // FIXME: filter2d window contrast - different from pp_pctclip
  //   j["compute_settings"]["view"]["window"]["filter2d"]["contrast"]["auto_refresh"] = false;
  j["compute_settings"]["view"]["window"]["filter2d"]["contrast"]["enabled"] = settings.pp_pctclip;
  //   j["compute_settings"]["view"]["window"]["filter2d"]["contrast"]["invert"]  = false;
  // FIXME: contrast min/max values
  //   j["compute_settings"]["view"]["window"]["filter2d"]["contrast"]["max"] = 65535.0;
  //   j["compute_settings"]["view"]["window"]["filter2d"]["contrast"]["min"] = 1.0;
  //   j["compute_settings"]["view"]["window"]["filter2d"]["log_enabled"]     = false;

  // XY window
  //   j["compute_settings"]["view"]["window"]["xy"]["contrast"]["auto_refresh"] = false;
  j["compute_settings"]["view"]["window"]["xy"]["contrast"]["enabled"] = settings.pp_pctclip;
  //   j["compute_settings"]["view"]["window"]["xy"]["contrast"]["invert"]       = false;
  // FIXME: xy contrast values - using pctclip but units may differ
  j["compute_settings"]["view"]["window"]["xy"]["contrast"]["max"] = settings.pp_pctclip_upper;
  j["compute_settings"]["view"]["window"]["xy"]["contrast"]["min"] = settings.pp_pctclip_lower;
  //   j["compute_settings"]["view"]["window"]["xy"]["enabled"]         = false;
  //   j["compute_settings"]["view"]["window"]["xy"]["horizontal_flip"] = false;
  //   j["compute_settings"]["view"]["window"]["xy"]["log_enabled"]     = false;
  j["compute_settings"]["view"]["window"]["xy"]["output_image_accumulation"] =
      settings.pp_accumulation;
  // FIXME: rotation - no mapping found
  //   j["compute_settings"]["view"]["window"]["xy"]["rotation"] = 0.0;

  // FIXME: xz and yz windows - no specific mappings
  j["compute_settings"]["view"]["window"]["xz"]["enabled"] = settings.view_3d_cuts;
  j["compute_settings"]["view"]["window"]["yz"]["enabled"] = settings.view_3d_cuts;

  // Coordinate ranges
  //   j["compute_settings"]["view"]["x"]["start"] = 0; // FIXME: no mapping for x range
  //   j["compute_settings"]["view"]["x"]["width"] = 0;
  //   j["compute_settings"]["view"]["y"]["start"] = 0; // FIXME: no mapping for y range
  //   j["compute_settings"]["view"]["y"]["width"] = 0;
  j["compute_settings"]["view"]["z"]["start"] = settings.time_z_begin;
  j["compute_settings"]["view"]["z"]["width"] = settings.time_z_end - settings.time_z_begin;
  // FIXME: z2 - no mapping found
  // j["compute_settings"]["view"]["z2"]["start"] = 0;
  // j["compute_settings"]["view"]["z2"]["width"] = 0;

  // Info section
  // FIXME: camera_fps - no mapping found
  //   j["info"]["camera_fps"] = 37037;
  // FIXME: contiguous - no mapping found
  j["info"]["contiguous"] = settings.cpu_rec_size;
  // FIXME: eye_type - no mapping found
  //   j["info"]["eye_type"] = "LEFT";
  // FIXME: get the version
  //   j["info"]["holovibes_version"] = "0.0.0";
  // FIXME: input_fps - no mapping found
  //   j["info"]["input_fps"]        = 37037;
  j["info"]["pixel_pitch"]["x"] = settings.spacial_pixel_size;
  j["info"]["pixel_pitch"]["y"] = settings.spacial_pixel_size;

  return j;
}

Settings old_json_to_settings(const nlohmann::json &j, const Settings &default_settings) {
  Settings settings{};

  // Shortcuts for readability
  const auto &compute =
      j.contains("compute_settings") ? j.at("compute_settings") : nlohmann::json{};
  const auto &adv  = compute.value("advanced", nlohmann::json{});
  const auto &buf  = adv.value("buffer_size", nlohmann::json{});
  const auto &rend = compute.value("image_rendering", nlohmann::json{});
  const auto &view = compute.value("view", nlohmann::json{});
  const auto &info = j.value("info", nlohmann::json{});

  // --- Advanced ---
  settings.cpu_in_size  = default_settings.cpu_in_size;
  settings.gpu_in_size  = val(buf, "input", default_settings.gpu_in_size);
  settings.cpu_rec_size = val(buf, "record", default_settings.cpu_rec_size);
  settings.cpu_out_size = default_settings.cpu_out_size;
  settings.gpu_out_size = val(buf, "output", default_settings.gpu_out_size);

  // --- Import ---
  settings.import_source      = default_settings.import_source;
  settings.load_path          = default_settings.load_path;
  settings.load_method        = default_settings.load_method;
  settings.load_begin         = default_settings.load_begin;
  settings.load_end           = default_settings.load_end;
  settings.load_batch         = val(rend, "batch_size", default_settings.load_batch);
  settings.camera_config_path = default_settings.camera_config_path;

  // --- Spatial ---
  {
    std::string space_transform = val(rend, "space_transformation", "");
    if (space_transform == "FRESNELTR")
      settings.spacial_method = SpacialMethod::FRESNEL_DIFFRACTION;
    else if (space_transform == "ANGULAR")
      settings.spacial_method = SpacialMethod::ANGULAR_SPECTRUM;
    else
      settings.spacial_method = SpacialMethod::NONE;
  }

  settings.spacial_lambda = val(rend, "lambda", default_settings.spacial_lambda);
  settings.spacial_z      = val(rend, "propagation_distance", default_settings.spacial_z);
  settings.spacial_pixel_size =
      val(info.value("pixel_pitch", nlohmann::json{}), "x", default_settings.spacial_pixel_size);

  // --- Filter ---
  const auto &f2d         = rend.value("filter2d", nlohmann::json{});
  settings.filter_2d      = val(f2d, "enabled", default_settings.filter_2d);
  settings.filter_r_inner = val(f2d, "inner_radius", default_settings.filter_r_inner);
  settings.filter_r_outer = val(f2d, "outer_radius", default_settings.filter_r_outer);

  const auto &smooth           = adv.value("filter2d_smooth", nlohmann::json{});
  settings.filter_smooth_inner = val(smooth, "low", default_settings.filter_smooth_inner);
  settings.filter_smooth_outer = val(smooth, "high", default_settings.filter_smooth_outer);

  // --- Temporal ---
  {
    std::string time_transform = val(rend, "time_transformation", "");
    if (time_transform == "PCA")
      settings.time_method = TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS;
    else if (time_transform == "STFT")
      settings.time_method = TimeMethod::SHORT_TIME_FOURIER;
    else
      settings.time_method = TimeMethod::NONE;
  }

  settings.time_window  = val(rend, "time_transformation_size", default_settings.time_window);
  settings.time_stride  = val(rend, "time_transformation_stride", default_settings.time_stride);
  settings.time_x_begin = default_settings.time_x_begin;
  settings.time_x_end   = default_settings.time_x_end;
  settings.time_y_begin = default_settings.time_y_begin;
  settings.time_y_end   = default_settings.time_y_end;

  const auto &zview     = view.value("z", nlohmann::json{});
  settings.time_z_begin = val(zview, "start", default_settings.time_z_begin);
  settings.time_z_end   = settings.time_z_begin + static_cast<int>(val(zview, "width", 0));

  // --- View ---
  const auto &win       = view.value("window", nlohmann::json{});
  const auto &xz        = win.value("xz", nlohmann::json{});
  const auto &yz        = win.value("yz", nlohmann::json{});
  settings.view_3d_cuts = val(xz, "enabled", false) || val(yz, "enabled", false);

  // --- Post-processing ---
  const auto &xy           = win.value("xy", nlohmann::json{});
  settings.pp_fps          = default_settings.pp_fps;
  settings.pp_fft_shift    = val(view, "fft_shift", default_settings.pp_fft_shift);
  settings.pp_accumulation = val(xy, "output_image_accumulation", default_settings.pp_accumulation);

  const auto &conv               = rend.value("convolution", nlohmann::json{});
  std::string conv_type          = val(conv, "type", default_settings.pp_convolution_path);
  settings.pp_convolution        = !conv_type.empty();
  settings.pp_convolution_path   = conv_type;
  settings.pp_convolution_divide = val(conv, "divide", default_settings.pp_convolution_divide);

  const auto &contrast       = xy.value("contrast", nlohmann::json{});
  settings.pp_pctclip        = val(contrast, "enabled", default_settings.pp_pctclip);
  settings.pp_pctclip_lower  = val(contrast, "min", default_settings.pp_pctclip_lower);
  settings.pp_pctclip_upper  = val(contrast, "max", default_settings.pp_pctclip_upper);
  settings.pp_pctclip_radius = default_settings.pp_pctclip_radius;

  const auto &reg          = view.value("registration", nlohmann::json{});
  settings.pp_registration = val(reg, "registration_enabled", default_settings.pp_registration);
  settings.pp_registration_radius =
      val(reg, "registration_zone", default_settings.pp_registration_radius);

  // --- Recording ---
  settings.recording_method = default_settings.recording_method;
  settings.recording_path   = default_settings.recording_path;
  settings.recording_count  = val(adv, "nb_frames_to_record", default_settings.recording_count);

  return settings;
}

} // namespace holovibes::pipeline