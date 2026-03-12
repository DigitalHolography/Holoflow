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
namespace {

using json = nlohmann::json;

[[nodiscard]] const json &child_or_empty(const json &j, const char *key) {
  static const json kEmpty = json::object();
  if (!j.is_object() || !j.contains(key) || !j.at(key).is_object()) {
    return kEmpty;
  }
  return j.at(key);
}

[[nodiscard]] int read_range_end(const json &node, int begin, int default_end) {
  constexpr int kMissingWidth = -1;
  const int     width         = val(node, "width", kMissingWidth);
  return width == kMissingWidth ? default_end : begin + width;
}

[[nodiscard]] std::string to_legacy_space_transform(SpacialMethod method) {
  switch (method) {
  case SpacialMethod::FRESNEL_DIFFRACTION:
    return "FRESNELTR";
  case SpacialMethod::ANGULAR_SPECTRUM:
    return "ANGULAR";
  case SpacialMethod::NONE:
  default:
    return "NONE";
  }
}

[[nodiscard]] SpacialMethod from_legacy_space_transform(const std::string &value) {
  if (value == "FRESNELTR") {
    return SpacialMethod::FRESNEL_DIFFRACTION;
  }
  if (value == "ANGULAR") {
    return SpacialMethod::ANGULAR_SPECTRUM;
  }
  return SpacialMethod::NONE;
}

[[nodiscard]] std::string to_legacy_time_transform(TimeMethod method) {
  switch (method) {
  case TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS:
    return "PCA";
  case TimeMethod::SHORT_TIME_FOURIER:
    return "STFT";
  case TimeMethod::NONE:
  default:
    return "NONE";
  }
}

[[nodiscard]] TimeMethod from_legacy_time_transform(const std::string &value) {
  if (value == "PCA") {
    return TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS;
  }
  if (value == "STFT") {
    return TimeMethod::SHORT_TIME_FOURIER;
  }
  return TimeMethod::NONE;
}

void write_advanced(json &j, const Settings &s) {
  auto &advanced = j["compute_settings"]["advanced"];

  advanced["buffer_size"]["input"]  = s.gpu_in_size;
  advanced["buffer_size"]["output"] = s.gpu_out_size;
  advanced["buffer_size"]["record"] = s.cpu_rec_size;

  advanced["filter2d_smooth"]["low"]  = s.filter_smooth_inner;
  advanced["filter2d_smooth"]["high"] = s.filter_smooth_outer;

  advanced["nb_frames_to_record"] = s.recording_count;
}

void write_image_rendering(json &j, const Settings &s) {
  auto &rendering = j["compute_settings"]["image_rendering"];

  rendering["batch_size"] = s.load_batch;

  rendering["convolution"]["divide"] = s.pp_convolution_divide;
  rendering["convolution"]["type"]   = s.pp_convolution ? s.pp_convolution_path : "";

  rendering["filter2d"]["enabled"]      = s.filter_2d;
  rendering["filter2d"]["inner_radius"] = s.filter_r_inner;
  rendering["filter2d"]["outer_radius"] = s.filter_r_outer;

  // Legacy format expects this field even if space transformation is NONE.
  rendering["image_mode"] = "HOLOGRAM";

  rendering["lambda"]               = s.spacial_lambda;
  rendering["propagation_distance"] = s.spacial_z;

  rendering["space_transformation"]       = to_legacy_space_transform(s.spacial_method);
  rendering["time_transformation"]        = to_legacy_time_transform(s.time_method);
  rendering["time_transformation_size"]   = s.time_window;
  rendering["time_transformation_stride"] = s.time_stride;
}

void write_view(json &j, const Settings &s) {
  auto &view = j["compute_settings"]["view"];

  view["fft_shift"] = s.pp_fft_shift;

  view["registration"]["registration_enabled"] = s.pp_registration;
  view["registration"]["registration_zone"]    = s.pp_registration_radius;

  view["window"]["filter2d"]["contrast"]["enabled"] = s.pp_pctclip;

  view["window"]["xy"]["contrast"]["enabled"]       = s.pp_pctclip;
  view["window"]["xy"]["contrast"]["min"]           = s.pp_pctclip_lower;
  view["window"]["xy"]["contrast"]["max"]           = s.pp_pctclip_upper;
  view["window"]["xy"]["output_image_accumulation"] = s.pp_accumulation;

  view["window"]["xz"]["enabled"] = s.view_3d_cuts;
  view["window"]["yz"]["enabled"] = s.view_3d_cuts;

  view["x"]["start"] = s.time_x_begin;
  view["x"]["width"] = s.time_x_end - s.time_x_begin;

  view["y"]["start"] = s.time_y_begin;
  view["y"]["width"] = s.time_y_end - s.time_y_begin;

  view["z"]["start"] = s.time_z_begin;
  view["z"]["width"] = s.time_z_end - s.time_z_begin;
}

void write_info(json &j, const Settings &s) {
  auto &info = j["info"];

  info["contiguous"]       = s.cpu_rec_size;
  info["pixel_pitch"]["x"] = s.spacial_pixel_size;
  info["pixel_pitch"]["y"] = s.spacial_pixel_size;
}

void read_advanced(Settings &s, const json &advanced) {
  const auto &buffer_size = child_or_empty(advanced, "buffer_size");
  const auto &smooth      = child_or_empty(advanced, "filter2d_smooth");

  s.gpu_in_size  = val(buffer_size, "input", s.gpu_in_size);
  s.gpu_out_size = val(buffer_size, "output", s.gpu_out_size);
  s.cpu_rec_size = val(buffer_size, "record", s.cpu_rec_size);

  s.filter_smooth_inner = val(smooth, "low", s.filter_smooth_inner);
  s.filter_smooth_outer = val(smooth, "high", s.filter_smooth_outer);

  s.recording_count = val(advanced, "nb_frames_to_record", s.recording_count);
}

void read_image_rendering(Settings &s, const json &rendering) {
  s.load_batch = val(rendering, "batch_size", s.load_batch);

  s.spacial_method = from_legacy_space_transform(val(rendering, "space_transformation", "NONE"));
  s.spacial_lambda = val(rendering, "lambda", s.spacial_lambda);
  s.spacial_z      = val(rendering, "propagation_distance", s.spacial_z);

  s.time_method = from_legacy_time_transform(val(rendering, "time_transformation", "NONE"));
  s.time_window = val(rendering, "time_transformation_size", s.time_window);
  s.time_stride = val(rendering, "time_transformation_stride", s.time_stride);

  const auto &filter2d = child_or_empty(rendering, "filter2d");
  s.filter_2d          = val(filter2d, "enabled", s.filter_2d);
  s.filter_r_inner     = val(filter2d, "inner_radius", s.filter_r_inner);
  s.filter_r_outer     = val(filter2d, "outer_radius", s.filter_r_outer);

  const auto &convolution = child_or_empty(rendering, "convolution");
  s.pp_convolution_path   = val(convolution, "type", s.pp_convolution_path);
  s.pp_convolution        = !s.pp_convolution_path.empty();
  s.pp_convolution_divide = val(convolution, "divide", s.pp_convolution_divide);
}

void read_view_ranges(Settings &s, const json &view) {
  const auto &x = child_or_empty(view, "x");
  const auto &y = child_or_empty(view, "y");
  const auto &z = child_or_empty(view, "z");

  s.time_x_begin = val(x, "start", s.time_x_begin);
  s.time_y_begin = val(y, "start", s.time_y_begin);
  s.time_z_begin = val(z, "start", s.time_z_begin);

  s.time_x_end = read_range_end(x, s.time_x_begin, s.time_x_end);
  s.time_y_end = read_range_end(y, s.time_y_begin, s.time_y_end);
  s.time_z_end = read_range_end(z, s.time_z_begin, s.time_z_end);
}

void read_view(Settings &s, const json &view) {
  s.pp_fft_shift = val(view, "fft_shift", s.pp_fft_shift);

  const auto &window = child_or_empty(view, "window");
  const auto &xy     = child_or_empty(window, "xy");
  const auto &xz     = child_or_empty(window, "xz");
  const auto &yz     = child_or_empty(window, "yz");

  s.view_3d_cuts = val(xz, "enabled", s.view_3d_cuts) || val(yz, "enabled", s.view_3d_cuts);

  s.pp_accumulation = val(xy, "output_image_accumulation", s.pp_accumulation);

  const auto &contrast = child_or_empty(xy, "contrast");
  s.pp_pctclip         = val(contrast, "enabled", s.pp_pctclip);
  s.pp_pctclip_lower   = val(contrast, "min", s.pp_pctclip_lower);
  s.pp_pctclip_upper   = val(contrast, "max", s.pp_pctclip_upper);

  const auto &registration = child_or_empty(view, "registration");
  s.pp_registration        = val(registration, "registration_enabled", s.pp_registration);
  s.pp_registration_radius = val(registration, "registration_zone", s.pp_registration_radius);

  const auto &reticle = child_or_empty(view, "reticle");
  s.pp_pctclip_radius = val(reticle, "scale", s.pp_pctclip_radius);

  read_view_ranges(s, view);
}

void read_info(Settings &s, const json &info) {
  const auto &pixel_pitch = child_or_empty(info, "pixel_pitch");
  s.spacial_pixel_size    = val(pixel_pitch, "x", s.spacial_pixel_size);
}

} // namespace

nlohmann::json settings_to_old_json(const Settings &settings) {
  json j = json::object();

  write_advanced(j, settings);
  write_image_rendering(j, settings);
  write_view(j, settings);
  write_info(j, settings);

  return j;
}

Settings old_json_to_settings(const nlohmann::json &j, const Settings &default_settings) {
  Settings settings = default_settings;

  const auto &compute   = child_or_empty(j, "compute_settings");
  const auto &advanced  = child_or_empty(compute, "advanced");
  const auto &rendering = child_or_empty(compute, "image_rendering");
  const auto &view      = child_or_empty(compute, "view");
  const auto &info      = child_or_empty(j, "info");

  read_advanced(settings, advanced);
  read_image_rendering(settings, rendering);
  read_view(settings, view);
  read_info(settings, info);

  return settings;
}

} // namespace holovibes::pipeline