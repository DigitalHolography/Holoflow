#pragma once

#include <optional>
#include <string>

namespace holovibes::pipeline {

enum class ImportLoadMethod {
  ReadLive,
  LoadInCPU,
  LoadInGPU,
};

enum class ExportImageType {
  Raw,
  Processed,
};

enum class ExportTag {
  LeftEye,
  RightEye,
};

enum class RenderType {
  Raw,
  Processed,
};

enum class RenderSpaceTransform {
  FresnelDiffraction,
  AngularSpectrum,
};

enum class RenderTimeTransform {
  ShortTimeFourier,
  PrincipalComponentAnalysis,
};

enum class RenderConvolution {
  Gaussian,
};

enum class ViewType {
  Magnitude,
};

enum class ViewAxis {
  XY,
  XZ,
  YZ,
};

struct Settings {
  // --- Settings for Import ---
  std::string import_file_path;
  size_t import_fps;
  size_t import_start_index;
  size_t import_end_index;
  ImportLoadMethod import_load_method;

  // --- Settings for Export ---
  ExportImageType export_image_type;
  std::string export_file_path;
  ExportTag export_tag;
  std::optional<size_t> export_frame_count;

  // --- Settings for Image Rendering ---
  RenderType render_type;
  size_t render_batch_size;
  size_t render_time_stride;
  bool render_filter_2d;
  std::optional<RenderSpaceTransform> render_space_transform;
  std::optional<RenderTimeTransform> render_time_transform;
  size_t render_time_window;
  size_t render_lambda;
  size_t render_focus;
  std::optional<RenderConvolution> render_convolution;
  bool render_convolution_divide;

  // ---Settings for View ---
  ViewType view_type;
  bool view_cuts_3d;
  bool view_fft_shift;
  bool view_lens_view;
  bool view_raw_view;
  size_t view_p_frame_start;
  size_t view_p_frame_width;
  ViewAxis view_axis;
  size_t view_accumulation;
  bool view_auto_contrast;
  bool view_invert_contrast;
  size_t view_contrast_low;
  size_t view_contrast_high;
  bool view_renormalize;
};

} // namespace holovibes::pipeline