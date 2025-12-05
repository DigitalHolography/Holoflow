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

#include "graph_builder.hh"
#include "bug.hh"
#include "logger.hh"
#include "settings_loader.hh"
#include "tasks/asyncs/batch_queue.hh"
#include "tasks/asyncs/slide_avg.hh"
#include "tasks/sinks/display_tensor.hh"
#include "tasks/sinks/holofile.hh"
#include "tasks/sources/ametek_s710_euresys_coaxlink_octo.hh"
#include "tasks/sources/ametek_s711_euresys_coaxlink_qsfp+.hh"
#include "tasks/sources/holofile.hh"
#include "tasks/syncs/angular_spectrum.hh"
#include "tasks/syncs/average.hh"
#include "tasks/syncs/conversion.hh"
#include "tasks/syncs/convolution.hh"
#include "tasks/syncs/crop.hh"
#include "tasks/syncs/fft_shift.hh"
#include "tasks/syncs/filter2d.hh"
#include "tasks/syncs/fresnel_diffraction.hh"
#include "tasks/syncs/memcpy.hh"
#include "tasks/syncs/pca.hh"
#include "tasks/syncs/pct_clip.hh"
#include "tasks/syncs/registration.hh"
#include "tasks/syncs/reshape.hh"
#include "tasks/syncs/rotation.hh"
#include "tasks/syncs/stft.hh"

using namespace holovibes::tasks;

namespace holovibes::pipeline {

GraphBuilder::GraphBuilder(holoflow::core::GraphSpec &spec, const Settings &settings, int src_width,
                           int src_height, bool opti_cpu_stride, bool opti_gpu_stride)
    : spec_(spec), s_(settings), src_width_(src_width), src_height_(src_height),
      opti_cpu_stride_(opti_cpu_stride), opti_gpu_stride_(opti_gpu_stride) {}

void GraphBuilder::build() {
  logger()->info("[GraphBuilder::build] Building graph spec...");

  auto source = add_source();
  auto parent = source;

  std::optional<V> cpu_in_queue = std::nullopt;

  if (s_.load_method != LoadMethod::LOAD_IN_GPU) {
    cpu_in_queue = add_cpu_in_queue(parent, 0, 0);
    if (s_.view_type == ViewType::PROCESSED) {
      auto cpu_gpu_cpy = add_cpu_gpu_cpy(*cpu_in_queue, 0, 0);
      parent           = cpu_gpu_cpy;
    }
  }

  // Build recording branch if needed
  if (s_.recording_method == RecordingMethod::RAW && cpu_in_queue) {
    current_section_name_ = "recording::";
    auto cpu_cpu_cpy      = add_cpu_cpu_cpy(*cpu_in_queue, 0, 0);
    auto record_queue     = add_record_queue(cpu_cpu_cpy, 0, 0);
    current_section_name_.clear();
    auto raw_record = add_record(record_queue, 0, 0);
    (void)raw_record;
  }

  // Build raw view branch
  if ((s_.raw_view || s_.view_type == ViewType::RAW) && cpu_in_queue) {
    build_raw_branch(*cpu_in_queue);

    if (s_.view_type == ViewType::RAW) {
      logger()->debug("[GraphBuilder::build] Graph spec built successfully for raw view");
      return;
    }
  }

  // Build processed branches
  auto gpu_in_queue     = add_gpu_in_queue(parent, 0, 0);
  auto processed_parent = build_processed_branch(gpu_in_queue);
  build_xy_branch(processed_parent);

  if (s_.view_3d_cuts) {
    build_xz_branch(processed_parent);
    build_yz_branch(processed_parent);
  }

  logger()->debug("[GraphBuilder::build] Graph spec built successfully");
}

GraphBuilder::V GraphBuilder::build_raw_branch(V cpu_in_queue) {
  current_section_name_ = "raw_view::";
  auto cpu_raw_view_cpy = add_cpu_raw_view_cpy(cpu_in_queue, 0, 0);
  auto raw_view_queue   = add_cpu_raw_queue(cpu_raw_view_cpy, 0, 0);
  auto raw_reshape      = add_raw_reshape(raw_view_queue, 0, 0);

  if (s_.raw_view) {
    auto display = add_xy_raw_display(raw_reshape, 0, 0);
    (void)display;
  }
  if (s_.view_type == ViewType::RAW) {
    auto display = add_xy_processed_display(raw_reshape, 0, 0);
    (void)display;
  }
  current_section_name_.clear();
  return raw_reshape;
}

GraphBuilder::V GraphBuilder::build_processed_branch(V parent) {
  current_section_name_ = "processed_view::";
  auto to_cf32          = add_to_cf32(parent, 0, 0);
  parent                = to_cf32;

  if (s_.filter_2d) {
    auto spacial_filter = add_spacial_filter(parent, 0, 0);
    parent              = spacial_filter;
  }

  if (s_.spacial_method != SpacialMethod::NONE) {
    auto spacial_transform = add_spacial_transform(parent, 0, 0);
    parent                 = spacial_transform;
  }

  if (s_.time_method != TimeMethod::NONE) {
    auto time_queue     = add_time_queue(parent, 0, 0);
    auto time_transform = add_time_transform(time_queue, 0, 0);
    parent              = time_transform;
  }

  auto to_f32         = add_to_f32(parent, 0, 0);
  auto debounce_queue = add_debounce_queue(to_f32, 0, 0);

  current_section_name_.clear();
  return debounce_queue;
}

void GraphBuilder::build_xy_branch(V debounce_queue) {
  current_section_name_ = "xy_processed_view::";
  auto cut_avg          = add_xy_cut_avg(debounce_queue, 0, 0);
  auto parent           = cut_avg;

  if (s_.pp_fft_shift) {
    auto fft_shift = add_fft_shift(parent, 0, 0);
    parent         = fft_shift;
  }

  if (s_.pp_registration) {
    auto registration = add_xy_registration(parent, 0, 0);
    parent            = registration;
  }

  auto slide_avg = add_xy_slide_avg(parent, 0, 0);
  parent         = slide_avg;

  if (s_.pp_convolution) {
    auto convolution = add_xy_convolution(parent, 0, 0);
    parent           = convolution;
  }

  if (s_.pp_pctclip) {
    auto pctclip = add_xy_pctclip(parent, 0, 0);
    parent       = pctclip;
  }

  auto to_u8         = add_xy_to_u8(parent, 0, 0);
  auto gpu_out_queue = add_xy_gpu_out_queue(to_u8, 0, 0);
  auto gpu_cpu       = add_xy_gpu_cpu_cpy(gpu_out_queue, 0, 0);
  auto cpu_out_queue = add_xy_cpu_out_queue(gpu_cpu, 0, 0);
  auto display       = add_xy_processed_display(cpu_out_queue, 0, 0);
  current_section_name_.clear();

  if (s_.recording_method == RecordingMethod::PROCESSED) {
    current_section_name_ = "processed_recording::";
    auto record_cpu_cpu   = add_cpu_cpu_cpy(cpu_out_queue, 0, 0);
    auto record_queue     = add_record_queue(record_cpu_cpu, 0, 0);
    current_section_name_.clear();
    auto record = add_record(record_queue, 0, 0);
    (void)record;
  }

  (void)display;
}

void GraphBuilder::build_xz_branch(V debounce_queue) {
  current_section_name_ = "xz_processed_view::";
  auto cut_avg          = add_xz_cut_avg(debounce_queue, 0, 0);
  auto reshape          = add_xz_reshape(cut_avg, 0, 0);
  auto slide_avg        = add_xz_slide_avg(reshape, 0, 0);
  auto crop             = add_xz_crop2frames(slide_avg, 0, 0);
  auto to_u8            = add_xz_to_u8(crop, 0, 0);
  auto gpu_out          = add_xz_gpu_out_queue(to_u8, 0, 0);
  auto gpu_cpu          = add_xz_gpu_cpu_cpy(gpu_out, 0, 0);
  auto cpu_out          = add_xz_cpu_out_queue(gpu_cpu, 0, 0);
  auto display          = add_xz_processed_display(cpu_out, 0, 0);
  (void)display;
  current_section_name_.clear();
}

void GraphBuilder::build_yz_branch(V debounce_queue) {
  current_section_name_ = "yz_processed_view::";
  auto cut_avg          = add_yz_cut_avg(debounce_queue, 0, 0);
  auto reshape          = add_yz_reshape(cut_avg, 0, 0);
  auto slide_avg        = add_yz_slide_avg(reshape, 0, 0);
  auto crop             = add_yz_crop2frames(slide_avg, 0, 0);
  auto to_u8            = add_yz_to_u8(crop, 0, 0);
  auto rotation         = add_yz_rotation(to_u8, 0, 0);
  auto gpu_out          = add_yz_gpu_out_queue(rotation, 0, 0);
  auto gpu_cpu          = add_yz_gpu_cpu_cpy(gpu_out, 0, 0);
  auto cpu_out          = add_yz_cpu_out_queue(gpu_cpu, 0, 0);
  auto display          = add_yz_processed_display(cpu_out, 0, 0);
  (void)display;
  current_section_name_.clear();
}

template <JsonSerializable S>
GraphBuilder::V GraphBuilder::add_node(const std::string &name, const std::string &kind,
                                       const S &settings, bool debug) {
  auto v = boost::add_vertex(holoflow::core::NodeSpec{.name     = current_section_name_ + name,
                                                      .kind     = kind,
                                                      .settings = nlohmann::json(settings),
                                                      .debug    = debug},
                             spec_);
  return v;
}

template <JsonSerializable S>
GraphBuilder::V GraphBuilder::add_node_after(const V &after, int out_idx, int in_idx,
                                             const std::string &name, const std::string &kind,
                                             const S &settings, bool debug) {
  auto v = add_node(name, kind, settings, debug);
  boost::add_edge(after, v, {out_idx, in_idx}, spec_);
  return v;
}

GraphBuilder::V GraphBuilder::add_source() {
  if (s_.import_source == ImportSource::HOLOFILE) {
    using sources::HolofileSettings;
    using LoadKind = HolofileSettings::LoadKind;
    std::map<LoadMethod, LoadKind> load_method_map{
        {LoadMethod::READ_LIVE, LoadKind::Live},
        {LoadMethod::LOAD_IN_CPU, LoadKind::CPUCached},
        {LoadMethod::LOAD_IN_GPU, LoadKind::GPUCached},
    };

    return add_node<HolofileSettings>("source", "Holofile",
                                      HolofileSettings{
                                          .path        = s_.load_path.string(),
                                          .load_kind   = load_method_map.at(s_.load_method),
                                          .start_frame = s_.load_begin,
                                          .end_frame   = s_.load_end,
                                          .batch_size  = s_.load_batch,
                                      });
  }

  else if (s_.import_source == ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO) {
    using sources::AmetekS710EuresysCoaxlinkOctoSettings;
    return add_node<AmetekS710EuresysCoaxlinkOctoSettings>(
        "source", "AmetekS710EuresysCoaxlinkOcto",
        AmetekS710EuresysCoaxlinkOctoSettings{
            .cfg_path = s_.camera_config_path.string(),
        });
  }

  else if (s_.import_source == ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP) {
    using sources::AmetekS711EuresysCoaxlinkQSFPSettings;
    return add_node<AmetekS711EuresysCoaxlinkQSFPSettings>(
        "source", "AmetekS711EuresysCoaxlinkQSFP+",
        AmetekS711EuresysCoaxlinkQSFPSettings{
            .cfg_path = s_.camera_config_path.string(),
        });
  }

  HOLOVIBES_UNIMPLEMENTED();
}

GraphBuilder::V GraphBuilder::add_cpu_raw_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(
      parent, out_idx, in_idx, "cpu_raw_queue", "BatchQueue",
      BatchQueueSettings{
          .target_capacity = s_.cpu_in_size,
          .output_size     = 1,
          .output_stride   = opti_cpu_stride_ ? s_.time_stride : s_.time_window,
      });
}

GraphBuilder::V GraphBuilder::add_raw_reshape(V parent, int out_idx, int in_idx) {
  using syncs::ReshapeSettings;
  return add_node_after<ReshapeSettings>(
      parent, out_idx, in_idx, "raw_reshape", "Reshape",
      ReshapeSettings{
          .shape = {1, static_cast<size_t>(src_height_), static_cast<size_t>(src_width_)},
      });
}

GraphBuilder::V GraphBuilder::add_cpu_in_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(
      parent, out_idx, in_idx, "cpu_in_queue", "BatchQueue",
      BatchQueueSettings{
          .target_capacity = s_.cpu_in_size,
          .output_size     = s_.time_window,
          .output_stride   = opti_cpu_stride_ ? s_.time_stride : s_.time_window,
      });
}

GraphBuilder::V GraphBuilder::add_record_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "record_queue", "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.recording_count,
                                                .output_size     = s_.time_window,
                                                .output_stride   = s_.time_window,
                                            });
}

GraphBuilder::V GraphBuilder::add_cpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "cpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

GraphBuilder::V GraphBuilder::add_cpu_raw_view_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "cpu_raw_view_cpy", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

GraphBuilder::V GraphBuilder::add_xy_raw_display(V parent, int out_idx, int in_idx) {
  using sinks::DisplayTensorSettings;
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "xy_raw_display",
                                               "DisplayTensorXYRaw", DisplayTensorSettings{});
}

GraphBuilder::V GraphBuilder::add_record(V parent, int out_idx, int in_idx) {
  using sinks::HolofileSettings;
  return add_node_after<HolofileSettings>(parent, out_idx, in_idx, "record", "HolofileWriter",
                                          HolofileSettings{
                                              .path              = s_.recording_path.string(),
                                              .count             = s_.recording_count,
                                              .pipeline_settings = settings_to_old_json(s_),
                                          },
                                          false);
}

GraphBuilder::V GraphBuilder::add_cpu_gpu_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "cpu_gpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Device,
                                        });
}

GraphBuilder::V GraphBuilder::add_gpu_in_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(
      parent, out_idx, in_idx, "gpu_in_queue", "BatchQueue",
      BatchQueueSettings{
          .target_capacity = s_.gpu_in_size,
          .output_size     = s_.time_window,
          .output_stride   = opti_gpu_stride_ ? s_.time_stride : s_.time_window,
      });
}

GraphBuilder::V GraphBuilder::add_to_cf32(V parent, int out_idx, int in_idx) {
  using syncs::ConversionSettings;
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "to_cf32", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::CF32,
                                                .strategy = ConversionSettings::Strategy::Real,
                                            });
}

GraphBuilder::V GraphBuilder::add_spacial_transform(V parent, int out_idx, int in_idx) {
  using syncs::AngularSpectrumSettings;
  using syncs::FresnelDiffractionSettings;
  if (s_.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION) {
    return add_node_after<FresnelDiffractionSettings>(parent, out_idx, in_idx, "spacial_transform",
                                                      "FresnelDiffraction",
                                                      FresnelDiffractionSettings{
                                                          .lambda = s_.spacial_lambda,
                                                          .dx     = s_.spacial_pixel_size,
                                                          .dy     = s_.spacial_pixel_size,
                                                          .z      = s_.spacial_z,
                                                      });
  }

  if (s_.spacial_method == SpacialMethod::ANGULAR_SPECTRUM) {
    std::optional<AngularSpectrumSettings::Filter> filter = std::nullopt;
    // FIXME: Temporarily disable filter optimization
    if (s_.filter_2d && false) {
      filter = AngularSpectrumSettings::Filter{
          .r_inner = s_.filter_r_inner,
          .r_outer = s_.filter_r_outer,
          .s_inner = s_.filter_smooth_inner,
          .s_outer = s_.filter_smooth_outer,
      };
    }

    return add_node_after<AngularSpectrumSettings>(parent, out_idx, in_idx, "spacial_transform",
                                                   "AngularSpectrum",
                                                   AngularSpectrumSettings{
                                                       .lambda = s_.spacial_lambda,
                                                       .dx     = s_.spacial_pixel_size,
                                                       .dy     = s_.spacial_pixel_size,
                                                       .z      = s_.spacial_z,
                                                       .filter = filter,
                                                   });
  }

  HOLOVIBES_CHECK(s_.spacial_method != SpacialMethod::NONE);
  HOLOVIBES_UNIMPLEMENTED();
}

GraphBuilder::V GraphBuilder::add_spacial_filter(V parent, int out_idx, int in_idx) {
  using syncs::Filter2DSettings;
  return add_node_after<Filter2DSettings>(parent, out_idx, in_idx, "spacial_filter", "Filter2D",
                                          Filter2DSettings{
                                              .r_inner = s_.filter_r_inner,
                                              .r_outer = s_.filter_r_outer,
                                              .s_inner = s_.filter_smooth_inner,
                                              .s_outer = s_.filter_smooth_outer,
                                          });
}

GraphBuilder::V GraphBuilder::add_time_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  auto time_stride = (opti_cpu_stride_ || opti_gpu_stride_) ? s_.time_window : s_.time_stride;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "time_queue", "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.time_stride * 2,
                                                .output_size     = s_.time_window,
                                                .output_stride   = time_stride,
                                            });
}

GraphBuilder::V GraphBuilder::add_time_transform(V parent, int out_idx, int in_idx) {
  using syncs::PcaSettings;
  using syncs::StftSettings;
  if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS && !s_.view_3d_cuts) {
    return add_node_after<PcaSettings>(parent, out_idx, in_idx, "time_transform", "Pca",
                                       PcaSettings{
                                           .begin = s_.time_z_begin,
                                           .end   = s_.time_z_end,
                                       });
  }

  if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS && s_.view_3d_cuts) {
    return add_node_after<PcaSettings>(parent, out_idx, in_idx, "time_transform", "Pca",
                                       PcaSettings{
                                           .begin = 0,
                                           .end   = s_.time_window,
                                       });
  }

  if (s_.time_method == TimeMethod::SHORT_TIME_FOURIER) {
    return add_node_after<StftSettings>(parent, out_idx, in_idx, "time_transform", "Stft",
                                        StftSettings{});
  }

  HOLOVIBES_CHECK(s_.time_method != TimeMethod::NONE);
  HOLOVIBES_UNIMPLEMENTED();
}

GraphBuilder::V GraphBuilder::add_to_f32(V parent, int out_idx, int in_idx) {
  using syncs::ConversionSettings;
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "to_f32", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::F32,
                                                .strategy = ConversionSettings::Strategy::Modulus,
                                            });
}

GraphBuilder::V GraphBuilder::add_debounce_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS && !s_.view_3d_cuts) {
    return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "debounce_queue",
                                              "BatchQueue",
                                              BatchQueueSettings{
                                                  .target_capacity = 128,
                                                  .output_size   = s_.time_z_end - s_.time_z_begin,
                                                  .output_stride = s_.time_z_end - s_.time_z_begin,
                                              });
  }

  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "debounce_queue", "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = 128,
                                                .output_size     = s_.time_window,
                                                .output_stride   = s_.time_window,
                                            });
}

GraphBuilder::V GraphBuilder::add_xy_cut_avg(V parent, int out_idx, int in_idx) {
  using syncs::AverageSettings;
  if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS && !s_.view_3d_cuts) {
    return add_node_after<AverageSettings>(parent, out_idx, in_idx, "xy_cut_avg", "Average",
                                           AverageSettings{
                                               .axis  = 0,
                                               .start = 0,
                                               .end   = s_.time_z_end - s_.time_z_begin,
                                           });
  }

  return add_node_after<AverageSettings>(parent, out_idx, in_idx, "xy_cut_avg", "Average",
                                         AverageSettings{
                                             .axis  = 0,
                                             .start = s_.time_z_begin,
                                             .end   = s_.time_z_end,
                                         });
}

GraphBuilder::V GraphBuilder::add_fft_shift(V parent, int out_idx, int in_idx) {
  using syncs::FFTShiftSettings;
  return add_node_after<FFTShiftSettings>(parent, out_idx, in_idx, "fft_shift", "FFTShift",
                                          FFTShiftSettings{});
}

GraphBuilder::V GraphBuilder::add_xy_registration(V parent, int out_idx, int in_idx) {
  using syncs::RegistrationSettings;

  return add_node_after<RegistrationSettings>(parent, out_idx, in_idx, "xy_registration",
                                              "Registration",
                                              RegistrationSettings{
                                                  .radius = s_.pp_registration_radius,
                                              });
}

GraphBuilder::V GraphBuilder::add_xy_slide_avg(V parent, int out_idx, int in_idx) {
  using asyncs::SlidingAverageSettings;
  return add_node_after<SlidingAverageSettings>(
      parent, out_idx, in_idx, "xy_slide_avg", "SlidingAverage",
      SlidingAverageSettings{.target_capacity = 128,
                             .window_size     = static_cast<size_t>(s_.pp_accumulation)});
}

GraphBuilder::V GraphBuilder::add_xy_fps_limiter(V parent, int out_idx, int in_idx) {
  HOLOVIBES_UNIMPLEMENTED();
  (void)parent;
  (void)out_idx;
  (void)in_idx;
}

GraphBuilder::V GraphBuilder::add_xy_convolution(V parent, int out_idx, int in_idx) {
  using syncs::ConvolutionSettings;
  return add_node_after<ConvolutionSettings>(parent, out_idx, in_idx, "xy_convolution",
                                             "Convolution",
                                             ConvolutionSettings{
                                                 .kernel_file = s_.pp_convolution_path,
                                                 .divide      = s_.pp_convolution_divide,
                                             });
}

GraphBuilder::V GraphBuilder::add_xy_pctclip(V parent, int out_idx, int in_idx) {
  using syncs::PctClipSettings;
  // TODO: THis formula is likely wrong
  auto r  = std::clamp(s_.pp_pctclip_radius, 0.0f, 1.0f);
  auto rx = r * 0.5f;
  auto ry = r * 0.5f;

  PctClipSettings::Ellipse roi{
      .cx    = 0.5f,
      .cy    = 0.5f,
      .rx    = rx,
      .ry    = ry,
      .angle = 0.0f,
  };

  return add_node_after<PctClipSettings>(parent, out_idx, in_idx, "xy_pctclip", "PctClip",
                                         PctClipSettings{
                                             .min_pct = s_.pp_pctclip_lower,
                                             .max_pct = s_.pp_pctclip_upper,
                                             .roi     = roi,
                                         });
}

GraphBuilder::V GraphBuilder::add_xy_to_u8(V parent, int out_idx, int in_idx) {
  using syncs::ConversionSettings;
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "xy_to_u8", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::U8,
                                                .strategy = ConversionSettings::Strategy::Scaled,
                                            });
}

GraphBuilder::V GraphBuilder::add_xy_gpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xy_gpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.gpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

GraphBuilder::V GraphBuilder::add_xy_gpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "xy_gpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

GraphBuilder::V GraphBuilder::add_xy_cpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xy_cpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.cpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

GraphBuilder::V GraphBuilder::add_xy_processed_display(V parent, int out_idx, int in_idx) {
  using sinks::DisplayTensorSettings;
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "xy_processed_display",
                                               "DisplayTensorXY", DisplayTensorSettings{});
}

GraphBuilder::V GraphBuilder::add_xz_cut_avg(V parent, int out_idx, int in_idx) {
  using syncs::AverageSettings;
  return add_node_after<AverageSettings>(parent, out_idx, in_idx, "xz_cut_avg", "Average",
                                         AverageSettings{
                                             .axis  = 1,
                                             .start = s_.time_y_begin,
                                             .end   = s_.time_y_end,
                                         });
}

GraphBuilder::V GraphBuilder::add_xz_reshape(V parent, int out_idx, int in_idx) {
  using syncs::ReshapeSettings;
  return add_node_after<ReshapeSettings>(
      parent, out_idx, in_idx, "xz_reshape", "Reshape",
      ReshapeSettings{
          .shape = {1, static_cast<size_t>(src_height_), static_cast<size_t>(s_.time_window)},
      });
}

GraphBuilder::V GraphBuilder::add_xz_slide_avg(V parent, int out_idx, int in_idx) {
  using asyncs::SlidingAverageSettings;
  return add_node_after<SlidingAverageSettings>(
      parent, out_idx, in_idx, "xz_slide_avg", "SlidingAverage",
      SlidingAverageSettings{.target_capacity = 128,
                             .window_size     = static_cast<size_t>(s_.pp_accumulation)});
}

GraphBuilder::V GraphBuilder::add_xz_to_u8(V parent, int out_idx, int in_idx) {
  using syncs::ConversionSettings;
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "xz_to_u8", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::U8,
                                                .strategy = ConversionSettings::Strategy::Scaled,
                                            });
}

GraphBuilder::V GraphBuilder::add_xz_crop2frames(V parent, int out_idx, int in_idx) {
  using syncs::CropSettings;
  return add_node_after<CropSettings>(
      parent, out_idx, in_idx, "xz_crop2frames", "Crop",
      CropSettings{
          .origin = {0, 10, 0},
          .shape  = {static_cast<size_t>(1), static_cast<size_t>(src_height_ - 20),
                     static_cast<size_t>(s_.time_window)},
      });
}

GraphBuilder::V GraphBuilder::add_xz_gpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xz_gpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.gpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

GraphBuilder::V GraphBuilder::add_xz_gpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "xz_gpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

GraphBuilder::V GraphBuilder::add_xz_cpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xz_cpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.cpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

GraphBuilder::V GraphBuilder::add_xz_processed_display(V parent, int out_idx, int in_idx) {
  using sinks::DisplayTensorSettings;
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "xz_processed_display",
                                               "DisplayTensorXZ", DisplayTensorSettings{});
}

GraphBuilder::V GraphBuilder::add_yz_cut_avg(V parent, int out_idx, int in_idx) {
  using syncs::AverageSettings;
  return add_node_after<AverageSettings>(parent, out_idx, in_idx, "yz_cut_avg", "Average",
                                         AverageSettings{
                                             .axis  = 2,
                                             .start = s_.time_x_begin,
                                             .end   = s_.time_x_end,
                                         });
}

GraphBuilder::V GraphBuilder::add_yz_reshape(V parent, int out_idx, int in_idx) {
  using syncs::ReshapeSettings;
  return add_node_after<ReshapeSettings>(
      parent, out_idx, in_idx, "yz_reshape", "Reshape",
      ReshapeSettings{
          .shape = {1, static_cast<size_t>(src_height_), static_cast<size_t>(s_.time_window)},
      });
}

GraphBuilder::V GraphBuilder::add_yz_slide_avg(V parent, int out_idx, int in_idx) {
  using asyncs::SlidingAverageSettings;
  return add_node_after<SlidingAverageSettings>(
      parent, out_idx, in_idx, "yz_slide_avg", "SlidingAverage",
      SlidingAverageSettings{.target_capacity = 128,
                             .window_size     = static_cast<size_t>(s_.pp_accumulation)});
}

GraphBuilder::V GraphBuilder::add_yz_to_u8(V parent, int out_idx, int in_idx) {
  using syncs::ConversionSettings;
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "yz_to_u8", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::U8,
                                                .strategy = ConversionSettings::Strategy::Scaled,
                                            });
}

GraphBuilder::V GraphBuilder::add_yz_rotation(V parent, int out_idx, int in_idx) {
  using syncs::RotationSettings;
  return add_node_after<RotationSettings>(parent, out_idx, in_idx, "yz_rotation", "Rotation",
                                          RotationSettings{
                                              .angle = 90,
                                          });
}

GraphBuilder::V GraphBuilder::add_yz_crop2frames(V parent, int out_idx, int in_idx) {
  using syncs::CropSettings;
  return add_node_after<CropSettings>(
      parent, out_idx, in_idx, "yz_crop", "Crop",
      CropSettings{
          .origin = {0, 10, 0},
          .shape  = {static_cast<size_t>(1), static_cast<size_t>(src_height_ - 20),
                     static_cast<size_t>(s_.time_window)},
      });
}

GraphBuilder::V GraphBuilder::add_yz_gpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "yz_gpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.gpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

GraphBuilder::V GraphBuilder::add_yz_gpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "yz_gpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

GraphBuilder::V GraphBuilder::add_yz_cpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "yz_cpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.cpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

GraphBuilder::V GraphBuilder::add_yz_processed_display(V parent, int out_idx, int in_idx) {
  using sinks::DisplayTensorSettings;
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "yz_processed_display",
                                               "DisplayTensorYZ", DisplayTensorSettings{});
}

} // namespace holovibes::pipeline