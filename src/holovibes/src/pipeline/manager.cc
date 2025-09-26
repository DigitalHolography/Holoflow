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

#include "pipeline/manager.hh"

#include <fstream>

#include "bug.hh"
#include "holoflow/runtime/graph_display.hh"
#include "logger.hh"
#include "tasks/angular_spectrum.hh"
#include "tasks/average.hh"
#include "tasks/batch_queue.hh"
#include "tasks/conversion.hh"
#include "tasks/display_tensor.hh"
#include "tasks/fft_shift.hh"
#include "tasks/fresnel_diffraction.hh"
#include "tasks/holofile.hh"
#include "tasks/memcpy.hh"
#include "tasks/pca.hh"
#include "tasks/pct_clip.hh"
#include "tasks/slide_avg.hh"
#include "tasks/stft.hh"

using namespace holovibes::tasks;

namespace holovibes::pipeline {

namespace {

template <class F, class... Args>
void reg_sync(holoflow::core::Registry &r, std::string_view name, Args &&...args) {
  r.register_sync(std::string{name}, std::make_unique<F>(std::forward<Args>(args)...));
}

template <class F, class... Args>
void reg_async(holoflow::core::Registry &r, std::string_view name, Args &&...args) {
  r.register_async(std::string{name}, std::make_unique<F>(std::forward<Args>(args)...));
}

} // namespace

Manager::Manager(ui::TensorDisplayWidget *xy_processed_widget,
                 ui::TensorDisplayWidget *xz_processed_widget,
                 ui::TensorDisplayWidget *yz_processed_widget,
                 ui::TensorDisplayWidget *xy_raw_widget)
    : xy_processed_widget_(xy_processed_widget), xz_processed_widget_(xz_processed_widget),
      yz_processed_widget_(yz_processed_widget), xy_raw_widget_(xy_raw_widget) {
  reg_sync<AngularSpectrumFactory>(registry_, "AngularSpectrum");
  reg_sync<AverageFactory>(registry_, "Average");
  reg_async<BatchQueueFactory>(registry_, "BatchQueue");
  reg_sync<ConversionFactory>(registry_, "Conversion");
  reg_sync<DisplayTensorFactory>(registry_, "DisplayTensorXY", xy_processed_widget_);
  reg_sync<DisplayTensorFactory>(registry_, "DisplayTensorXZ", xz_processed_widget_);
  reg_sync<DisplayTensorFactory>(registry_, "DisplayTensorYZ", yz_processed_widget_);
  reg_sync<DisplayTensorFactory>(registry_, "DisplayTensorXYRaw", xy_raw_widget_);
  reg_sync<FFTShiftFactory>(registry_, "FFTShift");
  reg_sync<FresnelDiffractionFactory>(registry_, "FresnelDiffraction");
  reg_sync<HolofileFactory>(registry_, "Holofile");
  reg_sync<MemcpyFactory>(registry_, "Memcpy");
  reg_sync<PcaFactory>(registry_, "Pca");
  reg_sync<PctClipFactory>(registry_, "PctClip");
  reg_sync<StftFactory>(registry_, "Stft");
  reg_async<SlidingAverageFactory>(registry_, "SlidingAverage");
}

void Manager::start_pipeline() {
  logger()->info("[Manager::start_pipeline] Starting pipeline...");
  std::lock_guard lock(mtx_);

  if (scheduler_ && scheduler_->is_running()) {
    logger()->error("[Manager::start_pipeline] Pipeline is already running");
    emit start_pipeline_failure();
    return;
  }

  try {
    build_and_run();
    logger()->info("[Manager::start_pipeline] Pipeline started successfully");
    emit start_pipeline_success();
  } catch (const std::exception &e) {
    logger()->error("[Manager::start_pipeline] Failed to start pipeline: {}", e.what());
    emit start_pipeline_failure();
  }
}

void Manager::stop_pipeline() {
  logger()->info("[Manager::stop_pipeline] Stopping pipeline...");
  std::lock_guard lock(mtx_);

  if (!scheduler_ || !scheduler_->is_running()) {
    logger()->error("[Manager::stop_pipeline] Pipeline is not running");
    emit stop_pipeline_failure();
    return;
  }

  try {
    scheduler_->request_stop();
    scheduler_->wait();
    logger()->info("[Manager::stop_pipeline] Pipeline stopped successfully");
    emit stop_pipeline_success();
  } catch (const std::exception &e) {
    logger()->error("[Manager::stop_pipeline] Failed to stop pipeline: {}", e.what());
    emit stop_pipeline_failure();
  }
}

void Manager::update_pipeline(const Settings &settings) {
  logger()->info("[Manager::update_pipeline] Updating pipeline settings...");
  std::lock_guard lock(mtx_);
  s_              = settings;
  settings_dirty_ = true;

  if (!scheduler_ || !scheduler_->is_running()) {
    logger()->debug("[Manager::update_pipeline] Pipeline is not running, no need to restart");
    logger()->info("[Manager::update_pipeline] Pipeline settings updated successfully");
    emit update_pipeline_success();
    return;
  }

  try {
    scheduler_->request_stop();
    scheduler_->wait();
    build_and_run();
    logger()->info("[Manager::update_pipeline] Pipeline updated successfully");
    emit update_pipeline_success();
  } catch (const std::exception &e) {
    logger()->error("[Manager::update_pipeline] Failed to update pipeline: {}", e.what());
    emit update_pipeline_failure();
  }
}

void Manager::build_and_run() {
  build_graph_spec();

  // TODO: Proper log path in app data folder
  using namespace std::chrono;
  constexpr const char *LOG_FOLDER_PATH = "logs";

  auto dot      = holoflow::core::to_dot(spec_);
  auto t        = floor<seconds>(system_clock::now());
  auto date     = std::format("{:%Y-%m-%d_%H-%M-%S}", t);
  auto log_path = std::format("pipeline_{}.dot", LOG_FOLDER_PATH, date);
  std::ofstream(log_path) << dot;
  logger()->info("[Manager::build_and_run] Pipeline graph saved to {}", log_path);

  auto old         = std::move(compiler_output_);
  compiler_output_ = holoflow::runtime::Compiler(registry_).compile(spec_);

  auto &graph     = compiler_output_->graph;
  auto &sections  = compiler_output_->sections;
  auto &resources = compiler_output_->resources;

  auto dot_compile = holoflow::runtime::to_dot(*compiler_output_, registry_);
  t                = floor<seconds>(system_clock::now());
  date             = std::format("{:%Y-%m-%d_%H-%M-%S}", t);
  log_path         = std::format("pipeline_compiled_{}.dot", LOG_FOLDER_PATH, date);
  std::ofstream(log_path) << dot_compile;
  logger()->info("[Manager::build_and_run] Compiled pipeline graph saved to {}", log_path);

  scheduler_ = std::make_unique<holoflow::runtime::Scheduler>(graph, sections, resources);
  scheduler_->start();
}

void Manager::build_graph_spec() {
  logger()->info("[Manager::build_graph_spec] Building graph spec...");
  HOLOVIBES_CHECK(settings_dirty_, "Settings are not dirty, no need to rebuild graph spec");

  reset_graph_spec();
  guess_optimizations();
  guess_source_dims();

  auto source = add_source();
  auto parent = source;

  if (s_.load_method != LoadMethod::LOAD_IN_GPU) {
    auto cpu_in_queue = add_cpu_in_queue(parent, 0, 0);
    auto cpu_gpu_cpy  = add_cpu_gpu_cpy(cpu_in_queue, 0, 0);
    parent            = cpu_gpu_cpy;
  }

  auto gpu_in_queue = add_gpu_in_queue(parent, 0, 0);
  auto to_cf32      = add_to_cf32(gpu_in_queue, 0, 0);
  parent            = to_cf32;

  if (s_.spacial_method != SpacialMethod::NONE) {
    auto spacial_transform = add_spacial_transform(parent, 0, 0);
    parent                 = spacial_transform;
  }

  if (s_.filter_2d) {
    auto spacial_filter = add_spacial_filter(parent, 0, 0);
    parent              = spacial_filter;
  }

  if (s_.time_method != TimeMethod::NONE) {
    auto time_queue     = add_time_queue(parent, 0, 0);
    auto time_transform = add_time_transform(time_queue, 0, 0);
    parent              = time_transform;
  }

  auto to_f32         = add_to_f32(parent, 0, 0);
  auto debounce_queue = add_debounce_queue(to_f32, 0, 0);

  // XY processed branch
  {
    auto cut_avg = add_xy_cut_avg(debounce_queue, 0, 0);
    parent       = cut_avg;

    if (s_.pp_fft_shift) {
      auto fft_shift = add_fft_shift(parent, 0, 0);
      parent         = fft_shift;
    }

    // auto identity_0 = add_xy_identity_0(parent, 0, 0);
    // parent          = identity_0;

    if (s_.pp_registration) {
      auto registration = add_xy_registration(parent, 0, 0);
      parent            = registration;
    }

    auto slide_avg = add_xy_slide_avg(parent, 0, 0);
    parent         = slide_avg;
    // auto identity_1  = add_xy_identity_1(slide_avg, 0, 0);
    // auto fps_limiter = add_xy_fps_limiter(identity_1, 0, 0);
    // parent           = fps_limiter;
    // parent = identity_1;

    // if (s_.pp_convolution) {
    //   auto convolution = add_xy_convolution(parent, 0, 0);
    //   parent           = convolution;
    // }

    // auto pctclip       = add_xy_pctclip(parent, 0, 0);
    auto to_u8         = add_xy_to_u8(parent, 0, 0);
    auto gpu_out_queue = add_xy_gpu_out_queue(to_u8, 0, 0);
    auto gpu_cpu       = add_xy_gpu_cpu_cpy(gpu_out_queue, 0, 0);
    auto cpu_out_queue = add_xy_cpu_out_queue(gpu_cpu, 0, 0);
    auto display       = add_xy_processed_display(cpu_out_queue, 0, 0);
    (void)display;
  }

  // XZ processed branch
  if (s_.view_3d_cuts) {
    auto cut_avg   = add_xz_cut_avg(debounce_queue, 0, 0);
    auto reshape   = add_xz_reshape(cut_avg, 0, 0);
    auto slide_avg = add_xz_slide_avg(reshape, 0, 0);
    auto to_u8     = add_xz_to_u8(slide_avg, 0, 0);
    auto gpu_out   = add_xz_gpu_out_queue(to_u8, 0, 0);
    auto gpu_cpu   = add_xz_gpu_cpu_cpy(gpu_out, 0, 0);
    auto cpu_out   = add_xz_cpu_out_queue(gpu_cpu, 0, 0);
    auto display   = add_xz_processed_display(cpu_out, 0, 0);
    (void)display;
  }

  // YZ processed branch
  if (s_.view_3d_cuts) {
    auto cut_avg   = add_yz_cut_avg(debounce_queue, 0, 0);
    auto reshape   = add_yz_reshape(cut_avg, 0, 0);
    auto slide_avg = add_yz_slide_avg(reshape, 0, 0);
    auto to_u8     = add_yz_to_u8(slide_avg, 0, 0);
    auto gpu_out   = add_yz_gpu_out_queue(to_u8, 0, 0);
    auto gpu_cpu   = add_yz_gpu_cpu_cpy(gpu_out, 0, 0);
    auto cpu_out   = add_yz_cpu_out_queue(gpu_cpu, 0, 0);
    auto display   = add_yz_processed_display(cpu_out, 0, 0);
    (void)display;
  }

  settings_dirty_ = false;
  logger()->debug("[Manager::build_graph_spec] Graph spec built successfully");
}

void Manager::reset_graph_spec() { spec_ = holoflow::core::GraphSpec{}; }

void Manager::guess_optimizations() {
  bool load_in_gpu     = (s_.load_method == LoadMethod::LOAD_IN_GPU);
  bool stride_multiple = (s_.time_stride % s_.time_window == 0);
  opti_cpu_stride_     = stride_multiple && !load_in_gpu;
  opti_gpu_stride_     = stride_multiple && load_in_gpu;
}

void Manager::guess_source_dims() {
  if (s_.import_source == ImportSource::HOLOFILE) {
    auto header = holofile::Reader(s_.load_path.string()).header();
    src_width_  = header.frame_width;
    src_height_ = header.frame_height;
    return;
  }

  HOLOVIBES_UNIMPLEMENTED();
}

template <JsonSerializable S>
Manager::V Manager::add_node(const std::string &name, const std::string &kind, const S &settings) {
  auto v = boost::add_vertex(
      holoflow::core::NodeSpec{
          .name     = name,
          .kind     = kind,
          .settings = nlohmann::json(settings),
      },
      spec_);
  return v;
}

template <JsonSerializable S>
Manager::V Manager::add_node_after(const V &after, int out_idx, int in_idx, const std::string &name,
                                   const std::string &kind, const S &settings) {
  auto v = add_node(name, kind, settings);
  boost::add_edge(after, v, {out_idx, in_idx}, spec_);
  return v;
}

Manager::V Manager::add_source() {
  if (s_.import_source == ImportSource::HOLOFILE) {
    std::map<LoadMethod, HolofileSettings::LoadKind> load_method_map{
        {LoadMethod::READ_LIVE, HolofileSettings::LoadKind::Live},
        {LoadMethod::LOAD_IN_CPU, HolofileSettings::LoadKind::CPUCached},
        {LoadMethod::LOAD_IN_GPU, HolofileSettings::LoadKind::GPUCached},
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

  HOLOVIBES_UNIMPLEMENTED();
}

Manager::V Manager::add_cpu_in_queue(V parent, int out_idx, int in_idx) {
  return add_node_after<BatchQueueSettings>(
      parent, out_idx, in_idx, "cpu_in_queue", "BatchQueue",
      BatchQueueSettings{
          .target_capacity = s_.cpu_in_size,
          .output_size     = s_.time_window,
          .output_stride   = opti_cpu_stride_ ? s_.time_stride : s_.time_window,
      });
}

Manager::V Manager::add_cpu_gpu_cpy(V parent, int out_idx, int in_idx) {
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "cpu_gpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Device,
                                        });
}

Manager::V Manager::add_gpu_in_queue(V parent, int out_idx, int in_idx) {
  return add_node_after<BatchQueueSettings>(
      parent, out_idx, in_idx, "gpu_in_queue", "BatchQueue",
      BatchQueueSettings{
          .target_capacity = s_.gpu_in_size,
          .output_size     = s_.time_window,
          .output_stride   = opti_gpu_stride_ ? s_.time_stride : s_.time_window,
      });
}

Manager::V Manager::add_to_cf32(V parent, int out_idx, int in_idx) {
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "to_cf32", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::CF32,
                                                .strategy = ConversionSettings::Strategy::Real,
                                            });
}

Manager::V Manager::add_spacial_transform(V parent, int out_idx, int in_idx) {
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
    if (s_.filter_2d) {
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

Manager::V Manager::add_spacial_filter(V parent, int out_idx, int in_idx) {
  HOLOVIBES_UNIMPLEMENTED();
  (void)parent;
  (void)out_idx;
  (void)in_idx;
}

Manager::V Manager::add_time_queue(V parent, int out_idx, int in_idx) {
  auto time_stride = (opti_cpu_stride_ || opti_gpu_stride_) ? s_.time_window : s_.time_stride;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "time_queue", "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.time_stride * 2,
                                                .output_size     = s_.time_window,
                                                .output_stride   = time_stride,
                                            });
}

Manager::V Manager::add_time_transform(V parent, int out_idx, int in_idx) {
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

Manager::V Manager::add_to_f32(V parent, int out_idx, int in_idx) {
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "to_f32", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::F32,
                                                .strategy = ConversionSettings::Strategy::Modulus,
                                            });
}

Manager::V Manager::add_debounce_queue(V parent, int out_idx, int in_idx) {
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

Manager::V Manager::add_xy_cut_avg(V parent, int out_idx, int in_idx) {
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

Manager::V Manager::add_fft_shift(V parent, int out_idx, int in_idx) {
  return add_node_after<FFTShiftSettings>(parent, out_idx, in_idx, "fft_shift", "FFTShift",
                                          FFTShiftSettings{});
}

Manager::V Manager::add_xy_identity_0(V parent, int out_idx, int in_idx) {
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "xy_identity_0", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Device,
                                        });
}

Manager::V Manager::add_xy_registration(V parent, int out_idx, int in_idx) {
  HOLOVIBES_UNIMPLEMENTED();
  (void)parent;
  (void)out_idx;
  (void)in_idx;
}

Manager::V Manager::add_xy_slide_avg(V parent, int out_idx, int in_idx) {
  return add_node_after<SlidingAverageSettings>(
      parent, out_idx, in_idx, "xy_slide_avg", "SlidingAverage",
      SlidingAverageSettings{.target_capacity = 128,
                             .window_size     = static_cast<size_t>(s_.pp_accumulation)});
}

Manager::V Manager::add_xy_identity_1(V parent, int out_idx, int in_idx) {
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "xy_identity_1", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Device,
                                        });
}

Manager::V Manager::add_xy_fps_limiter(V parent, int out_idx, int in_idx) {
  HOLOVIBES_UNIMPLEMENTED();
  (void)parent;
  (void)out_idx;
  (void)in_idx;
}

Manager::V Manager::add_xy_convolution(V parent, int out_idx, int in_idx) {
  HOLOVIBES_UNIMPLEMENTED();
  (void)parent;
  (void)out_idx;
  (void)in_idx;
}

Manager::V Manager::add_xy_pctclip(V parent, int out_idx, int in_idx) {
  // TODO: THis formula is likely wrong
  auto r  = std::clamp(s_.pp_pctclip_radius, 0.0f, 1.0f);
  auto s  = static_cast<float>(std::min(src_width_, src_height_));
  auto rx = r * (s / static_cast<float>(src_width_));
  auto ry = r * (s / static_cast<float>(src_height_));

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

Manager::V Manager::add_xy_to_u8(V parent, int out_idx, int in_idx) {
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "xy_to_u8", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::U8,
                                                .strategy = ConversionSettings::Strategy::Scaled,
                                            });
}

Manager::V Manager::add_xy_gpu_out_queue(V parent, int out_idx, int in_idx) {
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xy_gpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.gpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_xy_gpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "xy_gpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

Manager::V Manager::add_xy_cpu_out_queue(V parent, int out_idx, int in_idx) {
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xy_cpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.cpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_xy_processed_display(V parent, int out_idx, int in_idx) {
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "xy_processed_display",
                                               "DisplayTensorXY", DisplayTensorSettings{});
}

Manager::V Manager::add_xz_cut_avg(V parent, int out_idx, int in_idx) {
  return add_node_after<AverageSettings>(parent, out_idx, in_idx, "xz_cut_avg", "Average",
                                         AverageSettings{
                                             .axis  = 1,
                                             .start = s_.time_y_begin,
                                             .end   = s_.time_y_end,
                                         });
}

Manager::V Manager::add_xz_reshape(V parent, int out_idx, int in_idx) {
  HOLOVIBES_UNIMPLEMENTED();
  (void)parent;
  (void)out_idx;
  (void)in_idx;
}

Manager::V Manager::add_xz_slide_avg(V parent, int out_idx, int in_idx) {
  HOLOVIBES_UNIMPLEMENTED();
  (void)parent;
  (void)out_idx;
  (void)in_idx;
}

Manager::V Manager::add_xz_to_u8(V parent, int out_idx, int in_idx) {
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "xz_to_u8", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::U8,
                                                .strategy = ConversionSettings::Strategy::Scaled,
                                            });
}

Manager::V Manager::add_xz_gpu_out_queue(V parent, int out_idx, int in_idx) {
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xz_gpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.gpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_xz_gpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "xz_gpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

Manager::V Manager::add_xz_cpu_out_queue(V parent, int out_idx, int in_idx) {
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xz_cpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.cpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_xz_processed_display(V parent, int out_idx, int in_idx) {
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "xz_processed_display",
                                               "DisplayTensorXZ", DisplayTensorSettings{});
}

Manager::V Manager::add_yz_cut_avg(V parent, int out_idx, int in_idx) {
  return add_node_after<AverageSettings>(parent, out_idx, in_idx, "yz_cut_avg", "Average",
                                         AverageSettings{
                                             .axis  = 2,
                                             .start = s_.time_x_begin,
                                             .end   = s_.time_x_end,
                                         });
}

Manager::V Manager::add_yz_reshape(V parent, int out_idx, int in_idx) {
  HOLOVIBES_UNIMPLEMENTED();
  (void)parent;
  (void)out_idx;
  (void)in_idx;
}

Manager::V Manager::add_yz_slide_avg(V parent, int out_idx, int in_idx) {
  HOLOVIBES_UNIMPLEMENTED();
  (void)parent;
  (void)out_idx;
  (void)in_idx;
}

Manager::V Manager::add_yz_to_u8(V parent, int out_idx, int in_idx) {
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "yz_to_u8", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::U8,
                                                .strategy = ConversionSettings::Strategy::Scaled,
                                            });
}

Manager::V Manager::add_yz_gpu_out_queue(V parent, int out_idx, int in_idx) {
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "yz_gpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.gpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_yz_gpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "yz_gpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

Manager::V Manager::add_yz_cpu_out_queue(V parent, int out_idx, int in_idx) {
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "yz_cpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.cpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_yz_processed_display(V parent, int out_idx, int in_idx) {
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "yz_processed_display",
                                               "DisplayTensorYZ", DisplayTensorSettings{});
}

} // namespace holovibes::pipeline