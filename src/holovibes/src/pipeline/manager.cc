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

#include <QTimer>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <spdlog/fmt/ranges.h>
#include <string>
#include <string_view>
#include <system_error>

#include "bug.hh"
#include "holoflow/runtime/graph_display.hh"
#include "logger.hh"
#include "settings_loader.hh"
#include "tasks/asyncs/batch_queue.hh"
#include "tasks/asyncs/slide_avg.hh"
#include "tasks/sinks/display_tensor.hh"
#include "tasks/sinks/holofile.hh"
#include "tasks/sources/ametek_s710_euresys_coaxlink_octo.hh"
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
  reg_async<asyncs::BatchQueueFactory>(registry_, "BatchQueue");
  reg_async<asyncs::SlidingAverageFactory>(registry_, "SlidingAverage");
  reg_sync<sinks::DisplayTensorFactory>(registry_, "DisplayTensorXY", xy_processed_widget_);
  reg_sync<sinks::DisplayTensorFactory>(registry_, "DisplayTensorXZ", xz_processed_widget_);
  reg_sync<sinks::DisplayTensorFactory>(registry_, "DisplayTensorYZ", yz_processed_widget_);
  reg_sync<sinks::DisplayTensorFactory>(registry_, "DisplayTensorXYRaw", xy_raw_widget_);
  reg_sync<sinks::HolofileFactory>(registry_, "HolofileWriter");
  reg_sync<sources::HolofileFactory>(registry_, "Holofile");
  reg_sync<sources::AmetekS710EuresysCoaxlinkOctoFactory>(registry_,
                                                          "AmetekS710EuresysCoaxlinkOcto");
  reg_sync<syncs::AngularSpectrumFactory>(registry_, "AngularSpectrum");
  reg_sync<syncs::AverageFactory>(registry_, "Average");
  reg_sync<syncs::ConversionFactory>(registry_, "Conversion");
  reg_sync<syncs::FFTShiftFactory>(registry_, "FFTShift");
  reg_sync<syncs::FresnelDiffractionFactory>(registry_, "FresnelDiffraction");
  reg_sync<syncs::MemcpyFactory>(registry_, "Memcpy");
  reg_sync<syncs::PcaFactory>(registry_, "Pca");
  reg_sync<syncs::PctClipFactory>(registry_, "PctClip");
  reg_sync<syncs::StftFactory>(registry_, "Stft");
  reg_sync<syncs::ConvolutionFactory>(registry_, "Convolution");
  reg_sync<syncs::Filter2DFactory>(registry_, "Filter2D");
  reg_sync<syncs::RegistrationFactory>(registry_, "Registration");
  reg_sync<syncs::ReshapeFactory>(registry_, "Reshape");
  reg_sync<syncs::CropFactory>(registry_, "Crop");
  reg_sync<syncs::RotationFactory>(registry_, "Rotation");

  metrics_timer_ = new QTimer(this);
  metrics_timer_->setInterval(1000);
  metrics_timer_->setTimerType(Qt::TimerType::CoarseTimer);
  connect(metrics_timer_, &QTimer::timeout, this, [this] { poll_metrics(); });

  events_timer_ = new QTimer(this);
  events_timer_->setInterval(5);
  events_timer_->setTimerType(Qt::TimerType::CoarseTimer);
  connect(events_timer_, &QTimer::timeout, this, [this] { poll_events(); });
}

void Manager::start_pipeline() {
  logger()->info("[Manager::start_pipeline] Starting pipeline...");
  std::lock_guard lock(mtx_);

  if (scheduler_ && scheduler_->is_running()) {
    const QString msg = QString("Pipeline is already running");
    logger()->error("[Manager::start_pipeline] Pipeline is already running");
    emit start_pipeline_failure(msg);
    return;
  }

  try {
    build_and_run();
    logger()->info("[Manager::start_pipeline] Pipeline started successfully");
    emit start_pipeline_success();
  } catch (const std::exception &e) {
    const QString msg = QString("Failed to start pipeline: %1").arg(e.what());
    logger()->error("[Manager::start_pipeline] Failed to start pipeline: {}", e.what());
    emit start_pipeline_failure(msg);
  }
}

void Manager::stop_pipeline() {
  logger()->info("[Manager::stop_pipeline] Stopping pipeline...");
  std::lock_guard lock(mtx_);

  if (!scheduler_ || !scheduler_->is_running()) {
    const QString msg = QString("Pipeline is not running");
    logger()->error("[Manager::stop_pipeline] Pipeline is not running");
    emit stop_pipeline_failure(msg);
    return;
  }

  try {
    scheduler_->request_stop();
    scheduler_->wait();
    stop_metrics_updates();
    stop_event_polling();
    raw_recording_active_ = false;
    logger()->info("[Manager::stop_pipeline] Pipeline stopped successfully");
    emit stop_pipeline_success();
  } catch (const std::exception &e) {
    const QString msg = QString("Failed to stop pipeline: %1").arg(e.what());
    logger()->error("[Manager::stop_pipeline] Failed to stop pipeline: {}", e.what());
    emit stop_pipeline_failure(msg);
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
    raw_recording_active_ = false;
    build_and_run();
    logger()->info("[Manager::update_pipeline] Pipeline updated successfully");
    emit update_pipeline_success();
  } catch (const std::exception &e) {
    const QString msg = QString("Failed to update pipeline: %1").arg(e.what());
    logger()->error("[Manager::update_pipeline] Failed to update pipeline: {}", e.what());
    emit update_pipeline_failure(msg);
  }
}

void Manager::start_raw_record(std::filesystem::path record_path) {
  auto payload = nlohmann::json{
      {"type", "start_recording"},
      {"record_path", record_path},
  };

  std::lock_guard lock(mtx_);
  if (!scheduler_ || !scheduler_->is_running()) {
    const QString msg = QString("Pipeline is not running");
    logger()->error("[Manager::start_raw_record] Pipeline is not running");
    emit raw_record_started_failure(msg);
    return;
  }

  if (raw_recording_active_) {
    const QString msg = QString("Recording already in progress");
    logger()->warn("[Manager::start_raw_record] Recording already in progress");
    emit raw_record_started_failure(msg);
    return;
  }

  if (!scheduler_->ui_try_send("raw_record", std::move(payload))) {
    const QString msg = QString("Failed to enqueue start_recording event");
    logger()->error("[Manager::start_raw_record] Failed to enqueue start_recording event");
    emit raw_record_started_failure(msg);
    return;
  }

  raw_recording_active_ = true;

  logger()->info("[Manager::start_raw_record] Recording request enqueued");
  emit raw_record_started_success();
}

void Manager::stop_raw_record() {
  std::lock_guard lock(mtx_);
  if (!scheduler_ || !scheduler_->is_running()) {
    const QString msg = QString("Pipeline is not running");
    logger()->error("[Manager::stop_raw_record] Pipeline is not running");
    emit raw_record_stopped_failure(msg);
    return;
  }

  if (!raw_recording_active_) {
    const QString msg = QString("No active recording to stop");
    logger()->warn("[Manager::stop_raw_record] No active recording to stop");
    emit raw_record_stopped_failure(msg);
    return;
  }

  nlohmann::json payload{{"type", "stop_recording"}};
  if (!scheduler_->ui_try_send("raw_record", std::move(payload))) {
    const QString msg = QString("Failed to enqueue stop_recording event");
    logger()->error("[Manager::stop_raw_record] Failed to enqueue stop_recording event");
    emit raw_record_stopped_failure(msg);
    return;
  }

  logger()->info("[Manager::stop_raw_record] Stop request enqueued");
  raw_recording_active_ = false;
  emit raw_record_stopped_success();
}

void Manager::start_metrics_updates() {
  if (!metrics_timer_) {
    return;
  }
  if (!metrics_timer_->isActive()) {
    metrics_timer_->start();
  }
}

void Manager::stop_metrics_updates() {
  if (!metrics_timer_) {
    return;
  }
  if (metrics_timer_->isActive()) {
    metrics_timer_->stop();
  }
  emit metrics_updated(0.0);
}

void Manager::poll_metrics() {
  if (!scheduler_ || !scheduler_->is_running()) { // Not running
    emit metrics_updated(0.0);
    return;
  }

  auto snapshot = scheduler_->metrics();

  if (snapshot.empty()) { // No metrics available
    emit metrics_updated(0.0);
    return;
  }

  HOLOVIBES_CHECK(snapshot.contains("source"), "Missing 'source' node metrics");
  double input_fps = snapshot.at("source").runs_per_second;
  input_fps *= s_.load_batch;
  emit metrics_updated(input_fps);
}

void Manager::start_event_polling() {
  if (!events_timer_) {
    return;
  }
  if (!events_timer_->isActive()) {
    events_timer_->start();
  }
}

void Manager::stop_event_polling() {
  if (!events_timer_) {
    return;
  }
  if (events_timer_->isActive()) {
    events_timer_->stop();
  }
}

void Manager::poll_events() {
  if (!scheduler_ || !scheduler_->is_running()) { // Not running
    return;
  }

  while (true) {
    auto event = scheduler_->ui_try_receive();
    if (!event.has_value()) {
      break;
    }

    // Handle event
    logger()->info("[Manager::poll_events] Received event from node '{}': {}", event->node_id,
                   event->data.dump());

    std::string type;
    if (auto it = event->data.find("type"); it != event->data.end() && it->is_string()) {
      type = it->get<std::string>();
    }

    if (type == "recording_finished") {
      std::string path_str;
      if (auto pit = event->data.find("path"); pit != event->data.end() && pit->is_string()) {
        path_str = pit->get<std::string>();
      }

      bool should_emit = false;
      {
        std::lock_guard lock(mtx_);
        if (raw_recording_active_) {
          raw_recording_active_ = false;
          should_emit           = true;
        }
      }

      if (should_emit) {
        logger()->info("[Manager::poll_events] Recording finished successfully at {}", path_str);
        emit raw_record_stopped_success();
      }
    }
  }
}

void Manager::build_and_run() {
  using Scheduler = holoflow::runtime::Scheduler;
  stop_metrics_updates();
  stop_event_polling();
  build_graph_spec();

  // TODO: Proper log path in app data folder
  using namespace std::chrono;
  constexpr const char *LOG_FOLDER_PATH = "logs";

  const std::filesystem::path log_root{LOG_FOLDER_PATH};
  std::error_code             log_ec;
  std::filesystem::create_directories(log_root, log_ec);

  auto dot  = holoflow::core::to_dot(spec_);
  auto t    = floor<seconds>(system_clock::now());
  auto date = std::format("{:%Y-%m-%d_%H-%M-%S}", t);

  const auto pipeline_path = log_root / std::format("pipeline_{}.dot", date);
  std::ofstream(pipeline_path) << dot;
  logger()->info("[Manager::build_and_run] Pipeline graph saved to {}", pipeline_path.string());

  auto prev_output = std::move(compiler_output_);
  compiler_output_ =
      holoflow::runtime::Compiler(registry_, log_root).compile(spec_, std::move(prev_output));

  auto &graph     = compiler_output_->graph;
  auto &sections  = compiler_output_->sections;
  auto &resources = compiler_output_->resources;

  auto dot_compile = holoflow::runtime::to_dot(*compiler_output_, registry_);
  t                = floor<seconds>(system_clock::now());
  date             = std::format("{:%Y-%m-%d_%H-%M-%S}", t);

  const auto compiled_path = log_root / std::format("pipeline_compiled_{}.dot", date);
  std::ofstream(compiled_path) << dot_compile;
  logger()->info("[Manager::build_and_run] Compiled pipeline graph saved to {}",
                 compiled_path.string());

  const auto metrics_interval = metrics_timer_
                                    ? std::chrono::milliseconds{metrics_timer_->interval()}
                                    : std::chrono::milliseconds{1000};

  scheduler_ = std::make_unique<Scheduler>(graph, sections, resources, metrics_interval);
  scheduler_->set_metrics_interval(metrics_interval);
  raw_recording_active_ = false;
  scheduler_->start();
  start_metrics_updates();
  poll_metrics();
  start_event_polling();
}

void Manager::build_graph_spec() {
  logger()->info("[Manager::build_graph_spec] Building graph spec...");
  HOLOVIBES_CHECK(settings_dirty_, "Settings are not dirty, no need to rebuild graph spec");

  reset_graph_spec();
  guess_optimizations();
  guess_source_dims();

  auto source = add_source();
  auto parent = source;

  std::optional<V> cpu_in_queue = std::nullopt;
  if (s_.load_method != LoadMethod::LOAD_IN_GPU) {
    cpu_in_queue     = add_cpu_in_queue(parent, 0, 0);
    auto cpu_gpu_cpy = add_cpu_gpu_cpy(*cpu_in_queue, 0, 0);
    parent           = cpu_gpu_cpy;
  }

  if (cpu_in_queue.has_value()) {
    auto record_cpu_cpu = add_cpu_cpu_cpy(*cpu_in_queue, 0, 0);
    auto record_queue   = add_record_queue(record_cpu_cpu, 0, 0);
    auto raw_record     = add_raw_record(record_queue, 0, 0);
    (void)raw_record;
  } else {
    logger()->warn(
        "[Manager::build_graph_spec] Unable to add raw recording node: missing CPU queue");
  }

  auto gpu_in_queue = add_gpu_in_queue(parent, 0, 0);
  auto to_cf32      = add_to_cf32(gpu_in_queue, 0, 0);
  parent            = to_cf32;

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

    if (s_.pp_convolution) {
      auto convolution = add_xy_convolution(parent, 0, 0);
      parent           = convolution;
    }

    auto pctclip       = add_xy_pctclip(parent, 0, 0);
    auto to_u8         = add_xy_to_u8(pctclip, 0, 0);
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
    auto crop      = add_xz_crop2frames(slide_avg, 0, 0);
    auto to_u8     = add_xz_to_u8(crop, 0, 0);
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
    auto crop      = add_yz_crop2frames(slide_avg, 0, 0);
    auto to_u8     = add_yz_to_u8(crop, 0, 0);
    auto rotation  = add_yz_rotation(to_u8, 0, 0);
    auto gpu_out   = add_yz_gpu_out_queue(rotation, 0, 0);
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

  else if (s_.import_source == ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO) {
    auto cfg_file = std::ifstream(s_.camera_config_path);
    if (!cfg_file.is_open()) {
      throw std::runtime_error(
          std::format("Could not open camera config file: {}", s_.camera_config_path.string()));
    }

    auto cfg    = nlohmann::json::parse(cfg_file).at("s710");
    src_width_  = cfg.at("Width").get<int>();
    src_height_ = cfg.at("Height").get<int>();
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

  HOLOVIBES_UNIMPLEMENTED();
}

Manager::V Manager::add_cpu_in_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(
      parent, out_idx, in_idx, "cpu_in_queue", "BatchQueue",
      BatchQueueSettings{
          .target_capacity = s_.cpu_in_size,
          .output_size     = s_.time_window,
          .output_stride   = opti_cpu_stride_ ? s_.time_stride : s_.time_window,
      });
}

Manager::V Manager::add_record_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "record_queue", "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.recording_count,
                                                .output_size     = s_.time_window,
                                                .output_stride   = s_.time_stride,
                                            });
}

Manager::V Manager::add_cpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "cpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

Manager::V Manager::add_xy_raw_display(V parent, int out_idx, int in_idx) {
  using sinks::DisplayTensorSettings;
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "xy_raw_display",
                                               "DisplayTensorXYRaw", DisplayTensorSettings{});
}

Manager::V Manager::add_raw_record(V parent, int out_idx, int in_idx) {
  using sinks::HolofileSettings;
  return add_node_after<HolofileSettings>(parent, out_idx, in_idx, "raw_record", "HolofileWriter",
                                          HolofileSettings{
                                              .path              = s_.recording_path.string(),
                                              .count             = s_.recording_count,
                                              .pipeline_settings = settings_to_old_json(s_),
                                          });
}

Manager::V Manager::add_cpu_gpu_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "cpu_gpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Device,
                                        });
}

Manager::V Manager::add_gpu_in_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(
      parent, out_idx, in_idx, "gpu_in_queue", "BatchQueue",
      BatchQueueSettings{
          .target_capacity = s_.gpu_in_size,
          .output_size     = s_.time_window,
          .output_stride   = opti_gpu_stride_ ? s_.time_stride : s_.time_window,
      });
}

Manager::V Manager::add_to_cf32(V parent, int out_idx, int in_idx) {
  using syncs::ConversionSettings;
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "to_cf32", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::CF32,
                                                .strategy = ConversionSettings::Strategy::Real,
                                            });
}

Manager::V Manager::add_spacial_transform(V parent, int out_idx, int in_idx) {
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

Manager::V Manager::add_spacial_filter(V parent, int out_idx, int in_idx) {
  using syncs::Filter2DSettings;
  return add_node_after<Filter2DSettings>(parent, out_idx, in_idx, "spacial_filter", "Filter2D",
                                          Filter2DSettings{
                                              .r_inner = s_.filter_r_inner,
                                              .r_outer = s_.filter_r_outer,
                                              .s_inner = s_.filter_smooth_inner,
                                              .s_outer = s_.filter_smooth_outer,
                                          });
}

Manager::V Manager::add_time_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  auto time_stride = (opti_cpu_stride_ || opti_gpu_stride_) ? s_.time_window : s_.time_stride;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "time_queue", "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.time_stride * 2,
                                                .output_size     = s_.time_window,
                                                .output_stride   = time_stride,
                                            });
}

Manager::V Manager::add_time_transform(V parent, int out_idx, int in_idx) {
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

Manager::V Manager::add_to_f32(V parent, int out_idx, int in_idx) {
  using syncs::ConversionSettings;
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "to_f32", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::F32,
                                                .strategy = ConversionSettings::Strategy::Modulus,
                                            });
}

Manager::V Manager::add_debounce_queue(V parent, int out_idx, int in_idx) {
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

Manager::V Manager::add_xy_cut_avg(V parent, int out_idx, int in_idx) {
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

Manager::V Manager::add_fft_shift(V parent, int out_idx, int in_idx) {
  using syncs::FFTShiftSettings;
  return add_node_after<FFTShiftSettings>(parent, out_idx, in_idx, "fft_shift", "FFTShift",
                                          FFTShiftSettings{});
}

Manager::V Manager::add_xy_identity_0(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "xy_identity_0", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Device,
                                        });
}

Manager::V Manager::add_xy_registration(V parent, int out_idx, int in_idx) {
  using syncs::RegistrationSettings;

  return add_node_after<RegistrationSettings>(parent, out_idx, in_idx, "xy_registration",
                                              "Registration",
                                              RegistrationSettings{
                                                  .radius = s_.pp_registration_radius,
                                              });
}

Manager::V Manager::add_xy_slide_avg(V parent, int out_idx, int in_idx) {
  using asyncs::SlidingAverageSettings;
  return add_node_after<SlidingAverageSettings>(
      parent, out_idx, in_idx, "xy_slide_avg", "SlidingAverage",
      SlidingAverageSettings{.target_capacity = 128,
                             .window_size     = static_cast<size_t>(s_.pp_accumulation)});
}

Manager::V Manager::add_xy_identity_1(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
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
  using syncs::ConvolutionSettings;
  return add_node_after<ConvolutionSettings>(parent, out_idx, in_idx, "xy_convolution",
                                             "Convolution",
                                             ConvolutionSettings{
                                                 .kernel_file = s_.pp_convolution_path,
                                                 .divide      = s_.pp_convolution_divide,
                                             });
}

Manager::V Manager::add_xy_pctclip(V parent, int out_idx, int in_idx) {
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

Manager::V Manager::add_xy_to_u8(V parent, int out_idx, int in_idx) {
  using syncs::ConversionSettings;
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "xy_to_u8", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::U8,
                                                .strategy = ConversionSettings::Strategy::Scaled,
                                            });
}

Manager::V Manager::add_xy_gpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xy_gpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.gpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_xy_gpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "xy_gpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

Manager::V Manager::add_xy_cpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xy_cpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.cpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_xy_processed_display(V parent, int out_idx, int in_idx) {
  using sinks::DisplayTensorSettings;
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "xy_processed_display",
                                               "DisplayTensorXY", DisplayTensorSettings{});
}

Manager::V Manager::add_xz_cut_avg(V parent, int out_idx, int in_idx) {
  using syncs::AverageSettings;
  return add_node_after<AverageSettings>(parent, out_idx, in_idx, "xz_cut_avg", "Average",
                                         AverageSettings{
                                             .axis  = 1,
                                             .start = s_.time_y_begin,
                                             .end   = s_.time_y_end,
                                         });
}

Manager::V Manager::add_xz_reshape(V parent, int out_idx, int in_idx) {
  using syncs::ReshapeSettings;
  return add_node_after<ReshapeSettings>(
      parent, out_idx, in_idx, "xz_reshape", "Reshape",
      ReshapeSettings{
          .shape = {1, static_cast<size_t>(src_height_), static_cast<size_t>(s_.time_window)},
      });
}

Manager::V Manager::add_xz_slide_avg(V parent, int out_idx, int in_idx) {
  using asyncs::SlidingAverageSettings;
  return add_node_after<SlidingAverageSettings>(
      parent, out_idx, in_idx, "xz_slide_avg", "SlidingAverage",
      SlidingAverageSettings{.target_capacity = 128,
                             .window_size     = static_cast<size_t>(s_.pp_accumulation)});
}

Manager::V Manager::add_xz_to_u8(V parent, int out_idx, int in_idx) {
  using syncs::ConversionSettings;
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "xz_to_u8", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::U8,
                                                .strategy = ConversionSettings::Strategy::Scaled,
                                            });
}

Manager::V Manager::add_xz_crop2frames(V parent, int out_idx, int in_idx) {
  using syncs::CropSettings;
  return add_node_after<CropSettings>(
      parent, out_idx, in_idx, "xz_crop2frames", "Crop",
      CropSettings{
          .origin = {0, 10, 0},
          .shape  = {static_cast<size_t>(1), static_cast<size_t>(src_height_ - 20),
                     static_cast<size_t>(s_.time_window)},
      });
}

Manager::V Manager::add_xz_gpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xz_gpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.gpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_xz_gpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "xz_gpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

Manager::V Manager::add_xz_cpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "xz_cpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.cpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_xz_processed_display(V parent, int out_idx, int in_idx) {
  using sinks::DisplayTensorSettings;
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "xz_processed_display",
                                               "DisplayTensorXZ", DisplayTensorSettings{});
}

Manager::V Manager::add_yz_cut_avg(V parent, int out_idx, int in_idx) {
  using syncs::AverageSettings;
  return add_node_after<AverageSettings>(parent, out_idx, in_idx, "yz_cut_avg", "Average",
                                         AverageSettings{
                                             .axis  = 2,
                                             .start = s_.time_x_begin,
                                             .end   = s_.time_x_end,
                                         });
}

Manager::V Manager::add_yz_reshape(V parent, int out_idx, int in_idx) {
  using syncs::ReshapeSettings;
  return add_node_after<ReshapeSettings>(
      parent, out_idx, in_idx, "yz_reshape", "Reshape",
      ReshapeSettings{
          .shape = {1, static_cast<size_t>(src_height_), static_cast<size_t>(s_.time_window)},
      });
}

Manager::V Manager::add_yz_slide_avg(V parent, int out_idx, int in_idx) {
  using asyncs::SlidingAverageSettings;
  return add_node_after<SlidingAverageSettings>(
      parent, out_idx, in_idx, "yz_slide_avg", "SlidingAverage",
      SlidingAverageSettings{.target_capacity = 128,
                             .window_size     = static_cast<size_t>(s_.pp_accumulation)});
}

Manager::V Manager::add_yz_to_u8(V parent, int out_idx, int in_idx) {
  using syncs::ConversionSettings;
  return add_node_after<ConversionSettings>(parent, out_idx, in_idx, "yz_to_u8", "Conversion",
                                            ConversionSettings{
                                                .target   = ConversionSettings::Target::U8,
                                                .strategy = ConversionSettings::Strategy::Scaled,
                                            });
}

Manager::V Manager::add_yz_rotation(V parent, int out_idx, int in_idx) {
  using syncs::RotationSettings;
  return add_node_after<RotationSettings>(parent, out_idx, in_idx, "yz_rotation", "Rotation",
                                          RotationSettings{
                                              .angle = 90,
                                          });
}

Manager::V Manager::add_yz_crop2frames(V parent, int out_idx, int in_idx) {
  using syncs::CropSettings;
  return add_node_after<CropSettings>(
      parent, out_idx, in_idx, "yz_crop", "Crop",
      CropSettings{
          .origin = {0, 10, 0},
          .shape  = {static_cast<size_t>(1), static_cast<size_t>(src_height_ - 20),
                     static_cast<size_t>(s_.time_window)},
      });
}

Manager::V Manager::add_yz_gpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "yz_gpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.gpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_yz_gpu_cpu_cpy(V parent, int out_idx, int in_idx) {
  using syncs::MemcpySettings;
  return add_node_after<MemcpySettings>(parent, out_idx, in_idx, "yz_gpu_cpu", "Memcpy",
                                        MemcpySettings{
                                            .target = MemcpySettings::Target::Host,
                                        });
}

Manager::V Manager::add_yz_cpu_out_queue(V parent, int out_idx, int in_idx) {
  using asyncs::BatchQueueSettings;
  return add_node_after<BatchQueueSettings>(parent, out_idx, in_idx, "yz_cpu_out_queue",
                                            "BatchQueue",
                                            BatchQueueSettings{
                                                .target_capacity = s_.cpu_out_size,
                                                .output_size     = 1,
                                                .output_stride   = 1,
                                            });
}

Manager::V Manager::add_yz_processed_display(V parent, int out_idx, int in_idx) {
  using sinks::DisplayTensorSettings;
  return add_node_after<DisplayTensorSettings>(parent, out_idx, in_idx, "yz_processed_display",
                                               "DisplayTensorYZ", DisplayTensorSettings{});
}

} // namespace holovibes::pipeline
