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
#include "graph_builder.hh"
#include "graph_builder_v2.hh"
#include "holoflow/runtime/graph_display.hh"
#include "holonp/arange.hh"
#include "holonp/meshgrid.hh"
#include "holonp/slice_copy.hh"
#include "holonp/transpose.hh"
#include "holotask/asyncs/batch_queue.hh"
#include "holotask/asyncs/slide_avg.hh"
#include "holotask/sinks/holofile.hh"
#include "holotask/sources/ametek_s710_euresys_coaxlink_octo.hh"
#include "holotask/sources/ametek_s711_euresys_coaxlink_qsfp+.hh"
#include "holotask/sources/holofile.hh"
#include "holotask/syncs/angular_spectrum.hh"
#include "holotask/syncs/average.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/convolution.hh"
#include "holotask/syncs/crop.hh"
#include "holotask/syncs/fft_shift.hh"
#include "holotask/syncs/filter2d.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/log.hh"
#include "holotask/syncs/memcpy.hh"
#include "holotask/syncs/pca.hh"
#include "holotask/syncs/pct_clip.hh"
#include "holotask/syncs/registration.hh"
#include "holotask/syncs/reshape.hh"
#include "holotask/syncs/rotation.hh"
#include "holotask/syncs/stft.hh"
#include "logger.hh"
#include "settings_loader.hh"
#include "tasks/sinks/display_tensor.hh"

using namespace holotask;
using namespace holonp;

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
                 ui::TensorDisplayWidget *xy_raw_widget,
                 ui::TensorDisplayWidget *raw_spectrum_widget,
                 ui::TensorDisplayWidget *processed_spectrum_widget,
                 ui::TensorDisplayWidget *shack_hartmann_widget)
    : xy_processed_widget_(xy_processed_widget), xz_processed_widget_(xz_processed_widget),
      yz_processed_widget_(yz_processed_widget), xy_raw_widget_(xy_raw_widget),
      raw_spectrum_widget_(raw_spectrum_widget),
      processed_spectrum_widget_(processed_spectrum_widget),
      shack_hartmann_widget_(shack_hartmann_widget) {
  // clang-format off
  reg_async<asyncs::BatchQueueFactory>(registry_, "BatchQueue");
  reg_async<asyncs::SlidingAverageFactory>(registry_, "SlidingAverage");
  reg_sync<holovibes::tasks::sinks::DisplayTensorFactory>(registry_, "DisplayTensorXY", xy_processed_widget_);
  reg_sync<holovibes::tasks::sinks::DisplayTensorFactory>(registry_, "DisplayTensorXZ", xz_processed_widget_);
  reg_sync<holovibes::tasks::sinks::DisplayTensorFactory>(registry_, "DisplayTensorYZ", yz_processed_widget_);
  reg_sync<holovibes::tasks::sinks::DisplayTensorFactory>(registry_, "DisplayTensorXYRaw", xy_raw_widget_);
  reg_sync<holovibes::tasks::sinks::DisplayTensorFactory>(registry_, "DisplayRawSpectrum", raw_spectrum_widget_);
  reg_sync<holovibes::tasks::sinks::DisplayTensorFactory>(registry_, "DisplayProcessedSpectrum", processed_spectrum_widget_);
  reg_sync<holovibes::tasks::sinks::DisplayTensorFactory>(registry_, "DisplayTensorShackHartmann", shack_hartmann_widget_);
  reg_sync<sinks::HolofileFactory>(registry_, "HolofileWriter");
  reg_sync<sources::HolofileFactory>(registry_, "Holofile");
  reg_sync<sources::AmetekS710EuresysCoaxlinkOctoFactory>(registry_, "AmetekS710EuresysCoaxlinkOcto");
  reg_sync<sources::AmetekS711EuresysCoaxlinkQSFPFactory>(registry_, "AmetekS711EuresysCoaxlinkQSFP+");
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
  reg_sync<syncs::LogFactory>(registry_, "Log");
  reg_sync<syncs::RegistrationFactory>(registry_, "Registration");
  reg_sync<syncs::ReshapeFactory>(registry_, "Reshape");
  reg_sync<syncs::CropFactory>(registry_, "Crop");
  reg_sync<syncs::RotationFactory>(registry_, "Rotation");

  reg_sync<ArangeFactory>(registry_, "Arange");
  reg_sync<MeshgridFactory>(registry_, "Meshgrid");
  reg_sync<TransposeFactory>(registry_, "Transpose");
  reg_sync<SliceCopyFactory>(registry_, "SliceCopy");
  // clang-format on

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

  if (!scheduler_->ui_try_send("record", std::move(payload))) {
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
  if (!scheduler_->ui_try_send("record", std::move(payload))) {
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

  HOLOVIBES_CHECK(snapshot.contains("source_0"), "Missing 'source' node metrics");
  double input_fps = snapshot.at("source_0").runs_per_second;
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

  auto json = holoflow::core::to_json(spec_);
  auto dot  = holoflow::core::to_dot(spec_);
  auto t    = floor<seconds>(system_clock::now());
  auto date = std::format("{:%Y-%m-%d_%H-%M-%S}", t);

  const auto pipeline_path = log_root / std::format("pipeline_{}.dot", date);
  std::ofstream(pipeline_path) << dot;
  logger()->info("[Manager::build_and_run] Pipeline graph saved to {}", pipeline_path.string());
  std::ofstream json_file(log_root / std::format("pipeline_{}.json", date));
  json_file << json.dump(2);
  logger()->info("[Manager::build_and_run] Pipeline JSON saved to {}",
                 (log_root / std::format("pipeline_{}.json", date)).string());

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

  if (false) {
    using json      = nlohmann::json;
    using GraphSpec = holoflow::core::GraphSpec;
    using NodeSpec  = holoflow::core::NodeSpec;
    using EdgeSpec  = holoflow::core::EdgeSpec;

    auto x = boost::add_vertex(
        NodeSpec{
            .name = "source_0",
            .kind = "Arange",
            .settings =
                json{
                    {"start", 0},
                    {"stop", 512},
                    {"step", 1},
                    {"dtype", "F32"},
                    {"device", "Device"},
                },
        },
        spec_);

    auto y = boost::add_vertex(
        NodeSpec{
            .name = "arange_y",
            .kind = "Arange",
            .settings =
                json{
                    {"start", 0},
                    {"stop", 512},
                    {"step", 1},
                    {"dtype", "F32"},
                    {"device", "Device"},
                },
        },
        spec_);

    auto XY = boost::add_vertex(
        NodeSpec{
            .name = "meshgrid_xy",
            .kind = "Meshgrid",
            .settings =
                json{
                    {"indexing", "xy"},
                },
        },
        spec_);

    auto X_u8 = boost::add_vertex(
        NodeSpec{
            .name     = "convert_x_u8",
            .kind     = "Conversion",
            .settings = json{{"target", "U8"}, {"strategy", "Scaled"}},
        },
        spec_);

    auto Y_u8 = boost::add_vertex(
        NodeSpec{
            .name     = "convert_y_u8",
            .kind     = "Conversion",
            .settings = json{{"target", "U8"}, {"strategy", "Scaled"}},
        },
        spec_);

    auto X_display = boost::add_vertex(
        NodeSpec{
            .name     = "display_xy",
            .kind     = "DisplayTensorXY",
            .settings = json{{"refresh_rate_hz", 30}},
        },
        spec_);

    auto Y_display = boost::add_vertex(
        NodeSpec{
            .name     = "display_yz",
            .kind     = "DisplayTensorYZ",
            .settings = json{{"refresh_rate_hz", 30}},
        },
        spec_);

    boost::add_edge(x, XY,
                    EdgeSpec{
                        .out_idx = 0,
                        .in_idx  = 0,
                    },
                    spec_);

    boost::add_edge(y, XY,
                    EdgeSpec{
                        .out_idx = 0,
                        .in_idx  = 1,
                    },
                    spec_);

    boost::add_edge(XY, X_u8,
                    EdgeSpec{
                        .out_idx = 0,
                        .in_idx  = 0,
                    },
                    spec_);

    boost::add_edge(XY, Y_u8,
                    EdgeSpec{
                        .out_idx = 1,
                        .in_idx  = 0,
                    },
                    spec_);

    boost::add_edge(X_u8, X_display,
                    EdgeSpec{
                        .out_idx = 0,
                        .in_idx  = 0,
                    },
                    spec_);

    boost::add_edge(Y_u8, Y_display,
                    EdgeSpec{
                        .out_idx = 0,
                        .in_idx  = 0,
                    },
                    spec_);

    // boost::add_edge(XY, X_display,
    //                 EdgeSpec{
    //                     .out_idx = 0,
    //                     .in_idx  = 0,
    //                 },
    //                 spec_);

    // boost::add_edge(XY, Y_display,
    //                 EdgeSpec{
    //                     .out_idx = 1,
    //                     .in_idx  = 0,
    //                 },
    //                 spec_);

    return;
  }

  GraphBuilder builder{spec_, s_, src_width_, src_height_, opti_cpu_stride_, opti_gpu_stride_};
  builder.build();

  GraphBuilder_v2 builder_v2{s_, registry_};
  spec_ = builder_v2.build();

  settings_dirty_ = false;
  logger()->debug("[Manager::build_graph_spec] Graph spec built successfully");
}

void Manager::reset_graph_spec() { spec_ = holoflow::core::GraphSpec{}; }

void Manager::guess_optimizations() {
  bool load_in_gpu     = (s_.load_method == LoadMethod::LOAD_IN_GPU);
  bool stride_multiple = (s_.time_stride % s_.time_window == 0);
  opti_cpu_stride_     = stride_multiple && !load_in_gpu;
  opti_gpu_stride_     = (stride_multiple && load_in_gpu) || opti_cpu_stride_;
  // FIXME : disabled for now due to time stride in raw record
  opti_cpu_stride_ = false;
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

  else if (s_.import_source == ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP) {
    auto cfg_file = std::ifstream(s_.camera_config_path);
    if (!cfg_file.is_open()) {
      throw std::runtime_error(
          std::format("Could not open camera config file: {}", s_.camera_config_path.string()));
    }

    auto cfg    = nlohmann::json::parse(cfg_file).at("s711");
    src_width_  = cfg.at("Width").get<int>();
    src_height_ = cfg.at("Height").get<int>();
    return;
  }

  HOLOVIBES_UNIMPLEMENTED();
}

} // namespace holovibes::pipeline
