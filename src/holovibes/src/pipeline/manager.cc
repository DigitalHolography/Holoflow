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

#include <QStandardPaths>
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
#include "holonp/abs.hh"
#include "holonp/add.hh"
#include "holonp/arange.hh"
#include "holonp/argmax.hh"
#include "holonp/asarray.hh"
#include "holonp/ascontiguousarray.hh"
#include "holonp/assign.hh"
#include "holonp/concatenate.hh"
#include "holonp/conj.hh"
#include "holonp/copy.hh"
#include "holonp/div.hh"
#include "holonp/empty.hh"
#include "holonp/equal.hh"
#include "holonp/fft.hh"
#include "holonp/fft2.hh"
#include "holonp/fftshift.hh"
#include "holonp/irfft2.hh"
#include "holonp/max.hh"
#include "holonp/mean.hh"
#include "holonp/mean_abs.hh"
#include "holonp/meshgrid.hh"
#include "holonp/min.hh"
#include "holonp/mul.hh"
#include "holonp/normalize.hh"
#include "holonp/reshape.hh"
#include "holonp/rfft.hh"
#include "holonp/rfft2.hh"
#include "holonp/slice.hh"
#include "holonp/sub.hh"
#include "holonp/transpose.hh"
#include "holonp/where.hh"
#include "holonp/zeros.hh"
#include "holotask/asyncs/batch_queue.hh"
#include "holotask/asyncs/slide_avg.hh"
#include "holotask/sinks/holofile.hh"
#include "holotask/sources/ametek_s710_euresys_coaxlink_octo.hh"
#include "holotask/sources/ametek_s711_euresys_coaxlink_qsfp+.hh"
#include "holotask/sources/fresnel_qin.hh"
#include "holotask/sources/fresnel_qout.hh"
#include "holotask/sources/holofile.hh"
#include "holotask/syncs/angular_spectrum.hh"
#include "holotask/syncs/average.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/convolution.hh"
#include "holotask/syncs/correct_phase.hh"
#include "holotask/syncs/crop.hh"
#include "holotask/syncs/fft_shift.hh"
#include "holotask/syncs/filter2d.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/log.hh"
#include "holotask/syncs/memcpy.hh"
#include "holotask/syncs/pca.hh"
#include "holotask/syncs/pct_clip.hh"
#include "holotask/syncs/registration.hh"
#include "holotask/syncs/rotation.hh"
#include "holotask/syncs/stft.hh"
#include "holotask/syncs/wrap2pi.hh"
#include "holotask/syncs/zernike.hh"
#include "holotask/syncs/zernike_phase.hh"
#include "logger.hh"
#include "settings_loader.hh"
#include "tasks/sinks/display_tensor.hh"
#include "tasks/sinks/display_zernike_coefficients.hh"
#include "ui/widgets/auto_focus_widget.hh"

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

Manager::Manager(ui::AutoFocusWidget     *autofocus_widget,
                 ui::TensorDisplayWidget *xy_processed_widget,
                 ui::TensorDisplayWidget *xz_processed_widget,
                 ui::TensorDisplayWidget *yz_processed_widget,
                 ui::TensorDisplayWidget *xy_raw_widget,
                 ui::TensorDisplayWidget *raw_spectrum_widget,
                 ui::TensorDisplayWidget *processed_spectrum_widget,
                 ui::TensorDisplayWidget *shack_hartmann_widget,
                 ui::TensorDisplayWidget *shack_hartmann_xcorr_widget,
                 ui::TensorDisplayWidget *zernike_phase_widget)
    : autofocus_widget_(autofocus_widget), xy_processed_widget_(xy_processed_widget),
      xz_processed_widget_(xz_processed_widget), yz_processed_widget_(yz_processed_widget),
      xy_raw_widget_(xy_raw_widget), raw_spectrum_widget_(raw_spectrum_widget),
      processed_spectrum_widget_(processed_spectrum_widget),
      shack_hartmann_widget_(shack_hartmann_widget),
      shack_hartmann_xcorr_widget_(shack_hartmann_xcorr_widget),
      zernike_phase_widget_(zernike_phase_widget) {

  register_components();

  // Setup UI metric polling (1 FPS)
  metrics_timer_ = new QTimer(this);
  metrics_timer_->setInterval(1000);
  metrics_timer_->setTimerType(Qt::TimerType::CoarseTimer);
  connect(metrics_timer_, &QTimer::timeout, this, &Manager::poll_metrics);

  // Setup Event polling (200 FPS for responsive UI feedback)
  events_timer_ = new QTimer(this);
  events_timer_->setInterval(5);
  events_timer_->setTimerType(Qt::TimerType::CoarseTimer);
  connect(events_timer_, &QTimer::timeout, this, &Manager::poll_events);
}

void Manager::register_components() {
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
  reg_sync<holovibes::tasks::sinks::DisplayTensorFactory>(registry_, "DisplayTensorShackHartmannXcorr", shack_hartmann_xcorr_widget_);
  reg_sync<holovibes::tasks::sinks::DisplayTensorFactory>(registry_, "DisplayTensorZernikePhase", zernike_phase_widget_);
  reg_sync<holovibes::tasks::sinks::DisplayZernikeCoefficientsFactory>(registry_, "DisplayZernikeCoefficients", autofocus_widget_);
  reg_sync<sinks::HolofileFactory>(registry_, "HolofileWriter");
  reg_sync<sources::HolofileFactory>(registry_, "Holofile");
  reg_sync<sources::AmetekS710EuresysCoaxlinkOctoFactory>(registry_, "AmetekS710EuresysCoaxlinkOcto");
  reg_sync<sources::AmetekS711EuresysCoaxlinkQSFPFactory>(registry_, "AmetekS711EuresysCoaxlinkQSFP+");
  reg_sync<sources::FresnelQinFactory>(registry_, "FresnelQin");
  reg_sync<sources::FresnelQoutFactory>(registry_, "FresnelQout");
  reg_sync<syncs::AngularSpectrumFactory>(registry_, "AngularSpectrum");
  reg_sync<syncs::AverageFactory>(registry_, "Average");
  reg_sync<syncs::ConversionFactory>(registry_, "Conversion");
  reg_sync<syncs::FFTShiftFactory>(registry_, "FFTShift");
  reg_sync<syncs::FresnelDiffractionFactory>(registry_, "FresnelDiffraction");
  reg_sync<syncs::MemcpyFactory>(registry_, "Memcpy");
  reg_sync<syncs::PcaFactory>(registry_, "Pca");
  reg_sync<syncs::PctClipFactory>(registry_, "PctClip");
  reg_sync<syncs::CorrectPhaseFactory>(registry_, "CorrectPhase");
  reg_sync<syncs::StftFactory>(registry_, "Stft");
  reg_sync<syncs::ConvolutionFactory>(registry_, "Convolution");
  reg_sync<syncs::Filter2DFactory>(registry_, "Filter2D");
  reg_sync<syncs::LogFactory>(registry_, "Log");
  reg_sync<syncs::RegistrationFactory>(registry_, "Registration");
  reg_sync<syncs::CropFactory>(registry_, "Crop");
  reg_sync<syncs::RotationFactory>(registry_, "Rotation");
  reg_sync<syncs::Wrap2PiFactory>(registry_, "Wrap2Pi");
  reg_sync<syncs::ZernikeFactory>(registry_, "Zernike");
  reg_sync<syncs::ZernikePhaseFactory>(registry_, "ZernikePhase");
  reg_sync<ArangeFactory>(registry_, "Arange");
  reg_sync<AsArrayFactory>(registry_, "AsArray");
  reg_sync<AsContiguousArrayFactory>(registry_, "AsContiguousArray");
  reg_sync<CopyFactory>(registry_, "Copy");
  reg_sync<EmptyFactory>(registry_, "Empty");
  reg_sync<ZerosFactory>(registry_, "Zeros");
  reg_sync<MeshgridFactory>(registry_, "Meshgrid");
  reg_sync<TransposeFactory>(registry_, "Transpose");
  reg_sync<SliceFactory>(registry_, "Slice");
  reg_sync<AssignFactory>(registry_, "Assign");
  reg_sync<FFTFactory>(registry_, "FFT");
  reg_sync<FFT2Factory>(registry_, "FFT2");
  reg_sync<FFTShiftFactory>(registry_, "FFTShiftNp");
  reg_sync<AbsFactory>(registry_, "Abs");
  reg_sync<ConjFactory>(registry_, "Conj");
  reg_sync<MeanFactory>(registry_, "Mean");
  reg_sync<MeanAbsFactory>(registry_, "MeanAbs");
  reg_sync<MinFactory>(registry_, "Min");
  reg_sync<MaxFactory>(registry_, "Max");
  reg_sync<NormalizeFactory>(registry_, "Normalize");
  reg_sync<ArgmaxFactory>(registry_, "Argmax");
  reg_sync<ConcatenateFactory>(registry_, "Concatenate");
  reg_sync<RFFTFactory>(registry_, "RFFT");
  reg_sync<RFFT2Factory>(registry_, "RFFT2");
  reg_sync<IRFFT2Factory>(registry_, "IRFFT2");
  reg_sync<MulFactory>(registry_, "Mul");
  reg_sync<SubFactory>(registry_, "Sub");
  reg_sync<DivFactory>(registry_, "Div");
  reg_sync<AddFactory>(registry_, "Add");
  reg_sync<EqualFactory>(registry_, "Equal");
  reg_sync<WhereFactory>(registry_, "Where");
  reg_sync<ReshapeFactory>(registry_, "Reshape");
  // clang-format on
}

void Manager::start_pipeline() {
  logger()->info("[Manager::start_pipeline] Starting pipeline...");
  std::lock_guard lock(mtx_);

  if (scheduler_ && scheduler_->is_running()) {
    const QString msg = "Pipeline is already running";
    logger()->error("[Manager::start_pipeline] {}", msg.toStdString());
    emit start_pipeline_failure(msg);
    return;
  }

  try {
    build_and_run();
    logger()->info("[Manager::start_pipeline] Pipeline started successfully");
    emit start_pipeline_success();
  } catch (const std::exception &e) {
    const QString msg = QString("Failed to start pipeline: %1").arg(e.what());
    logger()->error("[Manager::start_pipeline] {}", msg.toStdString());
    emit start_pipeline_failure(msg);
  }
}

void Manager::stop_pipeline() {
  logger()->info("[Manager::stop_pipeline] Stopping pipeline...");
  std::lock_guard lock(mtx_);

  if (!scheduler_ || !scheduler_->is_running()) {
    const QString msg = "Pipeline is not running";
    logger()->error("[Manager::stop_pipeline] {}", msg.toStdString());
    emit stop_pipeline_failure(msg);
    return;
  }

  try {
    // Request an asynchronous stop and block until graph execution concludes safely.
    scheduler_->request_stop();
    scheduler_->wait();

    stop_metrics_updates();
    stop_event_polling();
    raw_recording_active_ = false;

    logger()->info("[Manager::stop_pipeline] Pipeline stopped successfully");
    emit stop_pipeline_success();
  } catch (const std::exception &e) {
    const QString msg = QString("Failed to stop pipeline: %1").arg(e.what());
    logger()->error("[Manager::stop_pipeline] {}", msg.toStdString());
    emit stop_pipeline_failure(msg);
  }
}

void Manager::update_pipeline(const Settings &settings) {
  logger()->info("[Manager::update_pipeline] Updating pipeline settings...");
  std::lock_guard lock(mtx_);
  s_              = settings;
  settings_dirty_ = true;

  if (!scheduler_ || !scheduler_->is_running()) {
    logger()->debug("[Manager::update_pipeline] Pipeline is not running, updated parameters only.");
    emit update_pipeline_success();
    return;
  }

  try {
    // Seamless restart mechanism
    scheduler_->request_stop();
    scheduler_->wait();
    raw_recording_active_ = false;

    build_and_run();

    logger()->info("[Manager::update_pipeline] Pipeline updated successfully");
    emit update_pipeline_success();
  } catch (const std::exception &e) {
    const QString msg = QString("Failed to update pipeline: %1").arg(e.what());
    logger()->error("[Manager::update_pipeline] {}", msg.toStdString());
    emit update_pipeline_failure(msg);
  }
}

void Manager::start_raw_record(std::filesystem::path record_path) {
  std::lock_guard lock(mtx_);
  if (!scheduler_ || !scheduler_->is_running()) {
    emit raw_record_started_failure("Pipeline is not running");
    return;
  }

  if (raw_recording_active_) {
    emit raw_record_started_failure("Recording already in progress");
    return;
  }

  auto payload = nlohmann::json{
      {"type", "start_recording"},
      {"record_path", record_path},
  };

  // Send the recording event to the UI message queue inside the scheduler
  if (!scheduler_->ui_try_send("record", std::move(payload))) {
    logger()->error("[Manager::start_raw_record] Failed to enqueue start_recording event");
    emit raw_record_started_failure("Failed to enqueue start_recording event");
    return;
  }

  raw_recording_active_ = true;
  logger()->info("[Manager::start_raw_record] Recording request enqueued to path: {}",
                 record_path.string());
  emit raw_record_started_success();
}

void Manager::stop_raw_record() {
  std::lock_guard lock(mtx_);
  if (!scheduler_ || !scheduler_->is_running()) {
    emit raw_record_stopped_failure("Pipeline is not running");
    return;
  }

  if (!raw_recording_active_) {
    emit raw_record_stopped_failure("No active recording to stop");
    return;
  }

  nlohmann::json payload{{"type", "stop_recording"}};
  if (!scheduler_->ui_try_send("record", std::move(payload))) {
    logger()->error("[Manager::stop_raw_record] Failed to enqueue stop_recording event");
    emit raw_record_stopped_failure("Failed to enqueue stop_recording event");
    return;
  }

  raw_recording_active_ = false;
  logger()->info("[Manager::stop_raw_record] Stop request enqueued");
  emit raw_record_stopped_success();
}

// --- Polling logic ---
void Manager::start_metrics_updates() {
  if (metrics_timer_ && !metrics_timer_->isActive())
    metrics_timer_->start();
}
void Manager::stop_metrics_updates() {
  if (metrics_timer_ && metrics_timer_->isActive())
    metrics_timer_->stop();
  emit metrics_updated(0.0);
}

void Manager::poll_metrics() {
  if (!scheduler_ || !scheduler_->is_running()) {
    emit metrics_updated(0.0);
    return;
  }

  auto snapshot = scheduler_->metrics();
  if (snapshot.empty()) {
    emit metrics_updated(0.0);
    return;
  }

  // Calculate FPS based on source execution rate and batch size
  HOLOVIBES_CHECK(snapshot.contains("source_0"), "Missing 'source' node metrics");
  double input_fps = snapshot.at("source_0").runs_per_second;
  input_fps *= s_.load_batch;

  emit metrics_updated(input_fps);
}

void Manager::start_event_polling() {
  if (events_timer_ && !events_timer_->isActive())
    events_timer_->start();
}
void Manager::stop_event_polling() {
  if (events_timer_ && events_timer_->isActive())
    events_timer_->stop();
}

void Manager::poll_events() {
  if (!scheduler_ || !scheduler_->is_running())
    return;

  while (true) {
    auto event = scheduler_->ui_try_receive();
    if (!event.has_value())
      break;

    logger()->info("[Manager::poll_events] Received event from node '{}': {}", event->node_id,
                   event->data.dump());

    std::string type = event->data.value("type", "");

    if (type == "recording_finished") {
      std::string path_str    = event->data.value("path", "");
      bool        should_emit = false;

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

// --- Graph Building & Execution ---
void Manager::build_and_run() {
  using Scheduler      = holoflow::runtime::Scheduler;
  using CompilerConfig = holoflow::runtime::Compiler::Config;
  using Compiler       = holoflow::runtime::Compiler;

  stop_metrics_updates();
  stop_event_polling();
  build_graph_spec();

  // Create an OS-appropriate, writable directory for logs (e.g., AppData/Local/Holovibes/logs)
  const std::filesystem::path log_root = "logs";

  std::error_code log_ec;
  std::filesystem::create_directories(log_root, log_ec);
  if (log_ec) {
    logger()->warn("[Manager::build_and_run] Failed to create log directory at: {}",
                   log_root.string());
  }

  // TODO: What should be done about this verbose logging?
  // if (dump_debug_graphs_) {
  //   dump_graph_logs(log_root);
  // }

  CompilerConfig config;
  config.log_dir             = log_root;
  config.dump_dot_on_failure = dump_debug_graphs_;
  config.verbose_tracing     = dump_debug_graphs_;

  auto     prev_output = std::move(compiler_output_);
  Compiler compiler(registry_, config);
  compiler_output_ = compiler.compile(spec_, std::move(prev_output));

  auto &graph     = compiler_output_->graph;
  auto &sections  = compiler_output_->sections;
  auto &resources = compiler_output_->resources;

  const auto metrics_interval = metrics_timer_
                                    ? std::chrono::milliseconds{metrics_timer_->interval()}
                                    : std::chrono::milliseconds{1000};

  scheduler_            = std::make_unique<Scheduler>(graph, sections, resources, metrics_interval);
  raw_recording_active_ = false;

  scheduler_->start();
  start_metrics_updates();
  poll_metrics();
  start_event_polling();
}

void Manager::dump_graph_logs(const std::filesystem::path &log_dir) {
  using namespace std::chrono;

  auto t    = floor<seconds>(system_clock::now());
  auto date = std::format("{:%Y-%m-%d_%H-%M-%S}", t);

  // Write original GraphSpec
  const auto json_path = log_dir / std::format("pipeline_{}.json", date);
  const auto dot_path  = log_dir / std::format("pipeline_{}.dot", date);

  std::ofstream(dot_path) << holoflow::core::to_dot(spec_);
  std::ofstream(json_path) << holoflow::core::to_json(spec_).dump(2);

  logger()->info("[Manager::dump_graph_logs] Pre-compile pipeline graphs saved to {}",
                 log_dir.string());
}

void Manager::build_graph_spec() {
  logger()->info("[Manager::build_graph_spec] Building graph spec...");
  HOLOVIBES_CHECK(settings_dirty_, "Settings are not dirty, no need to rebuild graph spec");

  reset_graph_spec();
  guess_optimizations();
  guess_source_dims();

  GraphBuilder_v2 builder_v2{s_, registry_};
  spec_ = builder_v2.build();

  settings_dirty_ = false;
  logger()->debug("[Manager::build_graph_spec] Graph spec built successfully");
}

void Manager::reset_graph_spec() { spec_ = holoflow::core::GraphSpec{}; }

void Manager::guess_optimizations() {
  bool load_in_gpu     = (s_.load_method == LoadMethod::LOAD_IN_GPU);
  bool stride_multiple = (s_.time_stride % s_.time_window == 0);

  opti_cpu_stride_ = stride_multiple && !load_in_gpu;
  opti_gpu_stride_ = (stride_multiple && load_in_gpu) || opti_cpu_stride_;

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

  // Refactored slightly to prevent repetition
  if (s_.import_source == ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO ||
      s_.import_source == ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP) {

    std::ifstream cfg_file(s_.camera_config_path);
    if (!cfg_file.is_open()) {
      throw std::runtime_error(
          std::format("Could not open camera config file: {}", s_.camera_config_path.string()));
    }

    auto        full_cfg = nlohmann::json::parse(cfg_file);
    std::string key =
        (s_.import_source == ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO) ? "s710" : "s711";

    auto cfg    = full_cfg.at(key);
    src_width_  = cfg.at("Width").get<int>();
    src_height_ = cfg.at("Height").get<int>();
    return;
  }

  HOLOVIBES_UNIMPLEMENTED();
}

} // namespace holovibes::pipeline
