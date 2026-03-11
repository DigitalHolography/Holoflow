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

#include <QObject>
#include <concepts>
#include <filesystem>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/graph_exec.hh"
#include "pipeline/settings.hh"

namespace holovibes::ui {
class AutoFocusWidget;
class TensorDisplayWidget;
} // namespace holovibes::ui

class QTimer;

namespace holovibes::pipeline {

/// @brief Manages the lifecycle, execution, and updating of the Holoflow pipeline.
class Manager : public QObject {
  Q_OBJECT

public:
  Manager(ui::AutoFocusWidget *autofocus_widget, ui::TensorDisplayWidget *xy_processed_widget,
          ui::TensorDisplayWidget *xz_processed_widget,
          ui::TensorDisplayWidget *yz_processed_widget, ui::TensorDisplayWidget *xy_raw_widget,
          ui::TensorDisplayWidget *raw_spectrum_widget,
          ui::TensorDisplayWidget *processed_spectrum_widget,
          ui::TensorDisplayWidget *shack_hartmann_widget,
          ui::TensorDisplayWidget *shack_hartmann_xcorr_widget,
          ui::TensorDisplayWidget *zernike_phase_widget);

  ~Manager() override = default;

  /// @brief Compiles and starts the pipeline graph.
  void start_pipeline();

  /// @brief Requests a stop and waits for the pipeline to halt cleanly.
  void stop_pipeline();

  /// @brief Updates settings and dynamically rebuilds/restarts the pipeline if currently running.
  void update_pipeline(const Settings &settings);

  /// @brief Sends an event to the pipeline to start writing raw data to disk.
  void start_raw_record(std::filesystem::path record_path);

  /// @brief Sends an event to the pipeline to stop writing raw data.
  void stop_raw_record();

signals:
  // Lifecycle signals
  void start_pipeline_success();
  void start_pipeline_failure(const QString &error);
  void stop_pipeline_success();
  void stop_pipeline_failure(const QString &error);
  void update_pipeline_success();
  void update_pipeline_failure(const QString &error);

  // Metric signals
  void metrics_updated(double input_fps);

  // Recording signals
  void raw_record_started_success();
  void raw_record_started_failure(const QString &error);
  void raw_record_stopped_success();
  void raw_record_stopped_failure(const QString &error);

private:
  using V = holoflow::core::GraphSpec::vertex_descriptor;

  // --- Initialization Helpers ---
  void register_components();

  // --- Polling Helpers ---
  void start_metrics_updates();
  void stop_metrics_updates();
  void poll_metrics();

  void start_event_polling();
  void stop_event_polling();
  void poll_events();

  // --- Graph Management ---
  void build_and_run();
  void build_graph_spec();
  void reset_graph_spec();
  void guess_optimizations();
  void guess_source_dims();

  // --- Logging Helpers ---
  void dump_graph_logs(const std::filesystem::path &log_dir);

  // --- UI Elements ---
  ui::AutoFocusWidget     *autofocus_widget_;
  ui::TensorDisplayWidget *xy_processed_widget_;
  ui::TensorDisplayWidget *xz_processed_widget_;
  ui::TensorDisplayWidget *yz_processed_widget_;
  ui::TensorDisplayWidget *xy_raw_widget_;
  ui::TensorDisplayWidget *raw_spectrum_widget_;
  ui::TensorDisplayWidget *processed_spectrum_widget_;
  ui::TensorDisplayWidget *shack_hartmann_widget_;
  ui::TensorDisplayWidget *shack_hartmann_xcorr_widget_;
  ui::TensorDisplayWidget *zernike_phase_widget_;

  // --- State & Configuration ---
  Settings s_;
  int      src_width_  = 0;
  int      src_height_ = 0;

  /// @brief Toggles debug dumps of the pipeline (.dot, .json) to disk.
  bool dump_debug_graphs_ = false;

  /// @brief Mutex to protect state shared between the main UI thread and pipeline callbacks.
  std::mutex mtx_;
  bool       settings_dirty_       = false;
  bool       raw_recording_active_ = false;

  // Optimizations
  bool opti_cpu_stride_ = false;
  bool opti_gpu_stride_ = false;

  // --- Holoflow Core Components ---
  holoflow::core::Registry                           registry_;
  holoflow::core::GraphSpec                          spec_;
  std::unique_ptr<holoflow::runtime::CompilerOutput> compiler_output_;
  std::unique_ptr<holoflow::runtime::Scheduler>      scheduler_;

  // Timers for regular UI updates
  QTimer *metrics_timer_ = nullptr;
  QTimer *events_timer_  = nullptr;
};

} // namespace holovibes::pipeline