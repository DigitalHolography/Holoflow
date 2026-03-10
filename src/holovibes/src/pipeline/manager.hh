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

  void start_pipeline();
  void stop_pipeline();
  void update_pipeline(const Settings &settings);

  void start_raw_record(std::filesystem::path record_path);
  void stop_raw_record();

signals:
  void start_pipeline_success();
  void start_pipeline_failure(const QString &error);
  void stop_pipeline_success();
  void stop_pipeline_failure(const QString &error);
  void update_pipeline_success();
  void update_pipeline_failure(const QString &error);
  void metrics_updated(double input_fps);
  void raw_record_started_success();
  void raw_record_started_failure(const QString &error);
  void raw_record_stopped_success();
  void raw_record_stopped_failure(const QString &error);

private:
  using V = holoflow::core::GraphSpec::vertex_descriptor;

  void start_metrics_updates();
  void stop_metrics_updates();
  void poll_metrics();

  void start_event_polling();
  void stop_event_polling();
  void poll_events();

  void build_and_run();
  void build_graph_spec();
  void reset_graph_spec();
  void guess_optimizations();
  void guess_source_dims();

  ui::AutoFocusWidget *autofocus_widget_;

  // External widgets to display tensors
  ui::TensorDisplayWidget *xy_processed_widget_;
  ui::TensorDisplayWidget *xz_processed_widget_;
  ui::TensorDisplayWidget *yz_processed_widget_;
  ui::TensorDisplayWidget *xy_raw_widget_;
  ui::TensorDisplayWidget *raw_spectrum_widget_;
  ui::TensorDisplayWidget *processed_spectrum_widget_;
  ui::TensorDisplayWidget *shack_hartmann_widget_;
  ui::TensorDisplayWidget *shack_hartmann_xcorr_widget_;
  ui::TensorDisplayWidget *zernike_phase_widget_;

  // Settings
  Settings s_;
  int      src_width_  = 0;
  int      src_height_ = 0;

  // Internal state
  std::mutex mtx_;
  bool       settings_dirty_       = false;
  bool       raw_recording_active_ = false;

  // Optimizations
  bool opti_cpu_stride_ = false;
  bool opti_gpu_stride_ = false;

  // Holoflow components
  holoflow::core::Registry                           registry_;
  holoflow::core::GraphSpec                          spec_;
  std::unique_ptr<holoflow::runtime::CompilerOutput> compiler_output_;
  std::unique_ptr<holoflow::runtime::Scheduler>      scheduler_;
  QTimer                                            *metrics_timer_ = nullptr;
  QTimer                                            *events_timer_  = nullptr;
};

} // namespace holovibes::pipeline
