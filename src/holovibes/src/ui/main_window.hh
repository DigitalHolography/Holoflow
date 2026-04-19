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

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QProgressBar>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QWidget>
#include <optional>

#include "pipeline/manager.hh"
#include "pipeline/settings.hh"
#include "pipeline/validation.hh"
#include "ui/widgets/export_widget.hh"
#include "ui/widgets/image_rendering_widget.hh"
#include "ui/widgets/import_widget.hh"
#include "ui/widgets/system_monitor_widget.hh"
#include "ui/widgets/view_widget.hh"

namespace holovibes::ui {

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow() override = default;

protected:
  void closeEvent(QCloseEvent *event) override;

private slots:
  void on_import_start_clicked();
  void on_import_stop_clicked();
  void on_metrics_updated(double input_fps);
  void on_export_record_clicked();
  void on_export_stop_clicked();
  void on_raw_record_started_success();
  void on_raw_record_started_failure(const QString &error);
  void on_raw_record_stopped_success();
  void on_raw_record_stopped_failure(const QString &error);

private:
  ImportWidget         *import_widget_;
  ExportWidget         *export_widget_;
  ImageRenderingWidget *render_widget_;
  ViewWidget           *view_widget_;
  SystemMonitorWidget  *monitor_widget_;

  void show_pipeline_error_popup(const QString &message);
  void on_start_pipeline_success();
  void on_start_pipeline_failure(const QString &error);
  void on_stop_pipeline_success();
  void on_stop_pipeline_failure(const QString &error);
  void on_update_pipeline_success();
  void on_update_pipeline_failure(const QString &error);

  bool                        validate_inputs();
  void                        apply_validation_result(const pipeline::ValidationResult &result);
  pipeline::ValidationContext build_validation_context(const pipeline::Settings &settings) const;
  void                        setup_validation_connections();
  void                        setup_update_connections();
  void                        update_if_running();

  QSize              guess_source_dims();
  pipeline::Settings get_pipeline_settings();
  void               set_pipeline_settings(const pipeline::Settings &s);

  void setup_menu_bar();
  void setup_main_layout();
  void initialize_display_widgets();
  void initialize_pipeline_manager();
  void connect_manager_signals();
  void connect_import_controls();
  void connect_export_controls();
  void configure_window();

  std::filesystem::path makeRecordingPath(const QString &userText) const;
  QStringList           load_available_camera_configs();
  std::string           get_selected_camera_config_path();

  // Current state
  bool update_in_progress_ = false;
  bool pipeline_running_   = false;
  bool export_in_progress_ = false;

  // Workers
  pipeline::Manager *pipeline_manager_;
  QThread           *pipeline_manager_thread_;

  // Display widgets
  TensorDisplayWidget *xy_processed_widget_;
  TensorDisplayWidget *xz_processed_widget_;
  TensorDisplayWidget *yz_processed_widget_;
  TensorDisplayWidget *xy_raw_widget_;
  TensorDisplayWidget *raw_spectrum_widget_;
  TensorDisplayWidget *processed_spectrum_widget_;
  TensorDisplayWidget *shack_hartmann_widget_;
  TensorDisplayWidget *shack_hartmann_xcorr_widget_;
  TensorDisplayWidget *zernike_phase_widget_;
};

} // namespace holovibes::ui
