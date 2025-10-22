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
  void show_pipeline_error_popup(const QString &message);
  void on_start_pipeline_success();
  void on_start_pipeline_failure(const QString &error);
  void on_stop_pipeline_success();
  void on_stop_pipeline_failure(const QString &error);
  void on_update_pipeline_success();
  void on_update_pipeline_failure(const QString &error);

  bool validate_inputs();
  void setup_validation_connections();
  void setup_update_connections();
  void update_if_running();

  QSize              guess_source_dims();
  pipeline::Settings get_pipeline_settings();

  void setup_menu_bar();
  void setup_main_layout();
  void initialize_display_widgets();
  void initialize_pipeline_manager();
  void connect_manager_signals();
  void connect_import_controls();
  void configure_window();

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

  // Helper methods to create UI sections
  QGroupBox *create_import_group();
  QGroupBox *create_export_group();
  QGroupBox *create_image_rendering_group();
  QGroupBox *create_view_group();
  QGroupBox *create_system_monitor_group();

  // UI Members for Import Group
  QCheckBox   *import_cam_check_;
  QLineEdit   *import_file_line_edit_;
  QPushButton *import_browse_button_;
  QSpinBox    *import_fps_spin_;
  QSpinBox    *import_start_index_spin_;
  QSpinBox    *import_end_index_spin_;
  QComboBox   *import_load_method_combo_;
  QComboBox   *import_camera_combo_;
  QLineEdit   *import_cam_config_line_edit_;
  QPushButton *import_cam_config_browse_button_;
  QPushButton *import_start_button_;
  QPushButton *import_stop_button_;

  // UI Members for Export Group
  QComboBox   *export_image_type_combo_;
  QLineEdit   *export_file_line_edit_;
  QPushButton *export_browse_button_;
  QComboBox   *export_tag_combo_;
  QCheckBox   *export_frames_check_;
  QSpinBox    *export_frames_spin_;
  QPushButton *export_record_button_;
  QPushButton *export_stop_button_;
  QPushButton *export_stop_fan_button_;

  // UI Members for Image Rendering Group
  QComboBox *render_image_combo_;
  QSpinBox  *render_batch_size_spin_;
  QSpinBox  *render_time_stride_spin_;
  QCheckBox *render_filter_2d_check_;
  QSpinBox  *render_filter_2d_inner_spin_;
  QSpinBox  *render_filter_2d_outer_spin_;
  QComboBox *render_space_transform_combo_;
  QComboBox *render_time_transform_combo_;
  QSpinBox  *render_time_window_spin_;
  QSpinBox  *render_lambda_spin_;
  QSpinBox  *render_boundary_spin_;
  QSpinBox  *render_focus_spin_;
  QSlider   *render_focus_slider_;
  QComboBox *render_convolution_combo_;
  QCheckBox *render_convolution_divide_check_;

  // UI Members for View Group
  QComboBox      *view_image_type_combo_;
  QCheckBox      *view_cuts_3d_check_;
  QCheckBox      *view_fft_shift_check_;
  QCheckBox      *view_lens_view_check_;
  QCheckBox      *view_raw_view_check_;
  QSpinBox       *view_x_spin_;
  QSpinBox       *view_x_width_spin_;
  QSpinBox       *view_y_spin_;
  QSpinBox       *view_y_width_spin_;
  QSpinBox       *view_z_spin_;
  QSpinBox       *view_z_width_spin_;
  QComboBox      *view_kind_combo_;
  QSpinBox       *view_accumulation_spin_;
  QCheckBox      *view_auto_check_;
  QCheckBox      *view_invert_check_;
  QSpinBox       *view_range_start_spin_;
  QSpinBox       *view_range_end_spin_;
  QCheckBox      *view_renormalize_check_;
  QCheckBox      *view_registration_check_;
  QDoubleSpinBox *view_registration_radius_;
  QCheckBox      *view_reticle_check_;
  QDoubleSpinBox *view_reticle_radius_;
  // UI Members for System Monitor
  QLabel       *metrics_gpu_load_value_;
  QLabel       *metrics_cpu_load_value_;
  QLabel       *metrics_input_throughput_fps_value_;
  QLabel       *metrics_input_throughput_bytes_value_;
  QLabel       *metrics_cpu_throughput_value_;
  QLabel       *metrics_gpu_throughput_value_;
  QLabel       *metrics_ram_usage_value_;
  QLabel       *metrics_vram_usage_value_;
  QLabel       *metrics_dropped_frames_value_;
  QLabel       *metrics_pipeline_latency_value_;
  QProgressBar *metrics_input_queue_bar_;
  QProgressBar *metrics_output_queue_bar_;
  QProgressBar *metrics_record_queue_bar_;
};

} // namespace holovibes::ui
