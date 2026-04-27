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
#include <array>
#include <optional>

#include "pipeline/field_help.hh"
#include "pipeline/manager.hh"
#include "pipeline/settings.hh"
#include "pipeline/validation.hh"
#include "ui/widgets/export_widget.hh"
#include "ui/widgets/image_rendering_widget.hh"
#include "ui/widgets/import_widget.hh"
#include "ui/widgets/system_monitor_widget.hh"
#include "ui/widgets/view_widget.hh"

class QSplitter;
class QGridLayout;

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
  enum class DisplayPanelZone { Main, Secondary };

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
  void                        refresh_validation_tooltips(const pipeline::ValidationResult &result);
  pipeline::ValidationContext build_validation_context(const pipeline::Settings &settings) const;
  void                        setup_validation_connections();
  void                        setup_update_connections();
  void                        update_if_running();

  QSize              guess_source_dims();
  pipeline::Settings get_pipeline_settings();
  void               set_pipeline_settings(const pipeline::Settings &s);

  void setup_main_layout();
  void initialize_display_widgets();
  void initialize_pipeline_manager();
  void connect_manager_signals();
  void connect_import_controls();
  void connect_export_controls();
  void configure_window();

  QGroupBox                 *create_display_panel(const QString &title, const QString &display_id,
                                                  TensorDisplayWidget *widget);
  std::array<QGroupBox *, 9> display_panels() const;
  QGroupBox                 *display_panel_for(TensorDisplayWidget *widget) const;
  QGroupBox                 *display_panel_for_id(const QString &display_id) const;
  void                       set_display_title(TensorDisplayWidget *widget, const QString &title);
  void                       set_display_visible(TensorDisplayWidget *widget, bool visible);
  void                       begin_display_layout_update();
  void                       end_display_layout_update();
  void                       move_display_panel(const QString &display_id, DisplayPanelZone zone);
  void                       set_display_panel_zone(QGroupBox *panel, DisplayPanelZone zone);
  bool                       is_display_panel_in_main(QGroupBox *panel) const;
  QStringList                main_display_panel_ids() const;
  void                       relayout_display_panels();
  void                       refresh_secondary_display_visibility();
  void                       configure_unsupported_features();
  void                       save_persistent_state();
  void                       restore_persistent_state();
  QString                    sanitize_recording_token(const QString &value) const;
  QString                    recording_file_name(int acquisition_id) const;
  QString                    acquisition_label(int acquisition_id) const;
  void                       update_acquisition_label();
  void                       update_recording_path_preview();
  void                       refresh_command_bar();
  void                       set_status_label(QLabel *label, const QString &text, const char *tone);

  std::filesystem::path makeRecordingPath(const QString &userText);
  QStringList           load_available_camera_configs();
  std::string           get_selected_camera_config_path();

  // Current state
  bool               update_in_progress_          = false;
  bool               pipeline_running_            = false;
  bool               export_in_progress_          = false;
  bool               geometry_restored_           = false;
  int                display_layout_update_depth_ = 0;
  bool               display_layout_dirty_        = false;
  QString            session_id_;
  int                next_acquisition_id_ = 1;
  std::optional<int> pending_recording_acquisition_id_;

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

  // Embedded display panels
  QSplitter   *main_splitter_               = nullptr;
  QLineEdit   *patient_line_edit_           = nullptr;
  QComboBox   *eye_side_combo_              = nullptr;
  QLabel      *session_value_label_         = nullptr;
  QLabel      *acquisition_value_label_     = nullptr;
  QPushButton *start_command_button_        = nullptr;
  QPushButton *stop_command_button_         = nullptr;
  QPushButton *record_command_button_       = nullptr;
  QPushButton *stop_record_command_button_  = nullptr;
  QLabel      *pipeline_status_label_       = nullptr;
  QLabel      *source_status_label_         = nullptr;
  QLabel      *view_status_label_           = nullptr;
  QLabel      *fps_status_label_            = nullptr;
  QLabel      *recording_status_label_      = nullptr;
  QWidget     *display_workspace_           = nullptr;
  QWidget     *main_display_container_      = nullptr;
  QWidget     *secondary_display_container_ = nullptr;
  QGridLayout *main_display_layout_         = nullptr;
  QGridLayout *secondary_display_layout_    = nullptr;
  QGroupBox   *xy_processed_panel_          = nullptr;
  QGroupBox   *xz_processed_panel_          = nullptr;
  QGroupBox   *yz_processed_panel_          = nullptr;
  QGroupBox   *xy_raw_panel_                = nullptr;
  QGroupBox   *raw_spectrum_panel_          = nullptr;
  QGroupBox   *processed_spectrum_panel_    = nullptr;
  QGroupBox   *shack_hartmann_panel_        = nullptr;
  QGroupBox   *shack_hartmann_xcorr_panel_  = nullptr;
  QGroupBox   *zernike_phase_panel_         = nullptr;
};

} // namespace holovibes::ui
