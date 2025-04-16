#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QLineEdit>
#include <QMainWindow>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QWidget>

#include "holovibes/pipeline/settings.hh"
#include "holovibes/pipeline/worker.hh"
#include "holovibes/ui/tensor_display_widget.hh"

namespace holovibes::ui {

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow() override = default;

private slots:
  void on_import_start_clicked();
  void on_import_stop_clicked();

private:
  bool validate_inputs();

  void setup_validation_connections();

  holovibes::pipeline::Settings get_pipeline_settings();

  // Workers
  pipeline::Worker *pipeline_worker_;
  QThread *pipeline_worker_thread_;

  // Display widgets
  dh::TensorDisplayWidget *display_widget_;

  // Helper methods to create UI sections
  QGroupBox *create_import_group();
  QGroupBox *create_export_group();
  QGroupBox *create_image_rendering_group();
  QGroupBox *create_view_group();

  // --- UI Members for Import Group ---
  QLineEdit *import_file_line_edit_;
  QPushButton *import_browse_button_;
  QSpinBox *import_fps_spin_;
  QSpinBox *import_start_index_spin_;
  QSpinBox *import_end_index_spin_;
  QComboBox *import_load_method_combo_;
  QPushButton *import_start_button_;
  QPushButton *import_stop_button_;

  // --- UI Members for Export Group ---
  QComboBox *export_image_type_combo_;
  QLineEdit *export_file_line_edit_;
  QPushButton *export_browse_button_;
  QComboBox *export_tag_combo_;
  QCheckBox *export_frames_check_;
  QSpinBox *export_frames_spin_;
  QPushButton *export_record_button_;
  QPushButton *export_stop_button_;
  QPushButton *export_stop_fan_button_;

  // --- UI Members for Image Rendering Group ---
  QComboBox *render_image_combo_;
  QSpinBox *render_batch_size_spin_;
  QSpinBox *render_time_stride_spin_;
  QCheckBox *render_filter_2d_check_;
  QComboBox *render_space_transform_combo_;
  QComboBox *render_time_transform_combo_;
  QSpinBox *render_time_window_spin_;
  QSpinBox *render_lambda_spin_;
  QSpinBox *render_boundary_spin_;
  QSpinBox *render_focus_spin_;
  QSlider *render_focus_slider_;
  QComboBox *render_convolution_combo_;
  QCheckBox *render_convolution_divide_check_;

  // --- UI Members for View Group ---
  QComboBox *view_image_type_combo_;
  QCheckBox *view_cuts_3d_check_;
  QCheckBox *view_fft_shift_check_;
  QCheckBox *view_lens_view_check_;
  QCheckBox *view_raw_view_check_;
  QSpinBox *view_z_spin_;
  QSpinBox *view_width_spin_;
  QComboBox *view_kind_combo_;
  QSpinBox *view_accumulation_spin_;
  QCheckBox *view_auto_check_;
  QCheckBox *view_invert_check_;
  QSpinBox *view_range_start_spin_;
  QSpinBox *view_range_end_spin_;
  QCheckBox *view_renormalize_check_;
};

} // namespace holovibes::ui