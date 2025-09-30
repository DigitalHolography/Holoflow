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

#include "ui/main_window.hh"

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDir>
#include <QDirIterator>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfoList>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenuBar>
#include <QProgressBar>
#include <QPushButton>
#include <QSlider>
#include <QSpacerItem>
#include <QSpinBox>
#include <QStackedLayout>
#include <QStandardPaths>
#include <QStringList>
#include <QThread>
#include <QVBoxLayout>

#include "bug.hh"
#include "holofile/holofile.hh"
#include "logger.hh"

namespace holovibes::ui {

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
  // Create menus
  menuBar()->addMenu("&File");
  menuBar()->addMenu("&View");
  menuBar()->addMenu("&Camera");
  menuBar()->addMenu("&Theme");

  // Settings
  auto *central_widget = new QWidget(this);
  setCentralWidget(central_widget);
  auto *main_layout = new QHBoxLayout(central_widget);

  auto *left_panel_layout = new QHBoxLayout();
  left_panel_layout->addWidget(create_image_rendering_group());
  left_panel_layout->addWidget(create_view_group());
  auto *import_export_layout = new QVBoxLayout();
  import_export_layout->addWidget(create_import_group());
  import_export_layout->addWidget(create_export_group());
  left_panel_layout->addLayout(import_export_layout);
  main_layout->addLayout(left_panel_layout);
  main_layout->addWidget(create_system_monitor_group(), 0, Qt::AlignTop);

  // Display widgets
  xy_raw_widget_       = new TensorDisplayWidget(nullptr);
  xy_processed_widget_ = new TensorDisplayWidget(nullptr);
  xz_processed_widget_ = new TensorDisplayWidget(nullptr);
  yz_processed_widget_ = new TensorDisplayWidget(nullptr);
  xy_processed_widget_->setWindowTitle("XY-Processed");
  xz_processed_widget_->setWindowTitle("XZ-Processed");
  yz_processed_widget_->setWindowTitle("YZ-Processed");
  xy_raw_widget_->setWindowTitle("XY-Raw");
  xy_processed_widget_->resize(400, 400);
  xy_processed_widget_->show();

  pipeline_manager_        = new pipeline::Manager(xy_processed_widget_, xz_processed_widget_,
                                                   yz_processed_widget_, xy_raw_widget_);
  pipeline_manager_thread_ = new QThread(this);
  pipeline_manager_->moveToThread(pipeline_manager_thread_);

  setup_validation_connections();
  setup_update_connections();

  // TODO: display reticle
  // connect(view_reticle_check_, &QCheckBox::toggled, xy_processed_display_widget_,
  //         &TensorDisplayWidget::set_display_reticle);

  // connect(view_reticle_radius_, qOverload<double>(&QDoubleSpinBox::valueChanged),
  //         xy_processed_display_widget_, &TensorDisplayWidget::set_reticle_radius);

  connect(pipeline_manager_, &pipeline::Manager::start_pipeline_success, this,
          &MainWindow::on_start_pipeline_success);

  connect(pipeline_manager_, &pipeline::Manager::start_pipeline_failure, this,
          &MainWindow::on_start_pipeline_failure);

  connect(pipeline_manager_, &pipeline::Manager::stop_pipeline_success, this,
          &MainWindow::on_stop_pipeline_success);

  connect(pipeline_manager_, &pipeline::Manager::stop_pipeline_failure, this,
          &MainWindow::on_stop_pipeline_failure);

  connect(pipeline_manager_, &pipeline::Manager::update_pipeline_success, this,
          &MainWindow::on_update_pipeline_success);

  connect(pipeline_manager_, &pipeline::Manager::update_pipeline_failure, this,
          &MainWindow::on_update_pipeline_failure);

  connect(pipeline_manager_, &pipeline::Manager::metrics_updated, this,
          &MainWindow::on_metrics_updated, Qt::QueuedConnection);

  pipeline_manager_thread_->start();

  connect(import_start_button_, &QPushButton::clicked, this, &MainWindow::on_import_start_clicked);
  connect(import_stop_button_, &QPushButton::clicked, this, &MainWindow::on_import_stop_clicked);

  validate_inputs();
  setWindowTitle("Holovibes");
  this->setFixedSize(this->minimumSizeHint());
  // this->adjustSize();
  this->show();
}

void MainWindow::on_start_pipeline_success() {
  logger()->info("[MainWindow::on_start_pipeline_success]");
  pipeline_running_ = true;
  import_stop_button_->setEnabled(true);
}

void MainWindow::on_start_pipeline_failure() {
  logger()->error("[MainWindow::on_start_pipeline_failure]");
  pipeline_running_ = false;
  import_start_button_->setEnabled(true);
  import_stop_button_->setEnabled(false);
}

void MainWindow::on_stop_pipeline_success() {
  logger()->info("[MainWindow::on_stop_pipeline_success]");
  pipeline_running_ = false;
  import_start_button_->setEnabled(true);
  import_stop_button_->setEnabled(false);
  on_metrics_updated(0.0);
}

void MainWindow::on_stop_pipeline_failure() {
  logger()->error("[MainWindow::on_stop_pipeline_failure]");
  pipeline_running_ = false;
  import_start_button_->setEnabled(true);
  import_stop_button_->setEnabled(false);
  on_metrics_updated(0.0);
}

void MainWindow::on_metrics_updated(double input_fps) {
  if (!metrics_input_throughput_fps_value_) {
    return;
  }

  if (input_fps < 0.0) {
    input_fps = 0.0;
  }

  const QString text = QStringLiteral("%1 fps").arg(QString::number(input_fps, 'f', 1));
  metrics_input_throughput_fps_value_->setText(text);
}

void MainWindow::on_update_pipeline_success() {
  logger()->info("[MainWindow::on_update_pipeline_success]");
  update_in_progress_ = false;
  import_stop_button_->setEnabled(true);
}

void MainWindow::on_update_pipeline_failure() {
  logger()->error("[MainWindow::on_update_pipeline_failure]");
  pipeline_running_   = false;
  update_in_progress_ = false;
  import_start_button_->setEnabled(true);
  import_stop_button_->setEnabled(false);
}

void MainWindow::closeEvent(QCloseEvent *event) {
  if (pipeline_manager_ && import_stop_button_->isEnabled()) {
    pipeline_manager_->stop_pipeline();
  }

  if (pipeline_manager_thread_) {
    pipeline_manager_thread_->quit();
    pipeline_manager_thread_->wait();
  }

  QMainWindow::closeEvent(event);
  QCoreApplication::quit();
}

void MainWindow::on_import_start_clicked() {
  HOLOVIBES_CHECK(!pipeline_running_);

  if (!validate_inputs()) {
    return;
  }

  import_start_button_->setEnabled(false);
  pipeline::Settings settings = get_pipeline_settings();
  auto               update   = [=]() { pipeline_manager_->update_pipeline(settings); };
  auto               start    = [=]() { pipeline_manager_->start_pipeline(); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, update, Qt::QueuedConnection));
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, start, Qt::QueuedConnection));
}

void MainWindow::on_import_stop_clicked() {
  HOLOVIBES_CHECK(pipeline_running_);
  import_stop_button_->setEnabled(false);
  auto stop = [=]() { pipeline_manager_->stop_pipeline(); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, stop, Qt::QueuedConnection));
}

bool MainWindow::validate_inputs() {
  const QString style_fail = QStringLiteral("background-color: rgba(255, 0, 0, 50);");

  // Clear all styles up front
  const std::initializer_list<QWidget *> all_widgets = {
      render_batch_size_spin_,  render_time_stride_spin_,
      render_time_window_spin_, view_z_spin_,
      view_z_width_spin_,       import_start_index_spin_,
      import_end_index_spin_,   import_file_line_edit_,
      export_frames_spin_,      import_cam_config_line_edit_};

  for (auto *w : all_widgets) {
    w->setStyleSheet("");
  }

  bool all_good = true;

  // helper: if cond==false, mark widgets and flag failure
  auto mark_failure = [&](bool cond, std::initializer_list<QWidget *> widgets) {
    if (!cond) {
      all_good = false;
      for (auto *w : widgets) {
        w->setStyleSheet(style_fail);
      }
    }
  };

  int  batch_size    = render_batch_size_spin_->value();
  int  time_stride   = render_time_stride_spin_->value();
  int  time_window   = render_time_window_spin_->value();
  int  p_frame_start = view_z_spin_->value();
  int  p_frame_width = view_z_width_spin_->value();
  int  start_frame   = import_start_index_spin_->value();
  int  end_frame     = import_end_index_spin_->value();
  bool is_cam_mode   = import_cam_check_->isChecked();

  // time_window divides time_stride
  mark_failure(time_window > 0 && (time_stride % time_window == 0),
               {render_time_window_spin_, render_time_stride_spin_});

  // p_frame_start + p_frame_width <= time_window
  mark_failure(p_frame_start + p_frame_width <= time_window,
               {render_time_window_spin_, view_z_spin_, view_z_width_spin_});

  // end_frame > start_frame
  mark_failure(is_cam_mode || end_frame > start_frame,
               {import_start_index_spin_, import_end_index_spin_});

  // time_stride <= end_frame - start_frame
  mark_failure(is_cam_mode || time_stride <= (end_frame - start_frame),
               {import_start_index_spin_, import_end_index_spin_, render_time_stride_spin_});

  // import path non-empty
  if (!is_cam_mode) {
    bool hasPath = !import_file_line_edit_->text().isEmpty();
    mark_failure(hasPath, {import_file_line_edit_});
  } else {
    bool hasCamConfig = !import_cam_config_line_edit_->text().isEmpty();
    mark_failure(hasCamConfig, {import_cam_config_line_edit_});
  }

  // if exporting frames, ensure divisible by batch_size
  if (export_frames_check_->isChecked()) {
    int exportCount = export_frames_spin_->value();
    mark_failure((exportCount % batch_size) == 0, {export_frames_spin_, render_batch_size_spin_});
  }

  return all_good;
}

void MainWindow::setup_validation_connections() {
  bool (MainWindow::*cb)() = &MainWindow::validate_inputs;

  // Import Group Connections
  connect(import_file_line_edit_, &QLineEdit::editingFinished, this, cb);
  connect(import_browse_button_, &QPushButton::clicked, this, cb);
  connect(import_start_button_, &QPushButton::clicked, this, cb);
  connect(import_stop_button_, &QPushButton::clicked, this, cb);
  connect(import_fps_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(import_start_index_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(import_end_index_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(import_load_method_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(import_cam_check_, &QCheckBox::toggled, this, &MainWindow::validate_inputs);
  connect(import_camera_combo_, &QComboBox::currentIndexChanged, this, cb);
  connect(import_cam_config_line_edit_, &QLineEdit::editingFinished, this, cb);
  connect(import_cam_config_browse_button_, &QPushButton::clicked, this, cb);

  // Export Group Connections
  connect(export_image_type_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(export_file_line_edit_, &QLineEdit::editingFinished, this, cb);
  connect(export_browse_button_, &QPushButton::clicked, this, cb);
  connect(export_tag_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(export_frames_check_, &QCheckBox::toggled, this, cb);
  connect(export_frames_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(export_record_button_, &QPushButton::clicked, this, cb);
  connect(export_stop_button_, &QPushButton::clicked, this, cb);
  connect(export_stop_fan_button_, &QPushButton::clicked, this, cb);

  // Image Rendering Group Connections
  connect(render_image_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(render_batch_size_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_time_stride_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_filter_2d_check_, &QCheckBox::toggled, this, cb);
  connect(render_space_transform_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(render_time_transform_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(render_time_window_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_lambda_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_boundary_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_focus_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_focus_slider_, &QSlider::valueChanged, this, cb);
  connect(render_convolution_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(render_convolution_divide_check_, &QCheckBox::toggled, this, cb);

  // View Group Connections
  connect(view_image_type_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(view_cuts_3d_check_, &QCheckBox::toggled, this, cb);
  connect(view_fft_shift_check_, &QCheckBox::toggled, this, cb);
  connect(view_lens_view_check_, &QCheckBox::toggled, this, cb);
  connect(view_raw_view_check_, &QCheckBox::toggled, this, cb);
  connect(view_x_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_x_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_y_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_y_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_z_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_z_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_kind_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(view_accumulation_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_auto_check_, &QCheckBox::toggled, this, cb);
  connect(view_invert_check_, &QCheckBox::toggled, this, cb);
  connect(view_range_start_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_range_end_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_renormalize_check_, &QCheckBox::toggled, this, cb);
  connect(view_registration_check_, &QCheckBox::toggled, this, cb);
  connect(view_registration_radius_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, cb);
  connect(view_reticle_check_, &QCheckBox::toggled, this, cb);
  connect(view_reticle_radius_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, cb);
}

void MainWindow::setup_update_connections() {
  void (MainWindow::*cb)() = &MainWindow::update_if_running;

  // Import Group Connections
  connect(import_file_line_edit_, &QLineEdit::editingFinished, this, cb);
  connect(import_browse_button_, &QPushButton::clicked, this, cb);
  connect(import_fps_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(import_start_index_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(import_end_index_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(import_load_method_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(import_cam_check_, &QCheckBox::toggled, this, cb);
  connect(import_camera_combo_, &QComboBox::currentIndexChanged, this, cb);
  connect(import_cam_config_line_edit_, &QLineEdit::editingFinished, this, cb);
  connect(import_cam_config_browse_button_, &QPushButton::clicked, this, cb);

  // Export Group Connections
  connect(export_image_type_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(export_file_line_edit_, &QLineEdit::editingFinished, this, cb);
  connect(export_browse_button_, &QPushButton::clicked, this, cb);
  connect(export_tag_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(export_frames_check_, &QCheckBox::toggled, this, cb);
  connect(export_frames_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(export_stop_fan_button_, &QPushButton::clicked, this, cb);

  // Image Rendering Group Connections
  connect(render_image_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(render_batch_size_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_time_stride_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_filter_2d_check_, &QCheckBox::toggled, this, cb);
  connect(render_filter_2d_inner_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_filter_2d_outer_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_space_transform_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(render_time_transform_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(render_time_window_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_lambda_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_boundary_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_focus_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(render_focus_slider_, &QSlider::valueChanged, this, cb);
  connect(render_convolution_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(render_convolution_divide_check_, &QCheckBox::toggled, this, cb);

  // View Group Connections
  connect(view_image_type_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(view_cuts_3d_check_, &QCheckBox::toggled, this, cb);
  connect(view_fft_shift_check_, &QCheckBox::toggled, this, cb);
  connect(view_lens_view_check_, &QCheckBox::toggled, this, cb);
  connect(view_raw_view_check_, &QCheckBox::toggled, this, cb);
  connect(view_x_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_x_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_y_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_y_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_z_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_z_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_kind_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, cb);
  connect(view_accumulation_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_auto_check_, &QCheckBox::toggled, this, cb);
  connect(view_invert_check_, &QCheckBox::toggled, this, cb);
  connect(view_range_start_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_range_end_spin_, qOverload<int>(&QSpinBox::valueChanged), this, cb);
  connect(view_renormalize_check_, &QCheckBox::toggled, this, cb);
  connect(view_registration_check_, &QCheckBox::toggled, this, cb);
  connect(view_registration_radius_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, cb);
  connect(view_reticle_check_, &QCheckBox::toggled, this, cb);
  connect(view_reticle_radius_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, cb);
}

void MainWindow::update_if_running() {
  if (!pipeline_manager_ || !import_stop_button_->isEnabled()) {
    return;
  }

  if (update_in_progress_) {
    return;
  }

  if (!validate_inputs()) {
    return;
  }

  update_in_progress_ = true;
  import_start_button_->setEnabled(false);
  pipeline::Settings settings = get_pipeline_settings();
  auto               update   = [=]() { pipeline_manager_->update_pipeline(settings); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, update, Qt::QueuedConnection));
}

pipeline::Settings MainWindow::get_pipeline_settings() {
  using namespace holovibes::pipeline;

  Settings s;

  // Advanced Settings
  {
    s.cpu_in_size  = 4096;
    s.gpu_in_size  = 4096;
    s.cpu_rec_size = 4096;
    s.cpu_out_size = 32;
    s.gpu_out_size = 32;
  }

  // Import Settings
  {
    std::map<std::string, LoadMethod> method_from_str{
        {"Read Live", LoadMethod::READ_LIVE},
        {"Load in CPU RAM", LoadMethod::LOAD_IN_CPU},
        {"Load in GPU RAM", LoadMethod::LOAD_IN_GPU},
    };
    std::map<std::string, ImportSource> source_from_str{
        {"Ametek S710 Euresys Coaxlink Octo", ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO},
    };

    if (!import_cam_check_->isChecked()) {
      s.import_source = ImportSource::HOLOFILE;
      s.load_path     = import_file_line_edit_->text().toStdString();
      s.load_begin    = static_cast<size_t>(import_start_index_spin_->value());
      s.load_end      = static_cast<size_t>(import_end_index_spin_->value());
      QString method  = import_load_method_combo_->currentText();
      s.load_method   = method_from_str.at(method.toStdString());
      s.load_batch    = 32;
    } else {
      QString source       = import_camera_combo_->currentText();
      s.import_source      = source_from_str.at(source.toStdString());
      s.camera_config_path = import_cam_config_line_edit_->text().toStdString();
    }
  }

  // Image Rendering Settings
  {
    std::map<std::string, SpacialMethod> method_from_str{
        {"None", SpacialMethod::NONE},
        {"Fresnel Diffraction", SpacialMethod::FRESNEL_DIFFRACTION},
        {"Angular Spectrum", SpacialMethod::ANGULAR_SPECTRUM},
    };
    QString method       = render_space_transform_combo_->currentText();
    s.spacial_method     = method_from_str.at(method.toStdString());
    s.spacial_lambda     = static_cast<float>(render_lambda_spin_->value()) * 1e-9f;
    s.spacial_z          = static_cast<float>(render_focus_spin_->value()) * 1e-3f;
    s.spacial_pixel_size = 20e-6f; // TODO: get from camera
  }
  {
    s.filter_2d           = render_filter_2d_check_->isChecked();
    s.filter_r_inner      = render_filter_2d_inner_spin_->value();
    s.filter_r_outer      = render_filter_2d_outer_spin_->value();
    s.filter_smooth_inner = 0;
    s.filter_smooth_outer = 1;
  }
  {
    std::map<std::string, TimeMethod> method_from_str{
        {"None", TimeMethod::NONE},
        {"Principal Component Analysis", TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS},
        {"Short Time Fourier", TimeMethod::SHORT_TIME_FOURIER},
    };
    s.time_window  = render_time_window_spin_->value();
    s.time_stride  = render_time_stride_spin_->value();
    QString method = render_time_transform_combo_->currentText();
    s.time_method  = method_from_str.at(method.toStdString());
    s.time_x_begin = view_x_spin_->value();
    s.time_x_end   = s.time_x_begin + view_x_width_spin_->value();
    s.time_y_begin = view_y_spin_->value();
    s.time_y_end   = s.time_y_begin + view_y_width_spin_->value();
    s.time_z_begin = view_z_spin_->value();
    s.time_z_end   = s.time_z_begin + view_z_width_spin_->value();
  }

  // View Settings
  { s.view_3d_cuts = view_cuts_3d_check_->isChecked(); }

  // Post-processing Settings
  {
    // TODO: load from file
    QString     appDataBase = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QString     appDataPath = appDataBase + "/" + QCoreApplication::applicationVersion();
    QString     convolutionsKernelsPath = appDataPath + "/" + "convolution_kernels/";
    std::string kernel_path             = convolutionsKernelsPath.toStdString() +
                              render_convolution_combo_->currentText().toStdString() + ".json";

    s.pp_fps                 = 60;
    s.pp_fft_shift           = view_fft_shift_check_->isChecked();
    s.pp_accumulation        = static_cast<size_t>(view_accumulation_spin_->value());
    s.pp_convolution         = render_convolution_combo_->currentText() != "None";
    s.pp_convolution_path    = kernel_path;
    s.pp_convolution_divide  = render_convolution_divide_check_->isChecked();
    s.pp_pctclip             = true;
    s.pp_pctclip_lower       = 0.02f;
    s.pp_pctclip_upper       = 99.98f;
    s.pp_pctclip_radius      = view_reticle_radius_->value();
    s.pp_registration        = view_registration_check_->isChecked();
    s.pp_registration_radius = view_registration_radius_->value();
  }

  return s;
}

QGroupBox *MainWindow::create_system_monitor_group() {
  auto *group  = new QGroupBox("System Monitor", this);
  auto *layout = new QVBoxLayout(group);

  auto *metrics_layout = new QGridLayout();
  metrics_layout->setColumnStretch(1, 1);

  auto add_metric_row = [&](int row, const QString &label, QLabel **value_label,
                            const QString &value) {
    metrics_layout->addWidget(new QLabel(label, group), row, 0);
    *value_label = new QLabel(value, group);
    (*value_label)->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    metrics_layout->addWidget(*value_label, row, 1);
  };

  // clang-format off
  add_metric_row(0, "GPU Load:", &metrics_gpu_load_value_, "68 %");
  add_metric_row(1, "CPU Load:", &metrics_cpu_load_value_, "42 %");
  add_metric_row(2, "Input Throughput (FPS):", &metrics_input_throughput_fps_value_, "0.0 fps");
  add_metric_row(3, "Input Throughput (Bytes):", &metrics_input_throughput_bytes_value_, "1.2 GB/s");
  add_metric_row(4, "CPU Throughput:", &metrics_cpu_throughput_value_, "3.4 GB/s");
  add_metric_row(5, "GPU Throughput:", &metrics_gpu_throughput_value_, "5.1 GB/s");
  add_metric_row(6, "RAM Usage:", &metrics_ram_usage_value_, "12.3 / 32 GB");
  add_metric_row(7, "VRAM Usage:", &metrics_vram_usage_value_, "6.5 / 12 GB");
  add_metric_row(8, "Dropped Frames:", &metrics_dropped_frames_value_, "2");
  add_metric_row(9, "Pipeline Latency:", &metrics_pipeline_latency_value_, "16 ms");
  // clang-format on

  layout->addLayout(metrics_layout);

  auto *queue_group   = new QGroupBox("Queues", group);
  auto *queue_layout  = new QVBoxLayout(queue_group);
  auto  configure_bar = [&](QProgressBar **bar, const QString &title, int value, int maximum) {
    queue_layout->addWidget(new QLabel(title, queue_group));
    *bar = new QProgressBar(queue_group);
    (*bar)->setRange(0, maximum);
    (*bar)->setValue(value);
    (*bar)->setFormat("%v / %m");
    (*bar)->setTextVisible(true);
    queue_layout->addWidget(*bar);
  };

  configure_bar(&metrics_input_queue_bar_, "Input Queue", 48, 64);
  configure_bar(&metrics_output_queue_bar_, "Output Queue", 22, 64);
  configure_bar(&metrics_record_queue_bar_, "Record Queue", 12, 32);

  layout->addWidget(queue_group);
  layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding));
  layout->addStretch(1);

  return group;
}

QGroupBox *MainWindow::create_import_group() {
  QGroupBox *group      = new QGroupBox("Import", this);
  auto      *mainLayout = new QVBoxLayout(group);

  // Top: Mode toggle
  import_cam_check_ = new QCheckBox("Use Camera", group);
  mainLayout->addWidget(import_cam_check_);

  // Middle: Stacked pages
  auto *stack = new QStackedLayout();
  mainLayout->addLayout(stack);

  // Page 0: File import
  QWidget *filePage = new QWidget(group);
  auto    *fileGrid = new QGridLayout(filePage);

  // Row 0: File selection
  import_file_line_edit_ = new QLineEdit(filePage);
  import_file_line_edit_->setPlaceholderText("Select File");
  import_file_line_edit_->setReadOnly(true);
  fileGrid->addWidget(import_file_line_edit_, 0, 0);

  import_browse_button_ = new QPushButton("…", filePage);
  import_browse_button_->setFixedWidth(30);
  fileGrid->addWidget(import_browse_button_, 0, 1);
  connect(import_browse_button_, &QPushButton::clicked, this, [=]() {
    // TODO: make this a separate method
    QString file = QFileDialog::getOpenFileName(this, tr("Select File"));
    if (file.isEmpty()) {
      return;
    }
    try {
      auto reader      = holofile::Reader(file.toStdString());
      auto frame_count = reader.header().frame_count;
      import_file_line_edit_->setText(file);
      import_start_index_spin_->setValue(0);
      import_end_index_spin_->setValue(frame_count);
      import_end_index_spin_->setRange(0, frame_count);
      view_x_spin_->setValue(0);
      view_x_width_spin_->setValue(reader.header().frame_width);
      view_y_spin_->setValue(0);
      view_y_width_spin_->setValue(reader.header().frame_height);
    } catch (std::exception &e) {
      logger()->error("failed to open \"{}\": \"{}\"", file.toStdString(), e.what());
    }
  });

  // Row 1: Input FPS
  fileGrid->addWidget(new QLabel("Input FPS", filePage), 1, 0);
  import_fps_spin_ = new QSpinBox(filePage);
  import_fps_spin_->setRange(1, 999999);
  import_fps_spin_->setValue(30000);
  fileGrid->addWidget(import_fps_spin_, 1, 1);

  // Row 2: Start Index
  fileGrid->addWidget(new QLabel("Start Index", filePage), 2, 0);
  import_start_index_spin_ = new QSpinBox(filePage);
  import_start_index_spin_->setRange(0, 999999);
  import_start_index_spin_->setValue(1);
  fileGrid->addWidget(import_start_index_spin_, 2, 1);

  // Row 3: End Index
  fileGrid->addWidget(new QLabel("End Index", filePage), 3, 0);
  import_end_index_spin_ = new QSpinBox(filePage);
  import_end_index_spin_->setRange(1, 999999);
  import_end_index_spin_->setValue(60);
  fileGrid->addWidget(import_end_index_spin_, 3, 1);

  // Row 4: Load method combo
  import_load_method_combo_ = new QComboBox(filePage);
  import_load_method_combo_->addItems({"Read Live", "Load in CPU RAM", "Load in GPU RAM"});
  fileGrid->addWidget(import_load_method_combo_, 4, 0, 1, 2);

  fileGrid->setRowStretch(5, 1);
  stack->addWidget(filePage);

  // Page 1: Camera import
  QWidget *camPage = new QWidget(group);
  auto    *camGrid = new QGridLayout(camPage);

  // Row 0: Camera selection
  camGrid->addWidget(new QLabel("Camera", camPage), 0, 0);
  import_camera_combo_ = new QComboBox(camPage);
  import_camera_combo_->addItems({"Ametek S710 Euresys Coaxlink Octo"});
  camGrid->addWidget(import_camera_combo_, 0, 1);

  // Row 1: Config file selection
  camGrid->addWidget(new QLabel("Config File", camPage), 1, 0);
  import_cam_config_line_edit_ = new QLineEdit(camPage);
  import_cam_config_line_edit_->setPlaceholderText("Select Config");
  import_cam_config_line_edit_->setReadOnly(true);
  camGrid->addWidget(import_cam_config_line_edit_, 1, 1);
  import_cam_config_browse_button_ = new QPushButton("…", camPage);
  import_cam_config_browse_button_->setFixedWidth(30);
  camGrid->addWidget(import_cam_config_browse_button_, 1, 2);
  connect(import_cam_config_browse_button_, &QPushButton::clicked, this, [=]() {
    // TODO [connect browse button to QFileDialog for config files]
    QString file = QFileDialog::getOpenFileName(this, tr("Select File"));
    if (file.isEmpty()) {
      return;
    }
    try {
      import_cam_config_line_edit_->setText(file);
    } catch (std::exception &e) {
      logger()->error("failed to open \"{}\": \"{}\"", file.toStdString(), e.what());
    }
  });

  camGrid->setRowStretch(2, 1);
  stack->addWidget(camPage);

  // Bottom: Shared Start / Stop + spacer
  auto *btnLayout      = new QHBoxLayout();
  import_start_button_ = new QPushButton("Start", group);
  import_stop_button_  = new QPushButton("Stop", group);
  import_stop_button_->setEnabled(false);
  btnLayout->addWidget(import_start_button_);
  btnLayout->addWidget(import_stop_button_);
  mainLayout->addLayout(btnLayout);

  mainLayout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding));

  // Logic: switch pages on toggle
  connect(import_cam_check_, &QCheckBox::toggled, stack,
          [stack](bool checked) { stack->setCurrentIndex(checked ? 1 : 0); });
  // initialize to file page
  stack->setCurrentIndex(0);

  return group;
}

QGroupBox *MainWindow::create_export_group() {
  QGroupBox *group  = new QGroupBox("Export", this);
  auto      *layout = new QGridLayout(group);

  // Row 0: Image type combo
  export_image_type_combo_ = new QComboBox(group);
  export_image_type_combo_->addItems({"Raw Image", "Processed Image"});
  layout->addWidget(export_image_type_combo_, 0, 0, 1, 2);

  // Row 1: File path and browse button
  export_file_line_edit_ = new QLineEdit(group);
  export_file_line_edit_->setText("holovibes\\capture");
  export_file_line_edit_->setReadOnly(true);
  layout->addWidget(export_file_line_edit_, 1, 0);

  export_browse_button_ = new QPushButton("...", group);
  export_browse_button_->setFixedWidth(30);
  layout->addWidget(export_browse_button_, 1, 1);
  connect(export_browse_button_, &QPushButton::clicked, this, [=]() {
    QString file = QFileDialog::getOpenFileName(this, tr("Select File"));
    if (!file.isEmpty()) {
      export_file_line_edit_->setText(file);
    }
  });

  // Row 2: Tag selection
  layout->addWidget(new QLabel("Tag", group), 2, 0);
  export_tag_combo_ = new QComboBox(group);
  export_tag_combo_->addItems({"None", "Left Eye", "Right Eye"});
  layout->addWidget(export_tag_combo_, 2, 1);

  // Row 3: Frames checkbox and spin box
  export_frames_check_ = new QCheckBox("Nb. of frames", group);
  export_frames_check_->setChecked(true);
  layout->addWidget(export_frames_check_, 3, 0);
  export_frames_spin_ = new QSpinBox(group);
  export_frames_spin_->setRange(1, 999999);
  export_frames_spin_->setValue(2048);
  layout->addWidget(export_frames_spin_, 3, 1);

  // Row 4: Action buttons: Record, Stop, Stop Fan
  export_record_button_ = new QPushButton("Record", group);
  export_stop_button_   = new QPushButton("Stop", group);
  export_stop_button_->setEnabled(false);
  export_stop_fan_button_    = new QPushButton("Stop fan", group);
  QHBoxLayout *button_layout = new QHBoxLayout();
  button_layout->addWidget(export_record_button_);
  button_layout->addWidget(export_stop_button_);
  button_layout->addWidget(export_stop_fan_button_);
  layout->addLayout(button_layout, 4, 0, 1, 2);

  // Spacer row
  layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), 5, 0, 1,
                  2);
  return group;
}

QStringList loadAvailableKernels() {
  QString appDataBase = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
  QString appDataPath = appDataBase + "/" + QCoreApplication::applicationVersion();
  QString convolutionsKernelsPath = appDataPath + "/" + "convolution_kernels";
  QDir    dir(convolutionsKernelsPath);

  QStringList filters;
  filters << QStringLiteral("*.json");
  QFileInfoList files = dir.entryInfoList(filters, QDir::Files | QDir::Readable, QDir::Name);

  QStringList names;
  names << QStringLiteral("None");
  for (const QFileInfo &fi : files) {
    names << fi.completeBaseName();
  }

  return names;
}

QGroupBox *MainWindow::create_image_rendering_group() {
  QGroupBox *group  = new QGroupBox("Image Rendering", this);
  auto      *layout = new QGridLayout(group);

  // Row 0: Image selection
  layout->addWidget(new QLabel("Image:"), 0, 0);
  render_image_combo_ = new QComboBox(group);
  render_image_combo_->addItems({"Raw", "Processed"});
  layout->addWidget(render_image_combo_, 0, 1);

  // Row 1: Batch Size
  layout->addWidget(new QLabel("Batch Size:"), 1, 0);
  render_batch_size_spin_ = new QSpinBox(group);
  render_batch_size_spin_->setRange(1, 1024 * 1024);
  render_batch_size_spin_->setValue(32);
  layout->addWidget(render_batch_size_spin_, 1, 1);

  // Row 2: Time Stride
  layout->addWidget(new QLabel("Time Stride:"), 2, 0);
  render_time_stride_spin_ = new QSpinBox(group);
  render_time_stride_spin_->setRange(1, 1024 * 1024);
  render_time_stride_spin_->setValue(32);
  layout->addWidget(render_time_stride_spin_, 2, 1);

  // Row 3: Filter 2D
  QGridLayout *sub_layout = new QGridLayout();
  render_filter_2d_check_ = new QCheckBox("Filter 2D", group);
  sub_layout->addWidget(render_filter_2d_check_, 0, 0);
  render_filter_2d_inner_spin_ = new QSpinBox(group);
  render_filter_2d_inner_spin_->setRange(0, 1024 * 1024);
  render_filter_2d_inner_spin_->setValue(0);
  sub_layout->addWidget(render_filter_2d_inner_spin_, 0, 1);
  render_filter_2d_outer_spin_ = new QSpinBox(group);
  render_filter_2d_outer_spin_->setRange(0, 1024 * 1024);
  render_filter_2d_outer_spin_->setValue(1024);
  sub_layout->addWidget(render_filter_2d_outer_spin_, 0, 2);
  layout->addLayout(sub_layout, 3, 0, 1, 2);

  // Row 4: Space Transform
  layout->addWidget(new QLabel("Space Transform:"), 4, 0);
  render_space_transform_combo_ = new QComboBox(group);
  render_space_transform_combo_->addItems({"None", "Fresnel Diffraction", "Angular Spectrum"});
  layout->addWidget(render_space_transform_combo_, 4, 1);

  // Row 5: Time Transform
  layout->addWidget(new QLabel("Time Transform:"), 5, 0);
  render_time_transform_combo_ = new QComboBox(group);
  render_time_transform_combo_->addItems(
      {"None", "Short Time Fourier", "Principal Component Analysis"});
  layout->addWidget(render_time_transform_combo_, 5, 1);

  // Row 6: Time Window
  layout->addWidget(new QLabel("Time Window:"), 6, 0);
  render_time_window_spin_ = new QSpinBox(group);
  render_time_window_spin_->setRange(1, 1024 * 1024);
  render_time_window_spin_->setValue(32);
  layout->addWidget(render_time_window_spin_, 6, 1);

  // Row 7: Lambda (nm)
  layout->addWidget(new QLabel("λ (nm):"), 7, 0);
  render_lambda_spin_ = new QSpinBox(group);
  render_lambda_spin_->setRange(1, 1024 * 1024);
  render_lambda_spin_->setValue(852);
  layout->addWidget(render_lambda_spin_, 7, 1);

  // Row 8: Boundary (mm)
  layout->addWidget(new QLabel("Boundary (mm):"), 8, 0);
  render_boundary_spin_ = new QSpinBox(group);
  render_boundary_spin_->setRange(1, 1024 * 1024);
  render_boundary_spin_->setValue(0);
  layout->addWidget(render_boundary_spin_, 8, 1);

  // Row 9: Focus (mm)
  layout->addWidget(new QLabel("Focus (mm):"), 9, 0);
  render_focus_spin_ = new QSpinBox(group);
  render_focus_spin_->setRange(1, 1024 * 1024);
  render_focus_spin_->setValue(380);
  layout->addWidget(render_focus_spin_, 9, 1);
  connect(render_focus_spin_, &QSpinBox::valueChanged, this, [=](int value) {
    if (render_focus_slider_->value() != value) {
      render_focus_slider_->blockSignals(true);
      render_focus_slider_->setValue(value);
      render_focus_slider_->blockSignals(false);
    }
  });

  // Row 10: Focus slider
  render_focus_slider_ = new QSlider(Qt::Horizontal, group);
  render_focus_slider_->setRange(0, 1000);
  render_focus_slider_->setValue(380);
  layout->addWidget(render_focus_slider_, 10, 0, 1, 2);
  connect(render_focus_slider_, &QSlider::valueChanged, this, [=](int value) {
    if (render_focus_spin_->value() != value) {
      render_focus_spin_->blockSignals(true);
      render_focus_spin_->setValue(value);
      render_focus_spin_->blockSignals(false);
    }
  });

  // Row 11: Convolution
  layout->addWidget(new QLabel("Convolution:"), 11, 0, 1, 2);
  render_convolution_combo_ = new QComboBox(group);
  auto names                = loadAvailableKernels();
  render_convolution_combo_->addItems(names);
  layout->addWidget(render_convolution_combo_, 12, 0);
  render_convolution_divide_check_ = new QCheckBox("Divide", group);
  layout->addWidget(render_convolution_divide_check_, 12, 1, 1, 1, Qt::AlignRight);

  // Spacer
  layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), 13, 0, 1,
                  2);
  return group;
}

QGroupBox *MainWindow::create_view_group() {
  QGroupBox *group  = new QGroupBox("View", this);
  auto      *layout = new QGridLayout(group);

  // Row 0: Image Type
  layout->addWidget(new QLabel("Image Type:"), 0, 0);
  view_image_type_combo_ = new QComboBox(group);
  view_image_type_combo_->addItems({"Magnitude", "Phase"});
  layout->addWidget(view_image_type_combo_, 0, 1);

  // Row 1: 3D Cuts and FFT Shift
  view_cuts_3d_check_ = new QCheckBox("3D Cuts", group);
  layout->addWidget(view_cuts_3d_check_, 1, 0);
  view_fft_shift_check_ = new QCheckBox("FFT Shift", group);
  layout->addWidget(view_fft_shift_check_, 1, 1);
  connect(view_cuts_3d_check_, &QCheckBox::checkStateChanged, this, [=](int state) {
    if (state == Qt::Checked) {
      xz_processed_widget_->show();
      yz_processed_widget_->show();
    } else {
      xz_processed_widget_->hide();
      yz_processed_widget_->hide();
    }
  });

  // Row 2: Lens View and Raw View
  view_lens_view_check_ = new QCheckBox("Lens View", group);
  layout->addWidget(view_lens_view_check_, 2, 0);
  view_raw_view_check_ = new QCheckBox("Raw View", group);
  layout->addWidget(view_raw_view_check_, 2, 1);

  // Row 3: Z and Width values
  QGridLayout *sub_layout = new QGridLayout();
  sub_layout->addWidget(new QLabel("X:"), 0, 0);
  view_x_spin_ = new QSpinBox(group);
  view_x_spin_->setRange(0, 1024 * 1024);
  sub_layout->addWidget(view_x_spin_, 0, 1);
  sub_layout->addWidget(new QLabel("Width:"), 0, 2);
  view_x_width_spin_ = new QSpinBox(group);
  view_x_width_spin_->setRange(1, 1024 * 1024);
  sub_layout->addWidget(view_x_width_spin_, 0, 3);
  sub_layout->addWidget(new QLabel("Y:"), 1, 0);
  view_y_spin_ = new QSpinBox(group);
  view_y_spin_->setRange(0, 1024 * 1024);
  sub_layout->addWidget(view_y_spin_, 1, 1);
  sub_layout->addWidget(new QLabel("Width:"), 1, 2);
  view_y_width_spin_ = new QSpinBox(group);
  view_y_width_spin_->setRange(1, 1024 * 1024);
  sub_layout->addWidget(view_y_width_spin_, 1, 3);
  sub_layout->addWidget(new QLabel("Z:"), 2, 0);
  view_z_spin_ = new QSpinBox(group);
  view_z_spin_->setRange(0, 1024 * 1024);
  sub_layout->addWidget(view_z_spin_, 2, 1);
  sub_layout->addWidget(new QLabel("Width:"), 2, 2);
  view_z_width_spin_ = new QSpinBox(group);
  view_z_width_spin_->setRange(1, 1024 * 1024);
  sub_layout->addWidget(view_z_width_spin_, 2, 3);
  layout->addLayout(sub_layout, 3, 0, 1, 2);

  // Row 4: View Kind Combo
  view_kind_combo_ = new QComboBox(group);
  view_kind_combo_->addItems({"XY", "XZ", "YZ"});
  layout->addWidget(view_kind_combo_, 4, 0, 1, 2);

  // Row 5: Output image accumulation
  layout->addWidget(new QLabel("Output image accumulation:"), 5, 0);
  view_accumulation_spin_ = new QSpinBox(group);
  view_accumulation_spin_->setRange(1, 1024 * 1024);
  view_accumulation_spin_->setValue(1);
  layout->addWidget(view_accumulation_spin_, 5, 1);

  // Row 6: Brightness/Contrast group box
  QGroupBox *brightness_group = new QGroupBox("Brightness/Contrast", group);
  brightness_group->setCheckable(true);
  brightness_group->setChecked(true);
  QGridLayout *bright_layout = new QGridLayout(brightness_group);

  view_auto_check_ = new QCheckBox("Auto", brightness_group);
  bright_layout->addWidget(view_auto_check_, 0, 0);
  view_invert_check_ = new QCheckBox("Invert", brightness_group);
  bright_layout->addWidget(view_invert_check_, 0, 1);

  QGridLayout *range_layout = new QGridLayout();
  range_layout->addWidget(new QLabel("Range:"), 0, 0);
  view_range_start_spin_ = new QSpinBox(brightness_group);
  view_range_start_spin_->setRange(1, 1024 * 1024);
  view_range_start_spin_->setValue(0);
  range_layout->addWidget(view_range_start_spin_, 0, 1);
  view_range_end_spin_ = new QSpinBox(brightness_group);
  view_range_end_spin_->setRange(1, 1024 * 1024);
  view_range_end_spin_->setValue(255);
  range_layout->addWidget(view_range_end_spin_, 0, 2);
  bright_layout->addLayout(range_layout, 1, 0, 1, 2);

  view_reticle_check_ = new QCheckBox("Display reticle", brightness_group);
  bright_layout->addWidget(view_reticle_check_, 2, 0);
  view_reticle_radius_ = new QDoubleSpinBox(brightness_group);
  view_reticle_radius_->setRange(0.05, 1.0);
  view_reticle_radius_->setSingleStep(0.05);
  view_reticle_radius_->setDecimals(2);
  view_reticle_radius_->setValue(1.0);
  bright_layout->addWidget(view_reticle_radius_, 2, 1);

  view_renormalize_check_ = new QCheckBox("Renormalize image levels", brightness_group);
  bright_layout->addWidget(view_renormalize_check_, 3, 0, 1, 2);
  layout->addWidget(brightness_group, 6, 0, 1, 2);

  view_registration_check_ = new QCheckBox("registration", brightness_group);
  bright_layout->addWidget(view_registration_check_, 4, 0);
  view_registration_radius_ = new QDoubleSpinBox(brightness_group);
  view_registration_radius_->setRange(0.05, 1.0);
  view_registration_radius_->setSingleStep(0.05);
  view_registration_radius_->setDecimals(2);
  view_registration_radius_->setValue(1.0);
  bright_layout->addWidget(view_registration_radius_, 4, 1);

  // Spacer
  layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), 7, 0, 1,
                  2);
  return group;
}

} // namespace holovibes::ui
