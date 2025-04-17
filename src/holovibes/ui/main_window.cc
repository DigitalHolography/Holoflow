#include "holovibes/ui/main_window.hh"

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenuBar>
#include <QPushButton>
#include <QSlider>
#include <QSpacerItem>
#include <QSpinBox>
#include <QThread>
#include <QVBoxLayout>

#include "holofile/holofile.hh"
#include "holovibes/holovibes.hh"
#include "holovibes/pipeline/settings.hh"

namespace holovibes::ui {

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
  // Create menus
  menuBar()->addMenu("&File");
  menuBar()->addMenu("&View");
  menuBar()->addMenu("&Camera");
  menuBar()->addMenu("&Theme");

  // Create central widget and main layout
  QWidget *central_widget = new QWidget(this);
  setCentralWidget(central_widget);
  auto *main_layout = new QHBoxLayout(central_widget);

  // Left panel layout: image rendering and view sections from left to right,
  // with import and export groups on the right in a vertical stacked layout.
  QHBoxLayout *left_panel_layout = new QHBoxLayout();
  left_panel_layout->addWidget(create_image_rendering_group());
  left_panel_layout->addWidget(create_view_group());

  QVBoxLayout *import_export_layout = new QVBoxLayout();
  import_export_layout->addWidget(create_import_group());
  import_export_layout->addWidget(create_export_group());

  left_panel_layout->addLayout(import_export_layout);
  main_layout->addLayout(left_panel_layout);

  setWindowTitle("Holovibes");
  this->adjustSize();
  this->setFixedSize(this->minimumSizeHint());

  display_widget_ = new dh::TensorDisplayWidget(800, 800, this);
  display_widget_->show();
  pipeline_worker_ = new pipeline::Worker(display_widget_);
  pipeline_worker_thread_ = new QThread(this);
  pipeline_worker_->moveToThread(pipeline_worker_thread_);

  setup_validation_connections();
  setup_update_connections();

  connect(view_reticle_check_, &QCheckBox::toggled, display_widget_,
          &dh::TensorDisplayWidget::set_display_reticle);

  connect(view_reticle_radius_,
          qOverload<double>(&QDoubleSpinBox::valueChanged), display_widget_,
          &dh::TensorDisplayWidget::set_reticle_radius);

  connect(pipeline_worker_, &pipeline::Worker::start_success, this,
          [this]() { import_stop_button_->setEnabled(true); });

  connect(pipeline_worker_, &pipeline::Worker::start_failure, this,
          [this]() { import_start_button_->setEnabled(true); });

  connect(pipeline_worker_, &pipeline::Worker::stop_success, this, [this]() {
    import_start_button_->setEnabled(true);
    import_stop_button_->setEnabled(false);
  });

  connect(pipeline_worker_, &pipeline::Worker::stop_failure, this, [this]() {
    import_start_button_->setEnabled(true);
    import_stop_button_->setEnabled(false);
  });

  connect(pipeline_worker_, &pipeline::Worker::update_success, this, [this]() {
    update_in_progress_ = false;
    import_stop_button_->setEnabled(true);
  });

  connect(pipeline_worker_, &pipeline::Worker::update_failure, this, [this]() {
    update_in_progress_ = false;
    import_start_button_->setEnabled(true);
  });

  pipeline_worker_thread_->start();

  connect(import_start_button_, &QPushButton::clicked, this,
          &MainWindow::on_import_start_clicked);
  connect(import_stop_button_, &QPushButton::clicked, this,
          &MainWindow::on_import_stop_clicked);
}

void MainWindow::closeEvent(QCloseEvent *event) {
  if (pipeline_worker_ && import_stop_button_->isEnabled()) {
    on_import_stop_clicked();
  }

  if (pipeline_worker_thread_) {
    pipeline_worker_thread_->quit();
    pipeline_worker_thread_->wait();
  }

  QMainWindow::closeEvent(event);
}

void MainWindow::on_import_start_clicked() {
  if (!validate_inputs())
    return;

  holovibes::pipeline::Settings settings = get_pipeline_settings();
  pipeline_worker_->set_settings(settings);
  import_start_button_->setEnabled(false);
  QMetaObject::invokeMethod(pipeline_worker_, "start", Qt::QueuedConnection);
}

void MainWindow::on_import_stop_clicked() {
  import_stop_button_->setEnabled(false);
  QMetaObject::invokeMethod(pipeline_worker_, "stop", Qt::QueuedConnection);
}

bool MainWindow::validate_inputs() {
  int batch_size = render_batch_size_spin_->value();
  int time_stride = render_time_stride_spin_->value();

  if (batch_size > 0 && (time_stride % batch_size != 0)) {
    render_batch_size_spin_->setStyleSheet(
        "background-color: rgba(255, 0, 0, 50);");
    render_time_stride_spin_->setStyleSheet(
        "background-color: rgba(255, 0, 0, 50);");
    return false;
  } else {
    render_batch_size_spin_->setStyleSheet("");
    render_time_stride_spin_->setStyleSheet("");
  }

  int time_window = render_time_window_spin_->value();
  int p_frame_start = view_z_spin_->value();
  int p_frame_width = view_width_spin_->value();

  if (p_frame_start + p_frame_width > time_window) {
    render_time_window_spin_->setStyleSheet(
        "background-color: rgba(255, 0, 0, 50);");
    view_z_spin_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
    view_width_spin_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
    return false;
  } else {
    render_time_window_spin_->setStyleSheet("");
    view_z_spin_->setStyleSheet("");
    view_width_spin_->setStyleSheet("");
  }

  int start_frame = import_start_index_spin_->value();
  int end_frame = import_end_index_spin_->value();
  if (end_frame <= start_frame) {
    import_start_index_spin_->setStyleSheet(
        "background-color: rgba(255, 0, 0, 50);");
    import_end_index_spin_->setStyleSheet(
        "background-color: rgba(255, 0, 0, 50);");
    return false;
  } else {
    import_start_index_spin_->setStyleSheet("");
    import_end_index_spin_->setStyleSheet("");
  }

  if (time_stride > end_frame - start_frame) {
    import_start_index_spin_->setStyleSheet(
        "background-color: rgba(255, 0, 0, 50);");
    import_end_index_spin_->setStyleSheet(
        "background-color: rgba(255, 0, 0, 50);");
    render_time_stride_spin_->setStyleSheet(
        "background-color: rgba(255, 0, 0, 50);");
    return false;
  } else {
    import_start_index_spin_->setStyleSheet("");
    import_end_index_spin_->setStyleSheet("");
    render_time_stride_spin_->setStyleSheet("");
  }

  return true;
}

void MainWindow::setup_validation_connections() {
  // --- Import Group Connections ---
  connect(import_file_line_edit_, &QLineEdit::editingFinished, this,
          &MainWindow::validate_inputs);
  connect(import_browse_button_, &QPushButton::clicked, this,
          &MainWindow::validate_inputs);
  connect(import_start_button_, &QPushButton::clicked, this,
          &MainWindow::validate_inputs);
  connect(import_stop_button_, &QPushButton::clicked, this,
          &MainWindow::validate_inputs);
  connect(import_fps_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
  connect(import_start_index_spin_, qOverload<int>(&QSpinBox::valueChanged),
          this, &MainWindow::validate_inputs);
  connect(import_end_index_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
  connect(import_load_method_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::validate_inputs);

  // --- Export Group Connections ---
  connect(export_image_type_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::validate_inputs);
  connect(export_file_line_edit_, &QLineEdit::editingFinished, this,
          &MainWindow::validate_inputs);
  connect(export_browse_button_, &QPushButton::clicked, this,
          &MainWindow::validate_inputs);
  connect(export_tag_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, &MainWindow::validate_inputs);
  connect(export_frames_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);
  connect(export_frames_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
  connect(export_record_button_, &QPushButton::clicked, this,
          &MainWindow::validate_inputs);
  connect(export_stop_button_, &QPushButton::clicked, this,
          &MainWindow::validate_inputs);
  connect(export_stop_fan_button_, &QPushButton::clicked, this,
          &MainWindow::validate_inputs);

  // --- Image Rendering Group Connections ---
  connect(render_image_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, &MainWindow::validate_inputs);
  connect(render_batch_size_spin_, qOverload<int>(&QSpinBox::valueChanged),
          this, &MainWindow::validate_inputs);
  connect(render_time_stride_spin_, qOverload<int>(&QSpinBox::valueChanged),
          this, &MainWindow::validate_inputs);
  connect(render_filter_2d_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);
  connect(render_space_transform_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::validate_inputs);
  connect(render_time_transform_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::validate_inputs);
  connect(render_time_window_spin_, qOverload<int>(&QSpinBox::valueChanged),
          this, &MainWindow::validate_inputs);
  connect(render_lambda_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
  connect(render_boundary_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
  connect(render_focus_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
  connect(render_focus_slider_, &QSlider::valueChanged, this,
          &MainWindow::validate_inputs);
  connect(render_convolution_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::validate_inputs);
  connect(render_convolution_divide_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);

  // --- View Group Connections ---
  connect(view_image_type_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::validate_inputs);
  connect(view_cuts_3d_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);
  connect(view_fft_shift_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);
  connect(view_lens_view_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);
  connect(view_raw_view_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);
  connect(view_z_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
  connect(view_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
  connect(view_kind_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, &MainWindow::validate_inputs);
  connect(view_accumulation_spin_, qOverload<int>(&QSpinBox::valueChanged),
          this, &MainWindow::validate_inputs);
  connect(view_auto_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);
  connect(view_invert_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);
  connect(view_range_start_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
  connect(view_range_end_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
  connect(view_renormalize_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);
  connect(view_reticle_check_, &QCheckBox::toggled, this,
          &MainWindow::validate_inputs);
  connect(view_reticle_radius_,
          qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          &MainWindow::validate_inputs);
}

void MainWindow::setup_update_connections() {
  // --- Import Group Connections ---
  connect(import_file_line_edit_, &QLineEdit::editingFinished, this,
          &MainWindow::update_if_running);
  connect(import_browse_button_, &QPushButton::clicked, this,
          &MainWindow::update_if_running);
  connect(import_start_button_, &QPushButton::clicked, this,
          &MainWindow::update_if_running);
  connect(import_stop_button_, &QPushButton::clicked, this,
          &MainWindow::update_if_running);
  connect(import_fps_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
  connect(import_start_index_spin_, qOverload<int>(&QSpinBox::valueChanged),
          this, &MainWindow::update_if_running);
  connect(import_end_index_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
  connect(import_load_method_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::update_if_running);

  // --- Export Group Connections ---
  connect(export_image_type_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::update_if_running);
  connect(export_file_line_edit_, &QLineEdit::editingFinished, this,
          &MainWindow::update_if_running);
  connect(export_browse_button_, &QPushButton::clicked, this,
          &MainWindow::update_if_running);
  connect(export_tag_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, &MainWindow::update_if_running);
  connect(export_frames_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);
  connect(export_frames_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
  connect(export_record_button_, &QPushButton::clicked, this,
          &MainWindow::update_if_running);
  connect(export_stop_button_, &QPushButton::clicked, this,
          &MainWindow::update_if_running);
  connect(export_stop_fan_button_, &QPushButton::clicked, this,
          &MainWindow::update_if_running);

  // --- Image Rendering Group Connections ---
  connect(render_image_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, &MainWindow::update_if_running);
  connect(render_batch_size_spin_, qOverload<int>(&QSpinBox::valueChanged),
          this, &MainWindow::update_if_running);
  connect(render_time_stride_spin_, qOverload<int>(&QSpinBox::valueChanged),
          this, &MainWindow::update_if_running);
  connect(render_filter_2d_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);
  connect(render_space_transform_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::update_if_running);
  connect(render_time_transform_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::update_if_running);
  connect(render_time_window_spin_, qOverload<int>(&QSpinBox::valueChanged),
          this, &MainWindow::update_if_running);
  connect(render_lambda_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
  connect(render_boundary_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
  connect(render_focus_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
  connect(render_focus_slider_, &QSlider::valueChanged, this,
          &MainWindow::update_if_running);
  connect(render_convolution_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::update_if_running);
  connect(render_convolution_divide_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);

  // --- View Group Connections ---
  connect(view_image_type_combo_,
          qOverload<int>(&QComboBox::currentIndexChanged), this,
          &MainWindow::update_if_running);
  connect(view_cuts_3d_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);
  connect(view_fft_shift_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);
  connect(view_lens_view_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);
  connect(view_raw_view_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);
  connect(view_z_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
  connect(view_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
  connect(view_kind_combo_, qOverload<int>(&QComboBox::currentIndexChanged),
          this, &MainWindow::update_if_running);
  connect(view_accumulation_spin_, qOverload<int>(&QSpinBox::valueChanged),
          this, &MainWindow::update_if_running);
  connect(view_auto_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);
  connect(view_invert_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);
  connect(view_range_start_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
  connect(view_range_end_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
  connect(view_renormalize_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);
  connect(view_reticle_check_, &QCheckBox::toggled, this,
          &MainWindow::update_if_running);
  connect(view_reticle_radius_,
          qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          &MainWindow::update_if_running);
}

void MainWindow::update_if_running() {
  if (!pipeline_worker_ || !import_stop_button_->isEnabled()) {
    return;
  }

  if (update_in_progress_) {
    return;
  }

  if (!validate_inputs())
    return;

  update_in_progress_ = true;

  holovibes::pipeline::Settings settings = get_pipeline_settings();
  pipeline_worker_->set_settings(settings);
  import_start_button_->setEnabled(false);
  QMetaObject::invokeMethod(pipeline_worker_, "update", Qt::QueuedConnection);
}

holovibes::pipeline::Settings MainWindow::get_pipeline_settings() {
  using namespace holovibes::pipeline;

  Settings s;

  // --- Import Settings ---
  s.import_file_path = import_file_line_edit_->text().toStdString();
  s.import_fps = static_cast<size_t>(import_fps_spin_->value());
  s.import_start_index = static_cast<size_t>(import_start_index_spin_->value());
  s.import_end_index = static_cast<size_t>(import_end_index_spin_->value());

  {
    QString method = import_load_method_combo_->currentText();
    if (method == "Read Live")
      s.import_load_method = ImportLoadMethod::ReadLive;
    else if (method == "Load in CPU RAM")
      s.import_load_method = ImportLoadMethod::LoadInCPU;
    else if (method == "Load in GPU RAM")
      s.import_load_method = ImportLoadMethod::LoadInGPU;
  }

  // --- Export Settings ---
  {
    QString type = export_image_type_combo_->currentText();
    if (type == "Raw Image")
      s.export_image_type = ExportImageType::Raw;
    else if (type == "Processed Image")
      s.export_image_type = ExportImageType::Processed;
  }
  s.export_file_path = export_file_line_edit_->text().toStdString();
  {
    QString tag = export_tag_combo_->currentText();
    if (tag == "Left Eye")
      s.export_tag = ExportTag::LeftEye;
    else if (tag == "Right Eye")
      s.export_tag = ExportTag::RightEye;
  }
  // If the export frames check is enabled, set the frame count; otherwise,
  // leave it empty.
  if (export_frames_check_->isChecked()) {
    s.export_frame_count = static_cast<size_t>(export_frames_spin_->value());
  } else {
    s.export_frame_count = std::nullopt;
  }

  // --- Image Rendering Settings ---
  {
    QString rtype = render_image_combo_->currentText();
    if (rtype == "Raw")
      s.render_type = RenderType::Raw;
    else if (rtype == "Processed")
      s.render_type = RenderType::Processed;
  }
  s.render_batch_size = static_cast<size_t>(render_batch_size_spin_->value());
  s.render_time_stride = static_cast<size_t>(render_time_stride_spin_->value());
  s.render_filter_2d = render_filter_2d_check_->isChecked();
  {
    QString st = render_space_transform_combo_->currentText();
    if (st == "Fresnel Diffraction")
      s.render_space_transform = RenderSpaceTransform::FresnelDiffraction;
    else if (st == "Angular Spectrum")
      s.render_space_transform = RenderSpaceTransform::AngularSpectrum;
    else
      s.render_space_transform = std::nullopt;
  }
  {
    QString tt = render_time_transform_combo_->currentText();
    if (tt == "Short Time Fourier Transform")
      s.render_time_transform = RenderTimeTransform::ShortTimeFourier;
    else if (tt == "Principal Component Analysis")
      s.render_time_transform = RenderTimeTransform::PrincipalComponentAnalysis;
    else
      s.render_time_transform = std::nullopt;
  }
  s.render_time_window = static_cast<size_t>(render_time_window_spin_->value());
  s.render_lambda = static_cast<size_t>(render_lambda_spin_->value());
  s.render_focus = static_cast<size_t>(render_focus_spin_->value());
  {
    QString conv = render_convolution_combo_->currentText();
    if (conv == "Gaussian")
      s.render_convolution = RenderConvolution::Gaussian;
    else
      s.render_convolution = std::nullopt;
  }
  s.render_convolution_divide = render_convolution_divide_check_->isChecked();

  // --- View Settings ---
  // We have only one view type available in the Settings.
  s.view_type = ViewType::Magnitude;
  s.view_cuts_3d = view_cuts_3d_check_->isChecked();
  s.view_fft_shift = view_fft_shift_check_->isChecked();
  s.view_lens_view = view_lens_view_check_->isChecked();
  s.view_raw_view = view_raw_view_check_->isChecked();
  s.view_p_frame_start = static_cast<size_t>(view_z_spin_->value());
  s.view_p_frame_width = static_cast<size_t>(view_width_spin_->value());
  {
    QString axis = view_kind_combo_->currentText();
    if (axis == "XY")
      s.view_axis = ViewAxis::XY;
    else if (axis == "XZ")
      s.view_axis = ViewAxis::XZ;
    else if (axis == "YZ")
      s.view_axis = ViewAxis::YZ;
  }
  s.view_accumulation = static_cast<size_t>(view_accumulation_spin_->value());
  s.view_auto_contrast = view_auto_check_->isChecked();
  s.view_invert_contrast = view_invert_check_->isChecked();
  s.view_contrast_low = static_cast<size_t>(view_range_start_spin_->value());
  s.view_contrast_high = static_cast<size_t>(view_range_end_spin_->value());
  s.view_renormalize = view_renormalize_check_->isChecked();
  s.view_lower_percentile_ = 0.2f;
  s.view_upper_percentile_ = 99.8f;
  s.view_reticule_radius_ = view_reticle_radius_->value();

  return s;
}

QGroupBox *MainWindow::create_import_group() {
  QGroupBox *group = new QGroupBox("Import", this);
  auto *layout = new QGridLayout(group);

  // Row 0: File selection
  import_file_line_edit_ = new QLineEdit(group);
  import_file_line_edit_->setPlaceholderText("Select File");
  import_file_line_edit_->setReadOnly(true);
  layout->addWidget(import_file_line_edit_, 0, 0);

  import_browse_button_ = new QPushButton("...", group);
  import_browse_button_->setFixedWidth(30);
  layout->addWidget(import_browse_button_, 0, 1);
  connect(import_browse_button_, &QPushButton::clicked, this, [=]() {
    QString file = QFileDialog::getOpenFileName(this, tr("Select File"));
    if (file.isEmpty()) {
      return;
    }

    auto reader_result = dh::HolofileReader::open(file.toStdString());
    if (!reader_result) {
      dh::holovibes_logger()->error("failed to open \"{}\": \"{}\"",
                                    file.toStdString(),
                                    reader_result.error().message());
      return;
    }

    auto reader = std::move(reader_result.value());
    auto frame_count = reader.header().frame_count;
    import_file_line_edit_->setText(file);
    import_start_index_spin_->setValue(0);
    import_end_index_spin_->setValue(frame_count);
  });

  // Row 1: Input FPS
  layout->addWidget(new QLabel("Input FPS", group), 1, 0);
  import_fps_spin_ = new QSpinBox(group);
  import_fps_spin_->setRange(1, 999999);
  import_fps_spin_->setValue(30000);
  layout->addWidget(import_fps_spin_, 1, 1);

  // Row 2: Start Index
  layout->addWidget(new QLabel("Start Index", group), 2, 0);
  import_start_index_spin_ = new QSpinBox(group);
  import_start_index_spin_->setRange(0, 999999);
  import_start_index_spin_->setValue(1);
  layout->addWidget(import_start_index_spin_, 2, 1);

  // Row 3: End Index
  layout->addWidget(new QLabel("End Index", group), 3, 0);
  import_end_index_spin_ = new QSpinBox(group);
  import_end_index_spin_->setRange(1, 999999);
  import_end_index_spin_->setValue(60);
  layout->addWidget(import_end_index_spin_, 3, 1);

  // Row 4: Load method combo
  import_load_method_combo_ = new QComboBox(group);
  import_load_method_combo_->addItems(
      {"Read Live", "Load in CPU RAM", "Load in GPU RAM"});
  layout->addWidget(import_load_method_combo_, 4, 0, 1, 2);

  // Row 5: Start / Stop buttons
  import_start_button_ = new QPushButton("Start", group);
  import_stop_button_ = new QPushButton("Stop", group);
  import_stop_button_->setEnabled(false);
  QHBoxLayout *btn_layout = new QHBoxLayout();
  btn_layout->addWidget(import_start_button_);
  btn_layout->addWidget(import_stop_button_);
  layout->addLayout(btn_layout, 5, 0, 1, 2);

  // Spacer to push components upward
  layout->addItem(
      new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), 6,
      0, 1, 2);
  return group;
}

QGroupBox *MainWindow::create_export_group() {
  QGroupBox *group = new QGroupBox("Export", this);
  auto *layout = new QGridLayout(group);

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
      // pipeline_mgr_->update_export_file(file);
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
  export_stop_button_ = new QPushButton("Stop", group);
  export_stop_fan_button_ = new QPushButton("Stop fan", group);
  QHBoxLayout *button_layout = new QHBoxLayout();
  button_layout->addWidget(export_record_button_);
  button_layout->addWidget(export_stop_button_);
  button_layout->addWidget(export_stop_fan_button_);
  layout->addLayout(button_layout, 4, 0, 1, 2);

  // Spacer row
  layout->addItem(
      new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), 5,
      0, 1, 2);
  return group;
}

QGroupBox *MainWindow::create_image_rendering_group() {
  QGroupBox *group = new QGroupBox("Image Rendering", this);
  auto *layout = new QGridLayout(group);

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
  render_filter_2d_check_ = new QCheckBox("Filter 2D", group);
  layout->addWidget(render_filter_2d_check_, 3, 0, 1, 2, Qt::AlignRight);

  // Row 4: Space Transform
  layout->addWidget(new QLabel("Space Transform:"), 4, 0);
  render_space_transform_combo_ = new QComboBox(group);
  render_space_transform_combo_->addItems(
      {"None", "Fresnel Diffraction", "Angular Spectrum"});
  layout->addWidget(render_space_transform_combo_, 4, 1);

  // Row 5: Time Transform
  layout->addWidget(new QLabel("Time Transform:"), 5, 0);
  render_time_transform_combo_ = new QComboBox(group);
  render_time_transform_combo_->addItems(
      {"None", "Short Time Fourier Transform", "Principal Component Analysis"});
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
  render_convolution_combo_->addItems({"None", "Gaussian", "Laplacian"});
  layout->addWidget(render_convolution_combo_, 12, 0);
  render_convolution_divide_check_ = new QCheckBox("Divide", group);
  layout->addWidget(render_convolution_divide_check_, 12, 1, 1, 1,
                    Qt::AlignRight);

  // Spacer
  layout->addItem(
      new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), 13,
      0, 1, 2);
  return group;
}

QGroupBox *MainWindow::create_view_group() {
  QGroupBox *group = new QGroupBox("View", this);
  auto *layout = new QGridLayout(group);

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

  // Row 2: Lens View and Raw View
  view_lens_view_check_ = new QCheckBox("Lens View", group);
  layout->addWidget(view_lens_view_check_, 2, 0);
  view_raw_view_check_ = new QCheckBox("Raw View", group);
  layout->addWidget(view_raw_view_check_, 2, 1);

  // Row 3: Z and Width values
  QGridLayout *sub_layout = new QGridLayout();
  sub_layout->addWidget(new QLabel("Z:"), 0, 0);
  view_z_spin_ = new QSpinBox(group);
  view_z_spin_->setRange(0, 1024 * 1024);
  sub_layout->addWidget(view_z_spin_, 0, 1);
  sub_layout->addWidget(new QLabel("Width:"), 0, 2);
  view_width_spin_ = new QSpinBox(group);
  view_width_spin_->setRange(1, 1024 * 1024);
  sub_layout->addWidget(view_width_spin_, 0, 3);
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

  view_renormalize_check_ =
      new QCheckBox("Renormalize image levels", brightness_group);
  bright_layout->addWidget(view_renormalize_check_, 3, 0, 1, 2);
  layout->addWidget(brightness_group, 6, 0, 1, 2);

  // Spacer
  layout->addItem(
      new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), 7,
      0, 1, 2);
  return group;
}

} // namespace holovibes::ui