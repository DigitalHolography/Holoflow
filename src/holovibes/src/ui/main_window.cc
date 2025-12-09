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
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfoList>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenuBar>
#include <QMessageBox>
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
#include <filesystem>
#include <fstream>
#include <optional>

#include "bug.hh"
#include "holofile/holofile.hh"
#include "logger.hh"
#include "settings_loader.hh"

namespace {

constexpr int kLargeSpinMax = 1024 * 1024;

QSpinBox *create_spin_box(QWidget *parent, int minimum, int maximum, int value) {
  auto *spin_box = new QSpinBox(parent);
  spin_box->setRange(minimum, maximum);
  spin_box->setValue(value);
  return spin_box;
}

QDoubleSpinBox *create_double_spin_box(QWidget *parent, double minimum, double maximum, double step,
                                       double value, int decimals = 2) {
  auto *spin_box = new QDoubleSpinBox(parent);
  spin_box->setRange(minimum, maximum);
  spin_box->setSingleStep(step);
  spin_box->setDecimals(decimals);
  spin_box->setValue(value);
  return spin_box;
}

QComboBox *create_combo_box(QWidget *parent, const QStringList &items) {
  auto *combo_box = new QComboBox(parent);
  combo_box->addItems(items);
  return combo_box;
}

} // namespace

namespace holovibes::ui {

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
  setup_menu_bar();
  setup_main_layout();
  initialize_display_widgets();
  initialize_pipeline_manager();

  connect_manager_signals();
  connect_import_controls();
  setup_validation_connections();
  setup_update_connections();

  validate_inputs();
  configure_window();
}

void MainWindow::setup_menu_bar() {
  auto *bar = menuBar();
  bar->addMenu("&File");
  bar->addMenu("&View");
  bar->addMenu("&Camera");
  bar->addMenu("&Theme");
}

void MainWindow::setup_main_layout() {
  auto *central_widget = new QWidget(this);
  setCentralWidget(central_widget);

  auto *main_layout = new QHBoxLayout(central_widget);
  main_layout->setSpacing(16);

  auto *left_panel_layout = new QHBoxLayout();
  left_panel_layout->setSpacing(12);

  // Create widget instances
  render_widget_ = new ImageRenderingWidget(this);
  view_widget_   = new ViewWidget(this);
  import_widget_ = new ImportWidget(this);
  export_widget_ = new ExportWidget(this);

  auto *import_export_layout = new QVBoxLayout();
  import_export_layout->setSpacing(12);
  import_export_layout->addWidget(import_widget_);
  import_export_layout->addWidget(export_widget_);

  left_panel_layout->addWidget(render_widget_);
  left_panel_layout->addWidget(view_widget_);
  left_panel_layout->addLayout(import_export_layout);

  monitor_widget_ = new SystemMonitorWidget(this);
  main_layout->addLayout(left_panel_layout);
  main_layout->addWidget(monitor_widget_, 0, Qt::AlignTop);
}

void MainWindow::initialize_display_widgets() {
  xy_raw_widget_       = new TensorDisplayWidget(nullptr);
  xy_processed_widget_ = new TensorDisplayWidget(nullptr);
  xz_processed_widget_ = new TensorDisplayWidget(nullptr);
  yz_processed_widget_ = new TensorDisplayWidget(nullptr);

  xy_processed_widget_->setWindowTitle("XY-Processed");
  xz_processed_widget_->setWindowTitle("XZ-Processed");
  yz_processed_widget_->setWindowTitle("YZ-Processed");
  xy_raw_widget_->setWindowTitle("XY-Raw");

  // xy_processed_widget_->resize(512 * 2, 320 * 2);
  // xy_processed_widget_->show();

  connect(view_widget_, &ViewWidget::cuts_3d_toggled, this, [this](bool checked) {
    if (checked) {
      xz_processed_widget_->show();
      yz_processed_widget_->show();
    } else {
      xz_processed_widget_->hide();
      yz_processed_widget_->hide();
    }
  });

  connect(view_widget_, &ViewWidget::raw_view_toggled, this, [this](bool checked) {
    if (pipeline_running_ && checked) {
      xy_raw_widget_->show();
    } else {
      xy_raw_widget_->hide();
    }
  });

  connect(view_widget_, &ViewWidget::reticle_toggled, this, [this](bool checked) {
    if (xy_processed_widget_) {
      xy_processed_widget_->set_reticle_enabled(checked);
      if (checked) {
        xy_processed_widget_->set_reticle_radius(view_widget_->get_reticle_radius());
      }
    }
  });

  connect(view_widget_, &ViewWidget::reticle_radius_changed, this, [this](double value) {
    if (xy_processed_widget_ && view_widget_->is_reticle_enabled()) {
      xy_processed_widget_->set_reticle_radius(value);
    }
  });
}

void MainWindow::initialize_pipeline_manager() {
  pipeline_manager_        = new pipeline::Manager(xy_processed_widget_, xz_processed_widget_,
                                                   yz_processed_widget_, xy_raw_widget_);
  pipeline_manager_thread_ = new QThread(this);
  pipeline_manager_->moveToThread(pipeline_manager_thread_);
  pipeline_manager_thread_->start();
}

void MainWindow::connect_manager_signals() {
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

  connect(pipeline_manager_, &pipeline::Manager::raw_record_started_success, this,
          &MainWindow::on_raw_record_started_success, Qt::QueuedConnection);

  connect(pipeline_manager_, &pipeline::Manager::raw_record_started_failure, this,
          &MainWindow::on_raw_record_started_failure, Qt::QueuedConnection);

  connect(pipeline_manager_, &pipeline::Manager::raw_record_stopped_success, this,
          &MainWindow::on_raw_record_stopped_success, Qt::QueuedConnection);

  connect(pipeline_manager_, &pipeline::Manager::raw_record_stopped_failure, this,
          &MainWindow::on_raw_record_stopped_failure, Qt::QueuedConnection);
}

void MainWindow::connect_import_controls() {
  connect(import_widget_, &ImportWidget::start_clicked, this, &MainWindow::on_import_start_clicked);
  connect(import_widget_, &ImportWidget::stop_clicked, this, &MainWindow::on_import_stop_clicked);
  connect(import_widget_, &ImportWidget::browse_clicked, this, [=]() {
    QString file = QFileDialog::getOpenFileName(this, tr("Select File"));
    if (file.isEmpty()) {
      return;
    }
    try {
      auto reader      = holofile::Reader(file.toStdString());
      auto frame_count = reader.header().frame_count;

      import_widget_->set_file_path(file);
      import_widget_->set_end_index_range(0, static_cast<int>(frame_count));
      import_widget_->set_end_index(static_cast<int>(frame_count));
      import_widget_->set_start_index(0);

      if (reader.footer().has_value()) {
        auto               footer        = reader.footer().value();
        pipeline::Settings prev_settings = get_pipeline_settings();
        pipeline::Settings new_settings =
            pipeline::old_json_to_settings(footer.pipeline_settings, prev_settings);
        set_pipeline_settings(new_settings);
      }
    } catch (std::exception &e) {
      logger()->error("failed to open \"{}\": \"{}\"", file.toStdString(), e.what());
    }
  });

  connect(export_widget_, &ExportWidget::record_clicked, this,
          &MainWindow::on_export_record_clicked);
  connect(export_widget_, &ExportWidget::stop_clicked, this, &MainWindow::on_export_stop_clicked);
  connect(export_widget_, &ExportWidget::browse_clicked, this, [=]() {
    QString file = QFileDialog::getSaveFileName(this, tr("Select File"));
    if (!file.isEmpty()) {
      QFileInfo info(file);
      QString   dirPath  = info.absolutePath();
      QString   baseName = info.completeBaseName();

      QRegularExpression      autoPattern(R"(^\d{6}_(.*?)$)");
      QRegularExpressionMatch match = autoPattern.match(baseName);
      if (match.hasMatch())
        baseName = match.captured(1);

      QString cleanedFileName = baseName + ".holo";

      QString finalDisplayPath =
          dirPath.isEmpty() ? cleanedFileName : dirPath + "/" + cleanedFileName;

      export_widget_->set_file_path(finalDisplayPath);
    }
  });
}

void MainWindow::configure_window() {
  setWindowTitle("Holovibes");
  setFixedSize(minimumSizeHint());
  show();
}

std::filesystem::path MainWindow::makeRecordingPath(const QString &userText) const {
  namespace fs = std::filesystem;

  QFileInfo info(userText);

  fs::path dir = info.absolutePath().isEmpty() ? fs::current_path()
                                               : fs::path(info.absolutePath().toStdString());

  QString baseName = info.completeBaseName();

  // Ensure .holo extension
  QString extension = ".holo";

  // Date prefix
  QString datePrefix = QDate::currentDate().toString("yyMMdd") + "_";

  // Candidate file
  QString  finalFileName = datePrefix + baseName + extension;
  fs::path candidate     = dir / finalFileName.toStdString();

  // Add suffix if needed
  int counter = 1;
  while (fs::exists(candidate)) {
    finalFileName = datePrefix + baseName + QString("_%1").arg(counter++) + extension;
    candidate     = dir / finalFileName.toStdString();
  }

  logger()->info("[MainWindow::makeRecordingPath] Generated recording path: {}",
                 candidate.string());

  return candidate;
}

void MainWindow::show_pipeline_error_popup(const QString &message) {
  QMessageBox msgBox(this);
  msgBox.setIcon(QMessageBox::Critical);
  msgBox.setWindowTitle(tr("Pipeline Error"));
  msgBox.setText(message);
  msgBox.setStandardButtons(QMessageBox::Ok);
  msgBox.setDefaultButton(QMessageBox::Ok);
  msgBox.exec();
}

void MainWindow::on_start_pipeline_success() {
  logger()->info("[MainWindow::on_start_pipeline_success]");
  pipeline_running_ = true;
  import_widget_->set_stop_enabled(true);
  export_widget_->set_record_enabled(!export_in_progress_ && export_widget_->isChecked());
  export_widget_->set_stop_enabled(export_in_progress_);

  auto dims = guess_source_dims();
  xy_raw_widget_->set_fixed_aspect(dims);
  if (render_widget_->get_space_transform() == "Fresnel Diffraction" &&
      render_widget_->get_image_mode() != "Raw") {
    dims = QSize(dims.width(), dims.width());
  }
  xy_processed_widget_->set_fixed_aspect(dims);
  xy_raw_widget_->set_fixed_aspect(guess_source_dims());

  xy_processed_widget_->show();
  if (view_widget_->is_raw_view_enabled()) {
    xy_raw_widget_->show();
  }
}

void MainWindow::on_start_pipeline_failure(const QString &error) {
  logger()->error("[MainWindow::on_start_pipeline_failure]");
  pipeline_running_   = false;
  export_in_progress_ = false;
  import_widget_->set_start_enabled(true);
  import_widget_->set_stop_enabled(false);
  export_widget_->set_record_enabled(false);
  export_widget_->set_stop_enabled(false);

  show_pipeline_error_popup(tr("An error occurred while starting the pipeline:\n%1").arg(error));
}

void MainWindow::on_stop_pipeline_success() {
  logger()->info("[MainWindow::on_stop_pipeline_success]");
  pipeline_running_ = false;
  import_widget_->set_start_enabled(true);
  import_widget_->set_stop_enabled(false);
  export_in_progress_ = false;
  export_widget_->set_record_enabled(false);
  export_widget_->set_stop_enabled(false);
  on_metrics_updated(0.0);

  xy_raw_widget_->hide();
  xy_processed_widget_->hide();
  xz_processed_widget_->hide();
  yz_processed_widget_->hide();
}

void MainWindow::on_stop_pipeline_failure(const QString &error) {
  logger()->error("[MainWindow::on_stop_pipeline_failure]");
  pipeline_running_ = false;
  import_widget_->set_start_enabled(true);
  import_widget_->set_stop_enabled(false);
  export_in_progress_ = false;
  export_widget_->set_record_enabled(false);
  export_widget_->set_stop_enabled(false);
  on_metrics_updated(0.0);

  xy_raw_widget_->hide();
  xy_processed_widget_->hide();
  xz_processed_widget_->hide();
  yz_processed_widget_->hide();

  show_pipeline_error_popup(error);
}

void MainWindow::on_metrics_updated(double input_fps) {
  if (input_fps < 0.0) {
    input_fps = 0.0;
  }

  const QString text = QStringLiteral("%1 fps").arg(QString::number(input_fps, 'f', 1));
  monitor_widget_->set_input_throughput_fps(text);

  monitor_widget_->set_gpu_load("N/A");
  monitor_widget_->set_cpu_load("N/A");
  monitor_widget_->set_input_throughput_bytes("N/A");
  monitor_widget_->set_cpu_throughput("N/A");
  monitor_widget_->set_gpu_throughput("N/A");
  monitor_widget_->set_ram_usage("N/A");
  monitor_widget_->set_vram_usage("N/A");
  monitor_widget_->set_dropped_frames("N/A");
  monitor_widget_->set_pipeline_latency("N/A");
}

void MainWindow::on_update_pipeline_success() {
  logger()->info("[MainWindow::on_update_pipeline_success]");
  update_in_progress_ = false;
  import_widget_->set_start_enabled(false);
  export_widget_->set_record_enabled(!export_in_progress_ && export_widget_->isChecked());
  export_widget_->set_stop_enabled(export_in_progress_);

  auto dims = guess_source_dims();
  xy_raw_widget_->set_fixed_aspect(dims);
  if (render_widget_->get_space_transform() == "Fresnel Diffraction" &&
      render_widget_->get_image_mode() != "Raw") {
    dims = QSize(dims.width(), dims.width());
  }
  xy_processed_widget_->set_fixed_aspect(dims);
  if (view_widget_->is_raw_view_enabled()) {
    xy_raw_widget_->show();
  }
}

void MainWindow::on_update_pipeline_failure(const QString &error) {
  logger()->error("[MainWindow::on_update_pipeline_failure]");
  pipeline_running_   = false;
  update_in_progress_ = false;
  import_widget_->set_start_enabled(true);
  import_widget_->set_stop_enabled(false);
  export_in_progress_ = false;
  export_widget_->set_record_enabled(false);
  export_widget_->set_stop_enabled(false);

  show_pipeline_error_popup(tr("An error occurred while updating the pipeline:\n%1").arg(error));
}

void MainWindow::closeEvent(QCloseEvent *event) {
  if (pipeline_manager_ && import_widget_->is_stop_enabled()) {
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

  import_widget_->set_start_enabled(false);
  pipeline::Settings settings = get_pipeline_settings();
  auto               update   = [=]() { pipeline_manager_->update_pipeline(settings); };
  auto               start    = [=]() { pipeline_manager_->start_pipeline(); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, update, Qt::QueuedConnection));
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, start, Qt::QueuedConnection));
}

void MainWindow::on_import_stop_clicked() {
  HOLOVIBES_CHECK(pipeline_running_);
  import_widget_->set_stop_enabled(false);
  auto stop = [=]() { pipeline_manager_->stop_pipeline(); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, stop, Qt::QueuedConnection));
}

void MainWindow::on_export_record_clicked() {
  if (export_in_progress_) {
    logger()->warn("[MainWindow::on_export_record_clicked] Recording already in progress");
    return;
  }
  if (!pipeline_running_) {
    logger()->warn("[MainWindow::on_export_record_clicked] Pipeline is not running");
    return;
  }

  if (!validate_inputs()) {
    logger()->warn("[MainWindow::on_export_record_clicked] Validation failed");
    export_widget_->set_stop_enabled(pipeline_running_);
    return;
  }

  export_widget_->set_record_enabled(false);
  export_widget_->set_stop_enabled(false);

  std::filesystem::path record_path = makeRecordingPath(export_widget_->get_file_path());
  std::optional<size_t> frame_count;
  if (export_widget_->is_frame_count_enabled()) {
    frame_count = static_cast<size_t>(export_widget_->get_frame_count());
  }

  auto start = [mgr = pipeline_manager_, record_path]() { mgr->start_raw_record(record_path); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, start, Qt::QueuedConnection));
}

void MainWindow::on_export_stop_clicked() {
  if (!export_in_progress_) {
    logger()->warn("[MainWindow::on_export_stop_clicked] No active recording to stop");
    return;
  }

  export_widget_->set_record_enabled(false);
  auto stop = [mgr = pipeline_manager_]() { mgr->stop_raw_record(); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, stop, Qt::QueuedConnection));
}

void MainWindow::on_raw_record_started_success() {
  logger()->info("[MainWindow::on_raw_record_started_success]");
  export_in_progress_ = true;
  export_widget_->set_stop_enabled(true);
  export_widget_->set_record_enabled(false);
}

void MainWindow::on_raw_record_started_failure(const QString &error) {
  logger()->error("[MainWindow::on_raw_record_started_failure]");
  export_in_progress_ = false;
  export_widget_->set_record_enabled(pipeline_running_ && export_widget_->isChecked());
  export_widget_->set_stop_enabled(false);

  show_pipeline_error_popup(tr("An error occurred while starting raw recording:\n%1").arg(error));
}

void MainWindow::on_raw_record_stopped_success() {
  logger()->info("[MainWindow::on_raw_record_stopped_success]");
  export_in_progress_ = false;
  export_widget_->set_record_enabled(pipeline_running_ && export_widget_->isChecked());
  export_widget_->set_stop_enabled(false);
}

void MainWindow::on_raw_record_stopped_failure(const QString &error) {
  logger()->error("[MainWindow::on_raw_record_stopped_failure]");
  export_widget_->set_stop_enabled(pipeline_running_);
  show_pipeline_error_popup(tr("An error occurred while stopping raw recording:\n%1").arg(error));
}

bool MainWindow::validate_inputs() {
  const QString style_fail = QStringLiteral("background-color: rgba(255, 0, 0, 50);");

  // Clear all styles up front
  import_widget_->clear_validation_styles();
  export_widget_->clear_validation_styles();
  render_widget_->clear_validation_styles();
  view_widget_->clear_validation_styles();

  bool all_good = true;

  int  batch_size    = render_widget_->get_batch_size();
  int  time_stride   = render_widget_->get_time_stride();
  int  time_window   = render_widget_->get_time_window();
  int  p_frame_start = view_widget_->get_z_origin();
  int  p_frame_width = view_widget_->get_z_width();
  int  start_frame   = import_widget_->get_start_index();
  int  end_frame     = import_widget_->get_end_index();
  bool is_cam_mode   = import_widget_->is_camera_mode();

  // time_window divides time_stride
  if (time_window <= 0 || (time_stride % time_window != 0)) {
    all_good = false;
    render_widget_->mark_time_window_invalid();
    render_widget_->mark_time_stride_invalid();
  }

  // p_frame_start + p_frame_width <= time_window
  if (!(p_frame_start + p_frame_width <= time_window)) {
    all_good = false;
    render_widget_->mark_time_window_invalid();
    view_widget_->mark_z_invalid();
    view_widget_->mark_z_width_invalid();
  }

  // end_frame > start_frame
  if (!(is_cam_mode || end_frame > start_frame)) {
    all_good = false;
    import_widget_->mark_start_index_invalid();
    import_widget_->mark_end_index_invalid();
  }

  // time_stride <= end_frame - start_frame
  if (!(is_cam_mode || time_stride <= (end_frame - start_frame))) {
    all_good = false;
    import_widget_->mark_start_index_invalid();
    import_widget_->mark_end_index_invalid();
    render_widget_->mark_time_stride_invalid();
  }

  // import path non-empty
  if (!is_cam_mode && import_widget_->get_file_path().isEmpty()) {
    all_good = false;
    import_widget_->mark_file_invalid();
  }

  // export path non-empty
  if (export_widget_->get_file_path().isEmpty()) {
    all_good = false;
    export_widget_->mark_file_invalid();
  }

  // if exporting frames, ensure divisible by batch_size
  if (export_widget_->is_frame_count_enabled()) {
    int frame_count = export_widget_->get_frame_count() % batch_size;
    if (frame_count != 0) {
      all_good = false;
      export_widget_->mark_frames_invalid();
      render_widget_->mark_batch_size_invalid();
    }
  }

  return all_good;
}

void MainWindow::setup_validation_connections() {
  bool (MainWindow::*cb)() = &MainWindow::validate_inputs;

  // Connect to widget signals
  connect(import_widget_, &ImportWidget::settings_changed, this, cb);
  connect(export_widget_, &ExportWidget::settings_changed, this, cb);
  connect(render_widget_, &ImageRenderingWidget::settings_changed, this, cb);
  connect(view_widget_, &ViewWidget::settings_changed, this, cb);
}

void MainWindow::setup_update_connections() {
  void (MainWindow::*cb)() = &MainWindow::update_if_running;

  connect(import_widget_, &ImportWidget::settings_changed, this, cb);
  connect(export_widget_, &ExportWidget::settings_changed, this, cb);
  connect(render_widget_, &ImageRenderingWidget::settings_changed, this, cb);
  connect(view_widget_, &ViewWidget::settings_changed, this, cb);
}

void MainWindow::update_if_running() {
  if (!pipeline_manager_ || !import_widget_->is_stop_enabled()) {
    return;
  }

  if (update_in_progress_) {
    return;
  }

  if (!validate_inputs()) {
    return;
  }

  update_in_progress_ = true;
  import_widget_->set_start_enabled(false);
  pipeline::Settings settings = get_pipeline_settings();
  auto               update   = [=]() { pipeline_manager_->update_pipeline(settings); };
  HOLOVIBES_CHECK(QMetaObject::invokeMethod(pipeline_manager_, update, Qt::QueuedConnection));
}

QSize MainWindow::guess_source_dims() {
  if (!import_widget_->is_camera_mode()) {
    auto header     = holofile::Reader(import_widget_->get_file_path().toStdString()).header();
    int  src_width  = header.frame_width;
    int  src_height = header.frame_height;
    return QSize(src_width, src_height);
  }

  else if (import_widget_->is_camera_mode()) {
    auto path     = get_selected_camera_config_path();
    auto cfg_file = std::ifstream(path);
    if (!cfg_file.is_open()) {
      throw std::runtime_error(std::format("Could not open camera config file: {}", path));
    }

    // FIXME: This needs to be properly handeled for more camera support
    auto cfg_json   = nlohmann::json::parse(cfg_file);
    auto cfg        = cfg_json.contains("s711") ? cfg_json.at("s711") : cfg_json.at("s710");
    int  src_width  = cfg.at("Width").get<int>();
    int  src_height = cfg.at("Height").get<int>();
    return QSize(src_width, src_height);
  }

  HOLOVIBES_UNIMPLEMENTED();
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
        {"Ametek S711 Euresys Coaxlink Octo", ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP},
    };

    if (!import_widget_->is_camera_mode()) {
      s.import_source = ImportSource::HOLOFILE;
      s.load_path     = import_widget_->get_file_path().toStdString();
      s.load_begin    = static_cast<size_t>(import_widget_->get_start_index());
      s.load_end      = static_cast<size_t>(import_widget_->get_end_index());
      QString method  = import_widget_->get_load_method();
      s.load_method   = method_from_str.at(method.toStdString());
      s.load_batch    = render_widget_->get_batch_size();
    } else {
      QString source       = import_widget_->get_camera_type();
      s.import_source      = source_from_str.at(source.toStdString());
      s.camera_config_path = get_selected_camera_config_path();
      s.load_batch         = 64;
    }
  }

  // Image Rendering Settings
  {
    s.view_type = render_widget_->get_image_mode() == "Raw" ? ViewType::RAW : ViewType::PROCESSED;

    std::map<std::string, SpacialMethod> method_from_str{
        {"None", SpacialMethod::NONE},
        {"Fresnel Diffraction", SpacialMethod::FRESNEL_DIFFRACTION},
        {"Angular Spectrum", SpacialMethod::ANGULAR_SPECTRUM},
    };
    QString method       = render_widget_->get_space_transform();
    s.spacial_method     = method_from_str.at(method.toStdString());
    s.spacial_lambda     = static_cast<float>(render_widget_->get_lambda()) * 1e-9f;
    s.spacial_z          = static_cast<float>(render_widget_->get_focus()) * 1e-3f;
    s.spacial_pixel_size = 20e-6f; // TODO: get from camera
  }
  {
    s.filter_2d           = render_widget_->is_filter_2d_enabled();
    s.filter_r_inner      = render_widget_->get_filter_inner();
    s.filter_r_outer      = render_widget_->get_filter_outer();
    s.filter_smooth_inner = 0;
    s.filter_smooth_outer = 1;
  }
  {
    std::map<std::string, TimeMethod> method_from_str{
        {"None", TimeMethod::NONE},
        {"Principal Component Analysis", TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS},
        {"Short Time Fourier", TimeMethod::SHORT_TIME_FOURIER},
    };
    s.time_window  = render_widget_->get_time_window();
    s.time_stride  = render_widget_->get_time_stride();
    QString method = render_widget_->get_time_transform();
    s.time_method  = method_from_str.at(method.toStdString());
    s.time_x_begin = view_widget_->get_x_origin();
    s.time_x_end   = s.time_x_begin + view_widget_->get_x_width();
    s.time_y_begin = view_widget_->get_y_origin();
    s.time_y_end   = s.time_y_begin + view_widget_->get_y_width();
    s.time_z_begin = view_widget_->get_z_origin();
    s.time_z_end   = s.time_z_begin + view_widget_->get_z_width();
  }

  // View Settings
  {
    s.view_3d_cuts = view_widget_->is_cuts_3d_enabled();
    s.raw_view     = view_widget_->is_raw_view_enabled();
  }

  // Post-processing Settings
  {
    QString     appDataBase = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QString     appDataPath = appDataBase + "/" + QCoreApplication::applicationVersion();
    QString     convolutionsKernelsPath = appDataPath + "/" + "convolution_kernels/";
    std::string kernel_path             = convolutionsKernelsPath.toStdString() +
                              render_widget_->get_convolution().toStdString() + ".json";

    s.pp_fps                 = 60;
    s.pp_fft_shift           = view_widget_->is_fft_shift_enabled();
    s.pp_accumulation        = static_cast<size_t>(view_widget_->get_accumulation());
    s.pp_convolution         = render_widget_->get_convolution() != "None";
    s.pp_convolution_path    = kernel_path;
    s.pp_convolution_divide  = render_widget_->is_convolution_divide();
    s.pp_pctclip             = view_widget_->is_pct_enabled();
    s.pp_pctclip_lower       = 0.02f;
    s.pp_pctclip_upper       = 99.98f;
    s.pp_pctclip_radius      = view_widget_->get_pct_radius();
    s.pp_registration        = view_widget_->is_registration_enabled();
    s.pp_registration_radius = view_widget_->get_registration_radius();
  }

  // Recording Settings
  {
    if (export_widget_->isChecked()) {

      s.recording_method = export_widget_->get_image_type() == "Raw Image"
                               ? RecordingMethod::RAW
                               : RecordingMethod::PROCESSED;
    } else {
      s.recording_method = RecordingMethod::NONE;
    }
    s.recording_path  = export_widget_->get_file_path().toStdString();
    s.recording_count = export_widget_->get_frame_count();
  }

  // Auto-Focus Settings
  {
    s.autofocus_enabled   = render_widget_->autofocus_widget()->is_enabled();
    s.autofocus_nb_subaps = render_widget_->autofocus_widget()->get_nb_subaps();
    s.autofocus_max_iter  = render_widget_->autofocus_widget()->get_max_iter();
    s.autofocus_tolerance = render_widget_->autofocus_widget()->get_tolerance();
    s.autofocus_gain      = render_widget_->autofocus_widget()->get_gain();
  }

  return s;
}

void MainWindow::set_pipeline_settings(const pipeline::Settings &s) {
  using namespace holovibes::pipeline;

  // --- Advanced Settings ---
  {
    // (none exposed in UI)
  }

  // --- Import Settings ---
  {
    // Source: camera or file
    if (s.import_source == ImportSource::HOLOFILE) {
      import_widget_->set_camera_mode(false);
      import_widget_->set_file_path(QString::fromStdString(s.load_path.string()));
      import_widget_->set_start_index(static_cast<int>(s.load_begin));
      import_widget_->set_end_index(static_cast<int>(s.load_end));

      QString method;
      switch (s.load_method) {
      case LoadMethod::READ_LIVE:
        method = "Read Live";
        break;
      case LoadMethod::LOAD_IN_CPU:
        method = "Load in CPU RAM";
        break;
      case LoadMethod::LOAD_IN_GPU:
        method = "Load in GPU RAM";
        break;
      default:
        method = "Read Live";
        break;
      }
      import_widget_->set_load_method(method);
    } else {
      import_widget_->set_camera_mode(true);

      QString source;
      switch (s.import_source) {
      case ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO:
        source = "Ametek S710 Euresys Coaxlink Octo";
        break;
      case ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP:
        source = "Ametek S711 Euresys Coaxlink QSFP+";
        break;
      default:
        source = "Ametek S710 Euresys Coaxlink Octo";
        break;
      }
      import_widget_->set_camera_type(source);
    }
  }

  // --- Image Rendering Settings ---
  {
    QString method;
    switch (s.spacial_method) {
    case SpacialMethod::FRESNEL_DIFFRACTION:
      method = "Fresnel Diffraction";
      break;
    case SpacialMethod::ANGULAR_SPECTRUM:
      method = "Angular Spectrum";
      break;
    default:
      method = "None";
      break;
    }
    render_widget_->set_batch_size(s.load_batch);
    render_widget_->set_space_transform(method);
    render_widget_->set_lambda(s.spacial_lambda * 1e9); // nm
    render_widget_->set_focus(s.spacial_z * 1e3);       // mm
  }
  {
    render_widget_->set_filter_2d_enabled(s.filter_2d);
    render_widget_->set_filter_inner(s.filter_r_inner);
    render_widget_->set_filter_outer(s.filter_r_outer);
    // smooth_inner/smooth_outer omitted since not exposed in UI
  }
  {
    QString method;
    switch (s.time_method) {
    case TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS:
      method = "Principal Component Analysis";
      break;
    case TimeMethod::SHORT_TIME_FOURIER:
      method = "Short Time Fourier";
      break;
    default:
      method = "None";
      break;
    }
    render_widget_->set_time_transform(method);
    render_widget_->set_time_window(static_cast<int>(s.time_window));
    render_widget_->set_time_stride(static_cast<int>(s.time_stride));

    view_widget_->set_x_origin(static_cast<int>(s.time_x_begin));
    view_widget_->set_x_width(static_cast<int>(s.time_x_end - s.time_x_begin));
    view_widget_->set_y_origin(static_cast<int>(s.time_y_begin));
    view_widget_->set_y_width(static_cast<int>(s.time_y_end - s.time_y_begin));
    view_widget_->set_z_origin(static_cast<int>(s.time_z_begin));
    view_widget_->set_z_width(static_cast<int>(s.time_z_end - s.time_z_begin));
  }

  // --- View Settings ---
  { view_widget_->set_cuts_3d_enabled(s.view_3d_cuts); }

  // --- Post-processing Settings ---
  {
    view_widget_->set_fft_shift_enabled(s.pp_fft_shift);
    view_widget_->set_accumulation(static_cast<int>(s.pp_accumulation));
    render_widget_->set_convolution_divide(s.pp_convolution_divide);

    // Select convolution kernel name from path
    QString convName = "None";
    if (s.pp_convolution && !s.pp_convolution_path.empty()) {
      auto fileName = QFileInfo(QString::fromStdString(s.pp_convolution_path)).baseName();
      convName      = fileName;
    }
    render_widget_->set_convolution(convName);

    view_widget_->set_pct_radius(s.pp_pctclip_radius);
    view_widget_->set_pct_enabled(s.pp_pctclip);
    view_widget_->set_registration_enabled(s.pp_registration);
    view_widget_->set_registration_radius(s.pp_registration_radius);
  }

  // --- Recording Settings ---
  {
    export_widget_->set_file_path(QString::fromStdString(s.recording_path.string()));
    export_widget_->set_frame_count(static_cast<int>(s.recording_count));
    // recording_method not exposed (always RAW in get_pipeline_settings)
  }
}

std::string MainWindow::get_selected_camera_config_path() {
  QString     appDataBase      = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
  QString     appDataPath      = appDataBase + "/" + QCoreApplication::applicationVersion();
  QString     cameraConfigPath = appDataPath + "/" + "camera_configs/";
  std::string config_path =
      cameraConfigPath.toStdString() + import_widget_->get_camera_config().toStdString() + ".json";
  return config_path;
}

} // namespace holovibes::ui
