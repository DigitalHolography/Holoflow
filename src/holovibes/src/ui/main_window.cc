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
#include <array>
#include <filesystem>
#include <fstream>
#include <optional>

#include "bug.hh"
#include "holofile/holofile.hh"
#include "logger.hh"
#include "settings_loader.hh"
#include "ui/widgets/tensor_display_widget.hh"

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

struct FieldBinding {
  holovibes::pipeline::SettingsField field;
  QWidget                           *widget;
};

QString to_qstring(holovibes::pipeline::ValidationSeverity severity) {
  switch (severity) {
  case holovibes::pipeline::ValidationSeverity::Warning:
    return "Warning";
  case holovibes::pipeline::ValidationSeverity::Error:
    return "Error";
  }

  HOLOVIBES_UNREACHABLE();
}

QString build_field_tooltip(const holovibes::pipeline::FieldHelp                        &help,
                            std::span<const holovibes::pipeline::ValidationIssue *const> issues) {
  QStringList lines;
  lines << QString::fromStdString(help.title);
  lines << "";
  lines << QString::fromStdString(help.description);

  if (!help.constraints.empty()) {
    lines << "";
    lines << "Constraints:";
    for (const auto *constraint : help.constraints) {
      lines << QString("- %1").arg(constraint);
    }
  }

  if (!issues.empty()) {
    lines << "";
    lines << "Current issues:";
    for (const auto *issue : issues) {
      lines << QString("- %1: %2")
                   .arg(to_qstring(issue->severity), QString::fromStdString(issue->message));
    }
  }

  return lines.join('\n');
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
  connect_export_controls();
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
  xy_raw_widget_               = new TensorDisplayWidget(nullptr);
  xy_processed_widget_         = new TensorDisplayWidget(nullptr);
  xz_processed_widget_         = new TensorDisplayWidget(nullptr);
  yz_processed_widget_         = new TensorDisplayWidget(nullptr);
  raw_spectrum_widget_         = new TensorDisplayWidget(nullptr);
  processed_spectrum_widget_   = new TensorDisplayWidget(nullptr);
  shack_hartmann_widget_       = new TensorDisplayWidget(nullptr);
  shack_hartmann_xcorr_widget_ = new TensorDisplayWidget(nullptr);
  zernike_phase_widget_        = new TensorDisplayWidget(nullptr);

  xy_raw_widget_->setWindowTitle("XY-Raw");
  xy_processed_widget_->setWindowTitle("XY-Processed");
  xz_processed_widget_->setWindowTitle("XZ-Processed");
  yz_processed_widget_->setWindowTitle("YZ-Processed");
  raw_spectrum_widget_->setWindowTitle("Raw Spectrum");
  processed_spectrum_widget_->setWindowTitle("Processed Spectrum");
  shack_hartmann_widget_->setWindowTitle("Shack Hartmann");
  shack_hartmann_xcorr_widget_->setWindowTitle("Shack Hartmann XCorr");
  zernike_phase_widget_->setWindowTitle("Zernike Phase");

  zernike_phase_widget_->set_colormap(Colormap::Twilight);
  zernike_phase_widget_->set_value_range(0.0f, 2 * static_cast<float>(M_PI));

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

  connect(view_widget_, &ViewWidget::raw_spectrum_view_toggled, this, [this](bool checked) {
    if (pipeline_running_ && checked) {
      raw_spectrum_widget_->show();
    } else {
      raw_spectrum_widget_->hide();
    }
  });

  connect(view_widget_, &ViewWidget::process_spectrum_view_toggled, this, [this](bool checked) {
    if (pipeline_running_ && checked) {
      processed_spectrum_widget_->show();
    } else {
      processed_spectrum_widget_->hide();
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
  pipeline_manager_ = new pipeline::Manager(
      render_widget_->autofocus_widget(), xy_processed_widget_, xz_processed_widget_,
      yz_processed_widget_, xy_raw_widget_, raw_spectrum_widget_, processed_spectrum_widget_,
      shack_hartmann_widget_, shack_hartmann_xcorr_widget_, zernike_phase_widget_);
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
        prev_settings.view_type          = pipeline::ViewType::PROCESSED;
        pipeline::Settings new_settings =
            pipeline::old_json_to_settings(footer.pipeline_settings, prev_settings);
        set_pipeline_settings(new_settings);
      }
    } catch (std::exception &e) {
      logger()->error("failed to open \"{}\": \"{}\"", file.toStdString(), e.what());
    }
  });
}

void MainWindow::connect_export_controls() {

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

  if (render_widget_->get_image_mode() == "Raw") {
    xy_processed_widget_->setWindowTitle("XY-Raw");
  } else {
    xy_processed_widget_->setWindowTitle("XY-Processed");
  }

  auto dims = guess_source_dims();
  xy_raw_widget_->set_fixed_aspect(dims);
  if (render_widget_->get_space_transform() == "Fresnel Diffraction" &&
      render_widget_->get_image_mode() != "Raw") {
    dims = QSize(dims.width(), dims.width());
  }
  xy_processed_widget_->set_fixed_aspect(dims);
  xy_raw_widget_->set_fixed_aspect(guess_source_dims());
  shack_hartmann_widget_->set_fixed_aspect(dims);
  shack_hartmann_xcorr_widget_->set_fixed_aspect(dims);
  zernike_phase_widget_->set_fixed_aspect(guess_source_dims());

  xy_processed_widget_->show();

  auto *autofocus_widget = render_widget_->autofocus_widget();
  if (autofocus_widget->is_enabled()) {
    const bool has_enabled_zernike =
        autofocus_widget->is_z2_enabled() || autofocus_widget->is_z3_enabled() ||
        autofocus_widget->is_z4_enabled() || autofocus_widget->is_z5_enabled() ||
        autofocus_widget->is_z6_enabled() || autofocus_widget->is_z7_enabled() ||
        autofocus_widget->is_z8_enabled() || autofocus_widget->is_z9_enabled() ||
        autofocus_widget->is_z10_enabled();
    if (!has_enabled_zernike) {
      autofocus_widget->reset_zernike_values();
    }

    if (autofocus_widget->show_shack_hartmann_sensor_view()) {
      shack_hartmann_widget_->show();
    } else {
      shack_hartmann_widget_->hide();
    }

    if (autofocus_widget->show_cross_correlation_view()) {
      shack_hartmann_xcorr_widget_->show();
    } else {
      shack_hartmann_xcorr_widget_->hide();
    }

    if (autofocus_widget->show_reconstructed_phase()) {
      zernike_phase_widget_->show();
    } else {
      zernike_phase_widget_->hide();
    }
  } else {
    shack_hartmann_widget_->hide();
    shack_hartmann_xcorr_widget_->hide();
    zernike_phase_widget_->hide();
    autofocus_widget->reset_zernike_values();
  }

  if (view_widget_->is_raw_view_enabled()) {
    xy_raw_widget_->show();
  }

  if (view_widget_->is_raw_spectrum_view_enabled()) {
    raw_spectrum_widget_->show();
  }

  if (view_widget_->is_process_spectrum_view_enabled()) {
    processed_spectrum_widget_->show();
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
  raw_spectrum_widget_->hide();
  processed_spectrum_widget_->hide();
  shack_hartmann_widget_->hide();
  shack_hartmann_xcorr_widget_->hide();
  zernike_phase_widget_->hide();
  render_widget_->autofocus_widget()->reset_zernike_values();
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
  raw_spectrum_widget_->hide();
  processed_spectrum_widget_->hide();
  render_widget_->autofocus_widget()->reset_zernike_values();

  show_pipeline_error_popup(error);
}

void MainWindow::on_metrics_updated(double input_fps) {
  if (input_fps < 0.0) {
    input_fps = 0.0;
  }

  int fps = static_cast<int>(input_fps);

  const QString text = QString("%1 fps").arg(fps, 6, 10, QChar('0'));

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

  if (render_widget_->get_image_mode() == "Raw") {
    xy_processed_widget_->setWindowTitle("XY-Raw");
  } else {
    xy_processed_widget_->setWindowTitle("XY-Processed");
  }

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

  shack_hartmann_widget_->set_fixed_aspect(dims);
  shack_hartmann_xcorr_widget_->set_fixed_aspect(dims);
  zernike_phase_widget_->set_fixed_aspect(guess_source_dims());

  auto *autofocus_widget = render_widget_->autofocus_widget();
  if (autofocus_widget->is_enabled()) {
    const bool has_enabled_zernike =
        autofocus_widget->is_z2_enabled() || autofocus_widget->is_z3_enabled() ||
        autofocus_widget->is_z4_enabled() || autofocus_widget->is_z5_enabled() ||
        autofocus_widget->is_z6_enabled() || autofocus_widget->is_z7_enabled() ||
        autofocus_widget->is_z8_enabled() || autofocus_widget->is_z9_enabled() ||
        autofocus_widget->is_z10_enabled();
    if (!has_enabled_zernike) {
      autofocus_widget->reset_zernike_values();
    }

    if (autofocus_widget->show_shack_hartmann_sensor_view()) {
      shack_hartmann_widget_->show();
    } else {
      shack_hartmann_widget_->hide();
    }

    if (autofocus_widget->show_cross_correlation_view()) {
      shack_hartmann_xcorr_widget_->show();
    } else {
      shack_hartmann_xcorr_widget_->hide();
    }

    if (autofocus_widget->show_reconstructed_phase()) {
      zernike_phase_widget_->show();
    } else {
      zernike_phase_widget_->hide();
    }
  } else {
    shack_hartmann_widget_->hide();
    shack_hartmann_xcorr_widget_->hide();
    zernike_phase_widget_->hide();
    autofocus_widget->reset_zernike_values();
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
  if (xy_processed_widget_) {
    if (xy_processed_widget_->isVisible())
      xy_processed_widget_->hide();
    xy_processed_widget_->deleteLater();
  }

  if (xz_processed_widget_) {
    if (xz_processed_widget_->isVisible())
      xz_processed_widget_->hide();
    xz_processed_widget_->deleteLater();
  }

  if (yz_processed_widget_) {
    if (yz_processed_widget_->isVisible())
      yz_processed_widget_->hide();
    yz_processed_widget_->deleteLater();
  }

  if (xy_raw_widget_) {
    if (xy_raw_widget_->isVisible())
      xy_raw_widget_->hide();
    xy_raw_widget_->deleteLater();
  }

  if (raw_spectrum_widget_) {
    if (raw_spectrum_widget_->isVisible())
      raw_spectrum_widget_->hide();
    raw_spectrum_widget_->deleteLater();
  }

  if (processed_spectrum_widget_) {
    if (processed_spectrum_widget_->isVisible())
      processed_spectrum_widget_->hide();
    processed_spectrum_widget_->deleteLater();
  }

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
  import_widget_->clear_validation_styles();
  export_widget_->clear_validation_styles();
  render_widget_->clear_validation_styles();
  view_widget_->clear_validation_styles();
  render_widget_->autofocus_widget()->clear_validation_styles();

  pipeline::Settings settings = get_pipeline_settings();
  const auto result = pipeline::validate_settings(settings, build_validation_context(settings));
  refresh_validation_tooltips(result);
  apply_validation_result(result);
  return result.ok();
}

void MainWindow::apply_validation_result(const pipeline::ValidationResult &result) {
  using pipeline::SettingsField;

  for (const auto &issue : result.issues) {
    for (const auto field : issue.fields) {
      switch (field) {
      case SettingsField::LoadPath:
        import_widget_->mark_file_invalid();
        break;
      case SettingsField::CameraConfigPath:
        import_widget_->mark_camera_config_invalid();
        break;
      case SettingsField::LoadBegin:
        import_widget_->mark_start_index_invalid();
        break;
      case SettingsField::LoadEnd:
        import_widget_->mark_end_index_invalid();
        break;
      case SettingsField::LoadBatch:
        render_widget_->mark_batch_size_invalid();
        break;
      case SettingsField::SpacialMethod:
        render_widget_->mark_space_transform_invalid();
        break;
      case SettingsField::TimeMethod:
        render_widget_->mark_time_transform_invalid();
        break;
      case SettingsField::TimeWindow:
        render_widget_->mark_time_window_invalid();
        break;
      case SettingsField::TimeStride:
        render_widget_->mark_time_stride_invalid();
        break;
      case SettingsField::TimeZBegin:
        view_widget_->mark_z_invalid();
        break;
      case SettingsField::TimeZEnd:
        view_widget_->mark_z_width_invalid();
        break;
      case SettingsField::View3DCuts:
        view_widget_->mark_cuts_3d_invalid();
        break;
      case SettingsField::ViewRawSpectrum:
        view_widget_->mark_raw_spectrum_invalid();
        break;
      case SettingsField::ViewProcessedSpectrum:
        view_widget_->mark_processed_spectrum_invalid();
        break;
      case SettingsField::PpConvolution:
        render_widget_->mark_convolution_invalid();
        break;
      case SettingsField::PpRegistration:
        view_widget_->mark_registration_invalid();
        break;
      case SettingsField::RecordingPath:
        export_widget_->mark_file_invalid();
        break;
      case SettingsField::RecordingCount:
        export_widget_->mark_frames_invalid();
        break;
      case SettingsField::AutofocusNbSubaps:
        render_widget_->autofocus_widget()->mark_nb_subaps_invalid();
        break;
      }
    }
  }
}

void MainWindow::refresh_validation_tooltips(const pipeline::ValidationResult &result) {
  using pipeline::SettingsField;

  const std::array bindings = {
      FieldBinding{SettingsField::LoadPath, import_widget_->file_line_edit()},
      FieldBinding{SettingsField::CameraConfigPath, import_widget_->camera_config_combo()},
      FieldBinding{SettingsField::LoadBegin, import_widget_->start_index_spin()},
      FieldBinding{SettingsField::LoadEnd, import_widget_->end_index_spin()},
      FieldBinding{SettingsField::LoadBatch, render_widget_->batch_size_spin()},
      FieldBinding{SettingsField::SpacialMethod, render_widget_->space_transform_combo()},
      FieldBinding{SettingsField::TimeMethod, render_widget_->time_transform_combo()},
      FieldBinding{SettingsField::TimeWindow, render_widget_->time_window_spin()},
      FieldBinding{SettingsField::TimeStride, render_widget_->time_stride_spin()},
      FieldBinding{SettingsField::TimeZBegin, view_widget_->z_spin()},
      FieldBinding{SettingsField::TimeZEnd, view_widget_->z_width_spin()},
      FieldBinding{SettingsField::View3DCuts, view_widget_->cuts_3d_check()},
      FieldBinding{SettingsField::ViewRawSpectrum, view_widget_->raw_spectrum_view_check()},
      FieldBinding{SettingsField::ViewProcessedSpectrum,
                   view_widget_->process_spectrum_view_check()},
      FieldBinding{SettingsField::PpConvolution, render_widget_->convolution_combo()},
      FieldBinding{SettingsField::PpRegistration, view_widget_->registration_check()},
      FieldBinding{SettingsField::RecordingPath, export_widget_->file_line_edit()},
      FieldBinding{SettingsField::RecordingCount, export_widget_->frames_spin()},
      FieldBinding{SettingsField::AutofocusNbSubaps,
                   render_widget_->autofocus_widget()->nb_subaps_spin()},
  };

  for (const auto &binding : bindings) {
    const auto  issues  = result.issues_for(binding.field);
    const auto &help    = pipeline::get_field_help(binding.field);
    const auto  tooltip = build_field_tooltip(help, std::span{issues});
    binding.widget->setToolTip(tooltip);
  }
}

pipeline::ValidationContext
MainWindow::build_validation_context(const pipeline::Settings &settings) const {
  pipeline::ValidationContext context;

  if (settings.import_source == pipeline::ImportSource::HOLOFILE) {
    context.load_path_exists =
        !settings.load_path.empty() && std::filesystem::exists(settings.load_path);

    if (context.load_path_exists) {
      try {
        auto header                = holofile::Reader(settings.load_path.string()).header();
        context.source_width       = static_cast<int>(header.frame_width);
        context.source_height      = static_cast<int>(header.frame_height);
        context.source_frame_count = static_cast<int>(header.frame_count);
      } catch (...) {
        context.load_path_exists = false;
      }
    }
  } else {
    context.camera_config_exists = !settings.camera_config_path.empty() &&
                                   std::filesystem::exists(settings.camera_config_path);

    if (context.camera_config_exists) {
      try {
        std::ifstream cfg_file(settings.camera_config_path);
        auto          cfg_json = nlohmann::json::parse(cfg_file);
        const auto   &cfg = cfg_json.contains("s711") ? cfg_json.at("s711") : cfg_json.at("s710");

        context.source_width        = cfg.at("Width").get<int>();
        context.source_height       = cfg.at("Height").get<int>();
        context.camera_config_valid = true;
      } catch (...) {
        context.camera_config_valid = false;
      }
    }
  }

  if (settings.recording_method != pipeline::RecordingMethod::NONE) {
    context.recording_path_error = pipeline::validate_recording_path(settings.recording_path);
  }

  return context;
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
    s.cpu_out_size = 64;
    s.gpu_out_size = 64;
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
        {"Ametek S711 Euresys Coaxlink QSFP+", ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP},
    };

    if (!import_widget_->is_camera_mode()) {
      s.import_source = ImportSource::HOLOFILE;
      s.load_path     = import_widget_->get_file_path().toStdString();
      s.load_begin    = static_cast<size_t>(import_widget_->get_start_index());
      s.load_end      = static_cast<size_t>(import_widget_->get_end_index());
      QString method  = import_widget_->get_load_method();
      s.load_method   = method_from_str.at(method.toStdString());
      if (render_widget_->get_time_transform() == "Principal Component Analysis") {
        s.load_batch = 32;
      } else {
        s.load_batch = render_widget_->get_batch_size();
      }
    } else {
      QString source       = import_widget_->get_camera_type();
      s.import_source      = source_from_str.at(source.toStdString());
      s.camera_config_path = get_selected_camera_config_path();
      s.load_batch         = 1;

      std::ifstream cfg_file(s.camera_config_path);
      if (cfg_file.is_open()) {
        try {
          auto cfg_json = nlohmann::json::parse(cfg_file);
          s.load_batch  = cfg_json.contains("s711")
                              ? cfg_json.at("s711").at("BufferPartCount").get<int>()
                              : cfg_json.at("s710").at("BufferPartCount").get<int>();
        } catch (...) {
        }
      }
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
        {"RFFT", TimeMethod::RFFT},
        {"FFT", TimeMethod::FFT},
    };
    s.time_window       = render_widget_->get_time_window();
    s.time_stride       = render_widget_->get_time_stride();
    s.time_accumulation = 4; // TODO: expose in UI
    QString method      = render_widget_->get_time_transform();
    s.time_method       = method_from_str.at(method.toStdString());
    s.time_x_begin      = view_widget_->get_x_origin();
    s.time_x_end        = s.time_x_begin + view_widget_->get_x_width();
    s.time_y_begin      = view_widget_->get_y_origin();
    s.time_y_end        = s.time_y_begin + view_widget_->get_y_width();
    s.time_z_begin      = view_widget_->get_z_origin();
    s.time_z_end        = s.time_z_begin + view_widget_->get_z_width();
  }

  // View Settings
  {
    std::map<std::string, MomentType> moment_from_str{
        {"M0", MomentType::M0},
        {"M1", MomentType::M1},
        {"M2", MomentType::M2},
    };
    s.view_3d_cuts            = view_widget_->is_cuts_3d_enabled();
    s.raw_view                = view_widget_->is_raw_view_enabled();
    s.moment_type             = moment_from_str.at(view_widget_->get_image_type().toStdString());
    s.view_raw_spectrum       = true;
    s.view_processed_spectrum = true;
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
    s.autofocus_enabled        = render_widget_->autofocus_widget()->is_enabled();
    s.autofocus_nb_subaps      = render_widget_->autofocus_widget()->get_nb_subaps();
    s.autofocus_zernike_orders = std::vector<int>();

    if (render_widget_->autofocus_widget()->is_z2_enabled()) {
      s.autofocus_zernike_orders.push_back(2);
    }

    if (render_widget_->autofocus_widget()->is_z3_enabled()) {
      s.autofocus_zernike_orders.push_back(3);
    }

    if (render_widget_->autofocus_widget()->is_z4_enabled()) {
      s.autofocus_zernike_orders.push_back(4);
    }

    if (render_widget_->autofocus_widget()->is_z5_enabled()) {
      s.autofocus_zernike_orders.push_back(5);
    }

    if (render_widget_->autofocus_widget()->is_z6_enabled()) {
      s.autofocus_zernike_orders.push_back(6);
    }

    if (render_widget_->autofocus_widget()->is_z7_enabled()) {
      s.autofocus_zernike_orders.push_back(7);
    }

    if (render_widget_->autofocus_widget()->is_z8_enabled()) {
      s.autofocus_zernike_orders.push_back(8);
    }

    if (render_widget_->autofocus_widget()->is_z9_enabled()) {
      s.autofocus_zernike_orders.push_back(9);
    }

    if (render_widget_->autofocus_widget()->is_z10_enabled()) {
      s.autofocus_zernike_orders.push_back(10);
    }
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
    case TimeMethod::RFFT:
      method = "RFFT";
      break;
    case TimeMethod::FFT:
      method = "FFT";
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
  {
    view_widget_->set_cuts_3d_enabled(s.view_3d_cuts);
  }

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
