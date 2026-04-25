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

#include "ui/widgets/import_widget.hh"
#include "ui/widgets/validation_style.hh"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QStackedLayout>
#include <QStandardPaths>
#include <QVBoxLayout>

namespace holovibes::ui {

namespace {

QSpinBox *create_spin_box(QWidget *parent, int minimum, int maximum, int value) {
  auto *spin_box = new QSpinBox(parent);
  spin_box->setRange(minimum, maximum);
  spin_box->setValue(value);
  return spin_box;
}

QComboBox *create_combo_box(QWidget *parent, const QStringList &items) {
  auto *combo_box = new QComboBox(parent);
  combo_box->addItems(items);
  return combo_box;
}

constexpr int kLargeSpinMax = 1024 * 1024;

} // namespace

ImportWidget::ImportWidget(QWidget *parent) : QGroupBox("IMPORT", parent) {
  setup_ui();
  connect_signals();
}

bool               ImportWidget::is_camera_mode() const { return cam_check_->isChecked(); }
QString            ImportWidget::get_file_path() const { return file_line_edit_->text(); }
std::optional<int> ImportWidget::get_fps_limit() const {
  if (fps_spin_->value() <= 0) {
    return std::nullopt;
  }

  return fps_spin_->value();
}
int     ImportWidget::get_start_index() const { return start_index_spin_->value(); }
int     ImportWidget::get_end_index() const { return end_index_spin_->value(); }
QString ImportWidget::get_load_method() const { return load_method_combo_->currentText(); }
QString ImportWidget::get_camera_type() const { return camera_combo_->currentText(); }
QString ImportWidget::get_camera_config() const { return camera_config_combo_->currentText(); }

void ImportWidget::set_fps_limit(std::optional<int> value) {
  fps_spin_->setValue(value.value_or(0));
}
void ImportWidget::set_start_index(int value) { start_index_spin_->setValue(value); }
void ImportWidget::set_end_index(int value) { end_index_spin_->setValue(value); }
void ImportWidget::set_end_index_range(int min, int max) { end_index_spin_->setRange(min, max); }
void ImportWidget::set_load_method(const QString &method) {
  load_method_combo_->setCurrentText(method);
}
void ImportWidget::set_camera_type(const QString &type) { camera_combo_->setCurrentText(type); }
void ImportWidget::set_camera_mode(bool enabled) { cam_check_->setChecked(enabled); }

void ImportWidget::set_start_enabled(bool enabled) { start_button_->setEnabled(enabled); }
void ImportWidget::set_stop_enabled(bool enabled) { stop_button_->setEnabled(enabled); }
bool ImportWidget::is_stop_enabled() const { return stop_button_->isEnabled(); }

void ImportWidget::mark_file_invalid() { mark_validation_error(file_line_edit_); }
void ImportWidget::mark_fps_invalid() { mark_validation_error(fps_spin_); }
void ImportWidget::mark_start_index_invalid() { mark_validation_error(start_index_spin_); }
void ImportWidget::mark_end_index_invalid() { mark_validation_error(end_index_spin_); }
void ImportWidget::mark_camera_config_invalid() { mark_validation_error(camera_config_combo_); }

QLineEdit   *ImportWidget::file_line_edit() { return file_line_edit_; }
QPushButton *ImportWidget::browse_button() { return browse_button_; }
QPushButton *ImportWidget::start_button() { return start_button_; }
QPushButton *ImportWidget::stop_button() { return stop_button_; }
QSpinBox    *ImportWidget::fps_spin() { return fps_spin_; }
QSpinBox    *ImportWidget::start_index_spin() { return start_index_spin_; }
QSpinBox    *ImportWidget::end_index_spin() { return end_index_spin_; }
QComboBox   *ImportWidget::load_method_combo() { return load_method_combo_; }
QCheckBox   *ImportWidget::cam_check() { return cam_check_; }
QComboBox   *ImportWidget::camera_combo() { return camera_combo_; }
QComboBox   *ImportWidget::camera_config_combo() { return camera_config_combo_; }

void ImportWidget::setup_ui() {
  auto *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  main_layout->setSpacing(6);

  cam_check_ = new QCheckBox("Use Camera", this);
  main_layout->addWidget(cam_check_);

  stack_ = new QStackedLayout();
  main_layout->addLayout(stack_);

  stack_->addWidget(create_file_page());
  stack_->addWidget(create_camera_page());

  auto *button_layout = new QHBoxLayout();
  start_button_       = new QPushButton("Start", this);
  stop_button_        = new QPushButton("Stop", this);
  stop_button_->setEnabled(false);
  button_layout->addWidget(start_button_);
  button_layout->addWidget(stop_button_);
  main_layout->addLayout(button_layout);

  main_layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding));

  stack_->setCurrentIndex(0);
}

void ImportWidget::connect_signals() {
  connect(cam_check_, &QCheckBox::toggled, this, [this](bool checked) {
    stack_->setCurrentIndex(checked ? 1 : 0);
    emit settings_changed();
  });

  connect(start_button_, &QPushButton::clicked, this, &ImportWidget::start_clicked);
  connect(stop_button_, &QPushButton::clicked, this, &ImportWidget::stop_clicked);
  connect(browse_button_, &QPushButton::clicked, this, &ImportWidget::browse_clicked);

  // Emit settings_changed for all control changes
  connect(file_line_edit_, &QLineEdit::editingFinished, this, &ImportWidget::settings_changed);
  connect(fps_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImportWidget::settings_changed);
  connect(start_index_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImportWidget::settings_changed);
  connect(end_index_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImportWidget::settings_changed);
  connect(load_method_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ImportWidget::settings_changed);
  connect(camera_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ImportWidget::settings_changed);
  connect(camera_config_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ImportWidget::settings_changed);
}

QWidget *ImportWidget::create_file_page() {
  auto *page = new QWidget(this);
  auto *grid = new QGridLayout(page);
  grid->setContentsMargins(0, 0, 0, 0);
  grid->setHorizontalSpacing(6);
  grid->setVerticalSpacing(4);
  int row = 0;

  file_line_edit_ = new QLineEdit(page);
  file_line_edit_->setPlaceholderText("Select File");
  file_line_edit_->setReadOnly(true);
  grid->addWidget(file_line_edit_, row, 0);

  browse_button_ = new QPushButton("...", page);
  browse_button_->setFixedWidth(30);
  grid->addWidget(browse_button_, row, 1);
  ++row;

  auto add_spin_row = [&](const QString &label, QSpinBox *&target, int minimum, int maximum,
                          int value) {
    grid->addWidget(new QLabel(label, page), row, 0);
    target = create_spin_box(page, minimum, maximum, value);
    grid->addWidget(target, row, 1);
    ++row;
  };

  add_spin_row("Input FPS Limit", fps_spin_, 0, 999999, 0);
  fps_spin_->setSpecialValueText("Unlimited");
  fps_spin_->setToolTip("Set to Unlimited to read the holofile as fast as possible.");
  add_spin_row("Start Index", start_index_spin_, 0, 999999, 1);
  add_spin_row("End Index", end_index_spin_, 1, 999999, 60);

  auto load_methods  = QStringList{"Read Live", "Load in CPU RAM", "Load in GPU RAM"};
  load_method_combo_ = create_combo_box(page, load_methods);
  grid->addWidget(load_method_combo_, row, 0, 1, 2);
  ++row;

  grid->setRowStretch(row, 1);

  return page;
}

QWidget *ImportWidget::create_camera_page() {
  auto *page = new QWidget(this);
  auto *grid = new QGridLayout(page);
  grid->setContentsMargins(0, 0, 0, 0);
  grid->setHorizontalSpacing(6);
  grid->setVerticalSpacing(4);
  int row = 0;

  grid->addWidget(new QLabel("Camera", page), row, 0);
  camera_combo_ = create_combo_box(
      page, QStringList{"Ametek S710 Euresys Coaxlink Octo", "Ametek S711 Euresys Coaxlink QSFP+"});
  grid->addWidget(camera_combo_, row, 1);
  ++row;

  grid->addWidget(new QLabel("Config File", page), row, 0);
  camera_config_combo_ = create_combo_box(page, load_available_camera_configs());
  grid->addWidget(camera_config_combo_, row, 1);
  ++row;

  grid->setRowStretch(row, 1);

  return page;
}

QStringList ImportWidget::load_available_camera_configs() {
  QString appDataBase       = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
  QString appDataPath       = appDataBase + "/" + QCoreApplication::applicationVersion();
  QString cameraConfigsPath = appDataPath + "/" + "camera_configs";
  QDir    dir(cameraConfigsPath);

  QStringList filters;
  filters << QStringLiteral("*.json");
  QFileInfoList files = dir.entryInfoList(filters, QDir::Files | QDir::Readable, QDir::Name);

  QStringList names;
  for (const QFileInfo &fi : files) {
    names << fi.completeBaseName();
  }

  return names;
}

void ImportWidget::set_file_path(const QString &path) { file_line_edit_->setText(path); }

void ImportWidget::clear_validation_styles() {
  clear_validation_error(file_line_edit_);
  clear_validation_error(fps_spin_);
  clear_validation_error(start_index_spin_);
  clear_validation_error(end_index_spin_);
  clear_validation_error(camera_config_combo_);
}

} // namespace holovibes::ui
