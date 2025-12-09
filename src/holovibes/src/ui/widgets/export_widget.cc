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

#include "ui/widgets/export_widget.hh"

#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpacerItem>

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

} // namespace

ExportWidget::ExportWidget(QWidget *parent) : QGroupBox("Export", parent) {
  setCheckable(true);
  setChecked(false);
  setup_ui();
  connect_signals();
}

QString ExportWidget::get_image_type() const { return image_type_combo_->currentText(); }
QString ExportWidget::get_file_path() const { return file_line_edit_->text(); }
QString ExportWidget::get_tag() const { return tag_combo_->currentText(); }
bool    ExportWidget::is_frame_count_enabled() const { return frames_check_->isChecked(); }
int     ExportWidget::get_frame_count() const { return frames_spin_->value(); }

void ExportWidget::set_file_path(const QString &path) { file_line_edit_->setText(path); }
void ExportWidget::set_frame_count(int count) { frames_spin_->setValue(count); }
void ExportWidget::set_image_type(const QString &type) { image_type_combo_->setCurrentText(type); }

void ExportWidget::set_record_enabled(bool enabled) { record_button_->setEnabled(enabled); }
void ExportWidget::set_stop_enabled(bool enabled) { stop_button_->setEnabled(enabled); }

void ExportWidget::mark_file_invalid() {
  file_line_edit_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
}
void ExportWidget::mark_frames_invalid() {
  frames_spin_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
}

QComboBox   *ExportWidget::image_type_combo() { return image_type_combo_; }
QLineEdit   *ExportWidget::file_line_edit() { return file_line_edit_; }
QPushButton *ExportWidget::browse_button() { return browse_button_; }
QComboBox   *ExportWidget::tag_combo() { return tag_combo_; }
QCheckBox   *ExportWidget::frames_check() { return frames_check_; }
QSpinBox    *ExportWidget::frames_spin() { return frames_spin_; }
QPushButton *ExportWidget::record_button() { return record_button_; }
QPushButton *ExportWidget::stop_button() { return stop_button_; }
QPushButton *ExportWidget::stop_fan_button() { return stop_fan_button_; }

void ExportWidget::setup_ui() {
  auto *layout = new QGridLayout(this);
  int   row    = 0;

  image_type_combo_ = create_combo_box(this, QStringList{"Raw Image", "Processed Image"});
  layout->addWidget(image_type_combo_, row, 0, 1, 2);
  ++row;

  file_line_edit_ = new QLineEdit(this);
  file_line_edit_->setText("holovibes\\capture");
  file_line_edit_->setReadOnly(true);
  layout->addWidget(file_line_edit_, row, 0);

  browse_button_ = new QPushButton("...", this);
  browse_button_->setFixedWidth(30);
  layout->addWidget(browse_button_, row, 1);
  ++row;

  layout->addWidget(new QLabel("Tag", this), row, 0);
  tag_combo_ = create_combo_box(this, QStringList{"None", "Left Eye", "Right Eye"});
  layout->addWidget(tag_combo_, row, 1);
  ++row;

  frames_check_ = new QCheckBox("Nb. of frames", this);
  frames_check_->setChecked(true);
  layout->addWidget(frames_check_, row, 0);
  frames_spin_ = create_spin_box(this, 1, 999999, 2048);
  layout->addWidget(frames_spin_, row, 1);
  ++row;

  auto *button_layout = new QHBoxLayout();
  record_button_      = new QPushButton("Record", this);
  stop_button_        = new QPushButton("Stop", this);
  stop_button_->setEnabled(false);
  stop_fan_button_ = new QPushButton("Stop fan", this);
  record_button_->setEnabled(false);
  button_layout->addWidget(record_button_);
  button_layout->addWidget(stop_button_);
  button_layout->addWidget(stop_fan_button_);
  layout->addLayout(button_layout, row, 0, 1, 2);
  ++row;

  layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), row, 0, 1,
                  2);
}

void ExportWidget::connect_signals() {
  connect(record_button_, &QPushButton::clicked, this, &ExportWidget::record_clicked);
  connect(stop_button_, &QPushButton::clicked, this, &ExportWidget::stop_clicked);
  connect(stop_fan_button_, &QPushButton::clicked, this, &ExportWidget::stop_fan_clicked);
  connect(browse_button_, &QPushButton::clicked, this, &ExportWidget::browse_clicked);

  // Emit settings_changed for all control changes
  connect(this, &QGroupBox::isChecked, this, &ExportWidget::settings_changed);
  connect(image_type_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ExportWidget::settings_changed);
  connect(file_line_edit_, &QLineEdit::editingFinished, this, &ExportWidget::settings_changed);
  connect(tag_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ExportWidget::settings_changed);
  connect(frames_check_, &QCheckBox::toggled, this, &ExportWidget::settings_changed);
  connect(frames_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ExportWidget::settings_changed);
}

void ExportWidget::clear_validation_styles() {
  file_line_edit_->setStyleSheet("");
  frames_spin_->setStyleSheet("");
}

} // namespace holovibes::ui