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
#include "ui/widgets/validation_style.hh"
#include "holotask/sinks/ffmpeg_formats.hh"

#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpacerItem>
#include <QSignalBlocker>
#include <QVBoxLayout>
#include <algorithm>

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

ExportWidget::ExportWidget(QWidget *parent) : QGroupBox("EXPORT", parent) {
  setup_ui();
  connect_signals();
  setChecked(false);
}

QString ExportWidget::get_image_type() const { return image_type_combo_->currentText(); }
QString ExportWidget::get_format() const { return format_combo_->currentData().toString(); }
QString ExportWidget::get_codec() const { return codec_combo_->currentData().toString(); }
QString ExportWidget::get_resize_algorithm() const {
  return resize_algorithm_combo_->currentData().toString();
}
QString ExportWidget::get_file_path() const { return file_line_edit_->text(); }
QString ExportWidget::get_tag() const { return tag_combo_->currentText(); }
bool    ExportWidget::is_frame_count_enabled() const { return frames_check_->isChecked(); }
int     ExportWidget::get_frame_count() const { return frames_spin_->value(); }
bool    ExportWidget::isChecked() const { return enable_check_->isChecked(); }

void ExportWidget::set_file_path(const QString &path) { file_line_edit_->setText(path); }
void ExportWidget::set_frame_count(int count) { frames_spin_->setValue(count); }
void ExportWidget::set_frame_batch_size(int batch_size) {
  frame_batch_size_ = std::max(1, batch_size);
  const auto value = frames_spin_->value();
  frames_lower_button_->setEnabled(value > frame_batch_size_);
}
void ExportWidget::set_image_type(const QString &type) { image_type_combo_->setCurrentText(type); }
void ExportWidget::set_resize_algorithm(const QString &algorithm) {
  const auto index = resize_algorithm_combo_->findData(algorithm);
  resize_algorithm_combo_->setCurrentIndex(index >= 0 ? index : 0);
}
void ExportWidget::setChecked(bool checked) {
  enable_check_->setChecked(checked);
  set_export_controls_enabled(checked);
}

void ExportWidget::set_record_enabled(bool enabled) { record_button_->setEnabled(enabled); }
void ExportWidget::set_stop_enabled(bool enabled) { stop_button_->setEnabled(enabled); }

void ExportWidget::mark_file_invalid() { mark_validation_error(file_line_edit_); }
void ExportWidget::mark_frames_invalid() { mark_validation_error(frames_spin_); }

QComboBox   *ExportWidget::image_type_combo() { return image_type_combo_; }
QComboBox   *ExportWidget::format_combo() { return format_combo_; }
QComboBox   *ExportWidget::codec_combo() { return codec_combo_; }
QComboBox   *ExportWidget::resize_algorithm_combo() { return resize_algorithm_combo_; }
QLineEdit   *ExportWidget::file_line_edit() { return file_line_edit_; }
QPushButton *ExportWidget::browse_button() { return browse_button_; }
QComboBox   *ExportWidget::tag_combo() { return tag_combo_; }
QCheckBox   *ExportWidget::frames_check() { return frames_check_; }
QSpinBox    *ExportWidget::frames_spin() { return frames_spin_; }
QPushButton *ExportWidget::record_button() { return record_button_; }
QPushButton *ExportWidget::stop_button() { return stop_button_; }
QPushButton *ExportWidget::stop_fan_button() { return stop_fan_button_; }

void ExportWidget::setup_ui() {
  auto *outer_layout = new QVBoxLayout(this);
  outer_layout->setContentsMargins(0, 0, 0, 0);
  outer_layout->setSpacing(6);

  enable_check_ = new QCheckBox("Enable export", this);
  outer_layout->addWidget(enable_check_);

  content_container_ = new QWidget(this);
  outer_layout->addWidget(content_container_);

  auto *layout = new QGridLayout(content_container_);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setHorizontalSpacing(6);
  layout->setVerticalSpacing(4);
  int row = 0;

  image_type_combo_ =
      create_combo_box(content_container_, QStringList{"Raw Image", "Processed Image"});
  layout->addWidget(image_type_combo_, row, 0, 1, 2);
  ++row;

  layout->addWidget(new QLabel("Format", content_container_), row, 0);
  format_combo_ = new QComboBox(content_container_);
  for (const auto &format : holotask::sinks::kFfmpegFormats) {
    format_combo_->addItem(QString::fromUtf8(format.label.data(), format.label.size()) +
                               " (." +
                               QString::fromUtf8(format.extension.data(), format.extension.size()) +
                               ")",
                           QString::fromUtf8(format.name.data(), format.name.size()));
  }
  layout->addWidget(format_combo_, row, 1);
  ++row;

  layout->addWidget(new QLabel("Codec", content_container_), row, 0);
  codec_combo_ = new QComboBox(content_container_);
  layout->addWidget(codec_combo_, row, 1);
  ++row;
  update_codec_choices();

  layout->addWidget(new QLabel("Resize algorithm", content_container_), row, 0);
  resize_algorithm_combo_ = create_combo_box(
      content_container_, QStringList{"CPU bilinear", "GPU bilinear"});
  resize_algorithm_combo_->setItemData(0, "CpuBilinear");
  resize_algorithm_combo_->setItemData(1, "CudaBilinear");
  resize_algorithm_combo_->setToolTip(
      "Algorithm used when exporting video with square resizing enabled.");
  layout->addWidget(resize_algorithm_combo_, row, 1);
  ++row;

  file_line_edit_ = new QLineEdit(content_container_);
  file_line_edit_->setText("holovibes\\capture");
  file_line_edit_->setReadOnly(true);
  layout->addWidget(file_line_edit_, row, 0);

  browse_button_ = new QPushButton("...", content_container_);
  browse_button_->setFixedWidth(30);
  layout->addWidget(browse_button_, row, 1);
  ++row;

  layout->addWidget(new QLabel("Tag", content_container_), row, 0);
  tag_combo_ = create_combo_box(content_container_, QStringList{"None", "Left Eye", "Right Eye"});
  layout->addWidget(tag_combo_, row, 1);
  ++row;

  frames_check_ = new QCheckBox("Nb. of frames", content_container_);
  frames_check_->setChecked(true);
  layout->addWidget(frames_check_, row, 0);
  auto *frame_controls = new QHBoxLayout();
  frames_spin_ = create_spin_box(content_container_, 1, 999999, 2048);
  frames_lower_button_ = new QPushButton("−", content_container_);
  frames_higher_button_ = new QPushButton("+", content_container_);
  frames_lower_button_->setFixedWidth(28);
  frames_higher_button_->setFixedWidth(28);
  frame_controls->addWidget(frames_lower_button_);
  frame_controls->addWidget(frames_spin_, 1);
  frame_controls->addWidget(frames_higher_button_);
  layout->addLayout(frame_controls, row, 1);
  ++row;

  auto *button_layout = new QHBoxLayout();
  record_button_      = new QPushButton("Record", content_container_);
  stop_button_        = new QPushButton("Stop", content_container_);
  stop_button_->setEnabled(false);
  stop_fan_button_ = new QPushButton("Stop fan", content_container_);
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
  connect(enable_check_, &QCheckBox::toggled, this, [this](bool enabled) {
    set_export_controls_enabled(enabled);
    emit settings_changed();
  });
  connect(record_button_, &QPushButton::clicked, this, &ExportWidget::record_clicked);
  connect(stop_button_, &QPushButton::clicked, this, &ExportWidget::stop_clicked);
  connect(stop_fan_button_, &QPushButton::clicked, this, &ExportWidget::stop_fan_clicked);
  connect(browse_button_, &QPushButton::clicked, this, &ExportWidget::browse_clicked);

  // Emit settings_changed for all control changes
  connect(image_type_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ExportWidget::settings_changed);
  connect(format_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          [this](int) {
            update_codec_choices();
            emit settings_changed();
          });
  connect(codec_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ExportWidget::settings_changed);
  connect(resize_algorithm_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ExportWidget::settings_changed);
  connect(file_line_edit_, &QLineEdit::textChanged, this, &ExportWidget::settings_changed);
  connect(tag_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ExportWidget::settings_changed);
  connect(frames_check_, &QCheckBox::toggled, this, &ExportWidget::settings_changed);
  connect(frames_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this](int value) {
            frames_lower_button_->setEnabled(value > frame_batch_size_);
            emit settings_changed();
          });
  connect(frames_lower_button_, &QPushButton::clicked, this, [this] {
    const auto batch = frame_batch_size_;
    const auto value = frames_spin_->value();
    const auto lower = ((value - 1) / batch) * batch;
    frames_spin_->setValue(std::max(batch, lower));
  });
  connect(frames_higher_button_, &QPushButton::clicked, this, [this] {
    const auto batch = frame_batch_size_;
    const auto value = frames_spin_->value();
    const auto higher = ((value / batch) + 1) * batch;
    frames_spin_->setValue(std::min(frames_spin_->maximum(), higher));
  });
  set_frame_batch_size(1);
}

void ExportWidget::clear_validation_styles() {
  clear_validation_error(file_line_edit_);
  clear_validation_error(frames_spin_);
}

void ExportWidget::set_export_controls_enabled(bool enabled) {
  content_container_->setEnabled(enabled);
}

void ExportWidget::update_codec_choices() {
  const auto format_name = format_combo_->currentData().toString().toStdString();
  const auto *format      = holotask::sinks::ffmpeg_format(format_name);
  const auto  previous    = codec_combo_->currentData().toString();
  QSignalBlocker blocker(codec_combo_);
  codec_combo_->clear();
  if (format == nullptr || format->codecs.empty()) {
    codec_combo_->setEnabled(false);
    codec_combo_->setToolTip(format_name == "holo" || format_name == "npy"
                                 ? tr("This format does not use a video codec")
                                 : tr("No codec available for this format"));
    return;
  }
  for (const auto &codec : format->codecs)
    codec_combo_->addItem(QString::fromUtf8(codec.label.data(), codec.label.size()),
                          QString::fromUtf8(codec.name.data(), codec.name.size()));
  const auto index = codec_combo_->findData(previous);
  codec_combo_->setCurrentIndex(index >= 0 ? index : 0);
  codec_combo_->setEnabled(true);
  codec_combo_->setToolTip({});
}

} // namespace holovibes::ui
