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

#include "ui/widgets/image_rendering_widget.hh"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QGridLayout>
#include <QLabel>
#include <QSpacerItem>
#include <QStandardPaths>

namespace holovibes::ui {

namespace {

constexpr int kLargeSpinMax = 1024 * 1024;

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

ImageRenderingWidget::ImageRenderingWidget(QWidget *parent) : QGroupBox("Image Rendering", parent) {
  setup_ui();
  connect_signals();
}

QString ImageRenderingWidget::get_image_mode() const { return image_combo_->currentText(); }
int     ImageRenderingWidget::get_batch_size() const { return batch_size_spin_->value(); }
int     ImageRenderingWidget::get_time_stride() const { return time_stride_spin_->value(); }
bool    ImageRenderingWidget::is_filter_2d_enabled() const { return filter_2d_check_->isChecked(); }
int     ImageRenderingWidget::get_filter_inner() const { return filter_2d_inner_spin_->value(); }
int     ImageRenderingWidget::get_filter_outer() const { return filter_2d_outer_spin_->value(); }
QString ImageRenderingWidget::get_space_transform() const {
  return space_transform_combo_->currentText();
}
QString ImageRenderingWidget::get_time_transform() const {
  return time_transform_combo_->currentText();
}
int     ImageRenderingWidget::get_time_window() const { return time_window_spin_->value(); }
int     ImageRenderingWidget::get_lambda() const { return lambda_spin_->value(); }
int     ImageRenderingWidget::get_focus() const { return focus_spin_->value(); }
QString ImageRenderingWidget::get_convolution() const { return convolution_combo_->currentText(); }
bool    ImageRenderingWidget::is_convolution_divide() const {
  return convolution_divide_check_->isChecked();
}

void ImageRenderingWidget::set_batch_size(int value) { batch_size_spin_->setValue(value); }
void ImageRenderingWidget::set_time_stride(int value) { time_stride_spin_->setValue(value); }
void ImageRenderingWidget::set_filter_2d_enabled(bool enabled) {
  filter_2d_check_->setChecked(enabled);
}
void ImageRenderingWidget::set_filter_inner(int value) { filter_2d_inner_spin_->setValue(value); }
void ImageRenderingWidget::set_filter_outer(int value) { filter_2d_outer_spin_->setValue(value); }
void ImageRenderingWidget::set_space_transform(const QString &method) {
  space_transform_combo_->setCurrentText(method);
}
void ImageRenderingWidget::set_time_transform(const QString &method) {
  time_transform_combo_->setCurrentText(method);
}
void ImageRenderingWidget::set_time_window(int value) { time_window_spin_->setValue(value); }
void ImageRenderingWidget::set_lambda(int value) { lambda_spin_->setValue(value); }
void ImageRenderingWidget::set_focus(int value) { focus_spin_->setValue(value); }
void ImageRenderingWidget::set_convolution(const QString &kernel) {
  convolution_combo_->setCurrentText(kernel);
}
void ImageRenderingWidget::set_convolution_divide(bool enabled) {
  convolution_divide_check_->setChecked(enabled);
}

void ImageRenderingWidget::mark_batch_size_invalid() {
  batch_size_spin_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
}
void ImageRenderingWidget::mark_time_stride_invalid() {
  time_stride_spin_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
}
void ImageRenderingWidget::mark_time_window_invalid() {
  time_window_spin_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
}

QComboBox *ImageRenderingWidget::image_combo() { return image_combo_; }
QSpinBox  *ImageRenderingWidget::batch_size_spin() { return batch_size_spin_; }
QSpinBox  *ImageRenderingWidget::time_stride_spin() { return time_stride_spin_; }
QCheckBox *ImageRenderingWidget::filter_2d_check() { return filter_2d_check_; }
QSpinBox  *ImageRenderingWidget::filter_2d_inner_spin() { return filter_2d_inner_spin_; }
QSpinBox  *ImageRenderingWidget::filter_2d_outer_spin() { return filter_2d_outer_spin_; }
QComboBox *ImageRenderingWidget::space_transform_combo() { return space_transform_combo_; }
QComboBox *ImageRenderingWidget::time_transform_combo() { return time_transform_combo_; }
QSpinBox  *ImageRenderingWidget::time_window_spin() { return time_window_spin_; }
QSpinBox  *ImageRenderingWidget::lambda_spin() { return lambda_spin_; }
QSpinBox  *ImageRenderingWidget::focus_spin() { return focus_spin_; }
QSlider   *ImageRenderingWidget::focus_slider() { return focus_slider_; }
QComboBox *ImageRenderingWidget::convolution_combo() { return convolution_combo_; }
QCheckBox *ImageRenderingWidget::convolution_divide_check() { return convolution_divide_check_; }
AutoFocusWidget *ImageRenderingWidget::autofocus_widget() { return autofocus_widget_; }

void ImageRenderingWidget::setup_ui() {
  auto *layout = new QGridLayout(this);
  int   row    = 0;

  auto add_combo_row = [&](const QString &label, QComboBox *&combo, const QStringList &items) {
    layout->addWidget(new QLabel(label, this), row, 0);
    combo = create_combo_box(this, items);
    layout->addWidget(combo, row, 1);
    ++row;
  };

  auto add_spin_row = [&](const QString &label, QSpinBox *&spin, int minimum, int maximum,
                          int value) {
    layout->addWidget(new QLabel(label, this), row, 0);
    spin = create_spin_box(this, minimum, maximum, value);
    layout->addWidget(spin, row, 1);
    ++row;
  };

  add_combo_row("Image:", image_combo_, QStringList{"Raw", "Processed"});
  add_spin_row("Batch Size:", batch_size_spin_, 1, kLargeSpinMax, 32);
  add_spin_row("Time Stride:", time_stride_spin_, 1, kLargeSpinMax, 32);

  auto *filter_layout = new QGridLayout();
  filter_2d_check_    = new QCheckBox("Filter 2D", this);
  filter_layout->addWidget(filter_2d_check_, 0, 0);
  filter_2d_inner_spin_ = create_spin_box(this, 0, kLargeSpinMax, 0);
  filter_layout->addWidget(filter_2d_inner_spin_, 0, 1);
  filter_2d_outer_spin_ = create_spin_box(this, 0, kLargeSpinMax, 1024);
  filter_layout->addWidget(filter_2d_outer_spin_, 0, 2);
  layout->addLayout(filter_layout, row, 0, 1, 2);
  ++row;

  auto space_transforms = QStringList{"None", "Fresnel Diffraction", "Angular Spectrum"};
  add_combo_row("Space Transform:", space_transform_combo_, space_transforms);
  auto time_transforms = QStringList{"None", "Short Time Fourier", "Principal Component Analysis"};
  add_combo_row("Time Transform:", time_transform_combo_, time_transforms);
  add_spin_row("Time Window:", time_window_spin_, 1, kLargeSpinMax, 32);
  add_spin_row("Lambda (nm):", lambda_spin_, 1, kLargeSpinMax, 852);
  add_spin_row("Focus (mm):", focus_spin_, 1, kLargeSpinMax, 380);

  focus_slider_ = new QSlider(Qt::Horizontal, this);
  focus_slider_->setRange(0, 1000);
  focus_slider_->setValue(focus_spin_->value());
  layout->addWidget(focus_slider_, row, 0, 1, 2);
  ++row;

  autofocus_widget_ = new AutoFocusWidget(this);
  layout->addWidget(autofocus_widget_, row, 0, 1, 2);
  ++row;

  layout->addWidget(new QLabel("Convolution:", this), row, 0, 1, 2);
  ++row;

  convolution_combo_ = create_combo_box(this, load_available_kernels());
  layout->addWidget(convolution_combo_, row, 0);
  convolution_divide_check_ = new QCheckBox("Divide", this);
  layout->addWidget(convolution_divide_check_, row, 1, 1, 1, Qt::AlignRight);
  ++row;

  layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), row, 0, 1,
                  2);
}

void ImageRenderingWidget::connect_signals() {
  // Synchronize focus spin and slider
  connect(focus_spin_, &QSpinBox::valueChanged, this, [this](int value) {
    if (focus_slider_->value() != value) {
      focus_slider_->blockSignals(true);
      focus_slider_->setValue(value);
      focus_slider_->blockSignals(false);
    }
    emit settings_changed();
  });

  connect(focus_slider_, &QSlider::valueChanged, this, [this](int value) {
    if (focus_spin_->value() != value) {
      focus_spin_->blockSignals(true);
      focus_spin_->setValue(value);
      focus_spin_->blockSignals(false);
    }
    emit settings_changed();
  });

  // Emit settings_changed for all control changes
  connect(image_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(batch_size_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(time_stride_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(filter_2d_check_, &QCheckBox::toggled, this, &ImageRenderingWidget::settings_changed);
  connect(filter_2d_inner_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(filter_2d_outer_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(space_transform_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(time_transform_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(time_window_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(lambda_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(convolution_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(convolution_divide_check_, &QCheckBox::toggled, this,
          &ImageRenderingWidget::settings_changed);
}

QStringList ImageRenderingWidget::load_available_kernels() {
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

void ImageRenderingWidget::clear_validation_styles() {
  batch_size_spin_->setStyleSheet("");
  time_stride_spin_->setStyleSheet("");
  time_window_spin_->setStyleSheet("");
}

} // namespace holovibes::ui