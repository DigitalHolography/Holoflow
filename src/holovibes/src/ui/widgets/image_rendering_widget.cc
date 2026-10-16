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
#include "ui/widgets/validation_style.hh"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpacerItem>
#include <QSignalBlocker>
#include <QStandardItemModel>
#include <QStandardPaths>
#include <algorithm>
#include <limits>

namespace holovibes::ui {

namespace {

// constexpr int kLargeSpinMax = 1024 * 1024;
constexpr int kLargeSpinMax = std::numeric_limits<int>::max();

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

ImageRenderingWidget::ImageRenderingWidget(QWidget *parent) : QGroupBox("IMAGE RENDERING", parent) {
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
int  ImageRenderingWidget::get_time_window() const { return time_window_spin_->value(); }
int  ImageRenderingWidget::get_lambda() const { return lambda_spin_->value(); }
int  ImageRenderingWidget::get_focus() const { return focus_spin_->value(); }
bool ImageRenderingWidget::is_asp_padding_enabled() const {
  return asp_padding_check_->isChecked();
}
int ImageRenderingWidget::get_asp_padded_width() const { return asp_padded_width_spin_->value(); }
int ImageRenderingWidget::get_asp_padded_height() const { return asp_padded_height_spin_->value(); }
QString ImageRenderingWidget::get_convolution() const { return convolution_combo_->currentText(); }
bool    ImageRenderingWidget::is_convolution_divide() const {
  return convolution_divide_check_->isChecked();
}
bool ImageRenderingWidget::is_spatial_contrast_enabled() const {
  return spatial_contrast_check_->isChecked();
}
int ImageRenderingWidget::get_spatial_contrast_window_size() const {
  return spatial_contrast_window_size_->value();
}
int ImageRenderingWidget::get_spatial_contrast_dimensions() const {
  return spatial_contrast_dimensions_->currentData().toInt();
}
QString ImageRenderingWidget::get_spatial_contrast_method() const {
  return spatial_contrast_method_->currentText();
}
double ImageRenderingWidget::get_spatial_contrast_display_max() const {
  return spatial_contrast_display_max_->value();
}
bool ImageRenderingWidget::is_spatial_contrast_display_inverted() const {
  return spatial_contrast_invert_display_->isChecked();
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
void ImageRenderingWidget::set_asp_padding_enabled(bool enabled) {
  asp_padding_check_->setChecked(enabled);
}
void ImageRenderingWidget::set_asp_padded_width(int value) {
  asp_padded_width_spin_->setValue(value);
}
void ImageRenderingWidget::set_asp_padded_height(int value) {
  asp_padded_height_spin_->setValue(value);
}
void ImageRenderingWidget::set_convolution(const QString &kernel) {
  convolution_combo_->setCurrentText(kernel);
}
void ImageRenderingWidget::set_convolution_divide(bool enabled) {
  convolution_divide_check_->setChecked(enabled);
}
void ImageRenderingWidget::set_spatial_contrast_enabled(bool enabled) {
  spatial_contrast_check_->setChecked(enabled);
  spatial_contrast_window_size_->setEnabled(enabled && get_spatial_contrast_dimensions() != 1);
  spatial_contrast_dimensions_->setEnabled(enabled);
  spatial_contrast_method_->setEnabled(enabled && get_spatial_contrast_dimensions() != 1);
  spatial_contrast_display_max_->setEnabled(enabled);
  spatial_contrast_invert_display_->setEnabled(enabled);
  update_time_window_limits();
}
void ImageRenderingWidget::set_spatial_contrast_window_size(int value) {
  spatial_contrast_window_size_->setValue(value);
}
void ImageRenderingWidget::set_spatial_contrast_dimensions(int value) {
  const int index = spatial_contrast_dimensions_->findData(value);
  if (index >= 0)
    spatial_contrast_dimensions_->setCurrentIndex(index);
  const bool enabled = spatial_contrast_check_->isChecked();
  spatial_contrast_window_size_->setEnabled(enabled && value != 1);
  spatial_contrast_method_->setEnabled(enabled && value != 1);
  update_time_window_limits();
}
void ImageRenderingWidget::set_time_window_max_for_temporal_lscsi(int value) {
  max_temporal_lscsi_frames_ = std::max(value, 3);
  update_time_window_limits();
}
void ImageRenderingWidget::update_time_window_limits() {
  const bool temporal_lscsi = spatial_contrast_check_->isChecked() &&
                              get_spatial_contrast_dimensions() != 2;
  const QSignalBlocker blocker(time_window_spin_);
  time_window_spin_->setRange(temporal_lscsi ? 3 : 1,
                              temporal_lscsi ? max_temporal_lscsi_frames_ : kLargeSpinMax);
}
void ImageRenderingWidget::set_spatial_contrast_method(const QString &method) {
  spatial_contrast_method_->setCurrentText(method);
}
void ImageRenderingWidget::set_spatial_contrast_display_max(double value) {
  spatial_contrast_display_max_->setValue(value);
}
void ImageRenderingWidget::set_spatial_contrast_display_inverted(bool inverted) {
  spatial_contrast_invert_display_->setChecked(inverted);
}

void ImageRenderingWidget::mark_batch_size_invalid() { mark_validation_error(batch_size_spin_); }
void ImageRenderingWidget::mark_time_stride_invalid() { mark_validation_error(time_stride_spin_); }
void ImageRenderingWidget::mark_filter_2d_invalid() { mark_validation_error(filter_2d_check_); }
void ImageRenderingWidget::mark_filter_inner_invalid() {
  mark_validation_error(filter_2d_inner_spin_);
}
void ImageRenderingWidget::mark_filter_outer_invalid() {
  mark_validation_error(filter_2d_outer_spin_);
}
void ImageRenderingWidget::mark_time_window_invalid() { mark_validation_error(time_window_spin_); }
void ImageRenderingWidget::mark_space_transform_invalid() {
  mark_validation_error(space_transform_combo_);
}
void ImageRenderingWidget::mark_asp_padding_invalid() {
  mark_validation_error(asp_padding_check_);
  mark_validation_error(asp_padded_width_spin_);
  mark_validation_error(asp_padded_height_spin_);
}
void ImageRenderingWidget::mark_time_transform_invalid() {
  mark_validation_error(time_transform_combo_);
}
void ImageRenderingWidget::mark_convolution_invalid() { mark_validation_error(convolution_combo_); }
void ImageRenderingWidget::mark_spatial_contrast_window_invalid() {
  mark_validation_error(spatial_contrast_window_size_);
}
void ImageRenderingWidget::mark_spatial_contrast_dimensions_invalid() {
  mark_validation_error(spatial_contrast_dimensions_);
}
void ImageRenderingWidget::mark_spatial_contrast_method_invalid() {
  mark_validation_error(spatial_contrast_method_);
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
QCheckBox *ImageRenderingWidget::asp_padding_check() { return asp_padding_check_; }
QSpinBox  *ImageRenderingWidget::asp_padded_width_spin() { return asp_padded_width_spin_; }
QSpinBox  *ImageRenderingWidget::asp_padded_height_spin() { return asp_padded_height_spin_; }
QComboBox *ImageRenderingWidget::convolution_combo() { return convolution_combo_; }
QCheckBox *ImageRenderingWidget::convolution_divide_check() { return convolution_divide_check_; }
QCheckBox *ImageRenderingWidget::spatial_contrast_check() { return spatial_contrast_check_; }
QSpinBox *ImageRenderingWidget::spatial_contrast_window_size() {
  return spatial_contrast_window_size_;
}
QComboBox *ImageRenderingWidget::spatial_contrast_dimensions() {
  return spatial_contrast_dimensions_;
}
QComboBox *ImageRenderingWidget::spatial_contrast_method() { return spatial_contrast_method_; }
QDoubleSpinBox *ImageRenderingWidget::spatial_contrast_display_max() {
  return spatial_contrast_display_max_;
}
AutoFocusWidget *ImageRenderingWidget::autofocus_widget() { return autofocus_widget_; }

void ImageRenderingWidget::setup_ui() {
  auto *layout = new QGridLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setHorizontalSpacing(6);
  layout->setVerticalSpacing(4);
  layout->setColumnStretch(1, 1);
  int row = 0;

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
  image_combo_->setCurrentText("Processed");

  spatial_contrast_check_ = new QCheckBox("Spatial LSCI contrast", this);
  spatial_contrast_check_->setToolTip(
      tr("Compute local K = standard deviation / mean. 1D uses the same pixel across a centered "
         "time window, 2D uses a spatial sliding window, and 3D uses both. Select Processed "
         "image mode to display K. Accumulation averages successive K maps. This direct LSCI "
         "path skips hologram propagation, time transforms, FFT shift, Flatfield, and PCT clipping. "
         "FFT convolution applies to spatial windows; 1D temporal-only mode uses direct samples."));
  layout->addWidget(spatial_contrast_check_, row, 0, 1, 2);
  ++row;
  layout->addWidget(new QLabel("Window (px):", this), row, 0);
  spatial_contrast_window_size_ = create_spin_box(this, 3, 255, 7);
  spatial_contrast_window_size_->setSingleStep(2);
  spatial_contrast_window_size_->setToolTip(
      tr("Odd spatial window size. Border windows use only pixels inside the frame."));
  spatial_contrast_window_size_->setEnabled(false);
  layout->addWidget(spatial_contrast_window_size_, row, 1);
  ++row;

  layout->addWidget(new QLabel("Dimensions:", this), row, 0);
  spatial_contrast_dimensions_ = new QComboBox(this);
  spatial_contrast_dimensions_->addItem(tr("1D (time only)"), 1);
  spatial_contrast_dimensions_->addItem(tr("2D (space)"), 2);
  spatial_contrast_dimensions_->addItem(tr("3D (space + time)"), 3);
  spatial_contrast_dimensions_->setCurrentIndex(1);
  spatial_contrast_dimensions_->setToolTip(
      tr("1D follows one pixel over time. 2D uses one frame. 3D uses a sliding temporal window "
         "and a spatial window."));
  spatial_contrast_dimensions_->setEnabled(false);
  layout->addWidget(spatial_contrast_dimensions_, row, 1);
  ++row;

  add_combo_row("LSCI method:", spatial_contrast_method_, QStringList{"Kernel", "FFT"});
  spatial_contrast_method_->setEnabled(false);
  layout->addWidget(new QLabel("Display max K:", this), row, 0);
  spatial_contrast_display_max_ = new QDoubleSpinBox(this);
  spatial_contrast_display_max_->setRange(0.05, 1.0);
  spatial_contrast_display_max_->setSingleStep(0.05);
  spatial_contrast_display_max_->setDecimals(2);
  spatial_contrast_display_max_->setValue(0.75);
  spatial_contrast_display_max_->setToolTip(
      tr("Map K=0 to black and this K value to white in the processed preview. Higher values "
         "are clipped. This affects display only."));
  spatial_contrast_display_max_->setEnabled(false);
  layout->addWidget(spatial_contrast_display_max_, row, 1);
  ++row;

  spatial_contrast_invert_display_ = new QCheckBox("Invert LSCI display", this);
  spatial_contrast_invert_display_->setToolTip(
      tr("Reverse black and white in the processed LSCI preview without changing the input or "
         "calculated contrast values."));
  spatial_contrast_invert_display_->setEnabled(false);
  layout->addWidget(spatial_contrast_invert_display_, row, 0, 1, 2);
  ++row;

  filter_2d_check_ = new QCheckBox("Filter 2D", this);
  layout->addWidget(filter_2d_check_, row, 0);

  auto *filter_controls = new QWidget(this);
  auto *filter_layout   = new QHBoxLayout(filter_controls);
  filter_layout->setContentsMargins(0, 0, 0, 0);
  filter_layout->setSpacing(6);
  filter_2d_inner_spin_ = create_spin_box(filter_controls, 0, kLargeSpinMax, 0);
  filter_layout->addWidget(filter_2d_inner_spin_);
  filter_2d_outer_spin_ = create_spin_box(filter_controls, 0, kLargeSpinMax, 1024);
  filter_layout->addWidget(filter_2d_outer_spin_);
  layout->addWidget(filter_controls, row, 1);
  ++row;

  auto space_transforms = QStringList{"None", "Fresnel Diffraction", "Angular Spectrum"};
  add_combo_row("Space Transform:", space_transform_combo_, space_transforms);

  asp_padding_check_ = new QCheckBox("ASP Padding", this);
  layout->addWidget(asp_padding_check_, row, 0);
  asp_padding_controls_    = new QWidget(this);
  auto *asp_padding_layout = new QHBoxLayout(asp_padding_controls_);
  asp_padding_layout->setContentsMargins(0, 0, 0, 0);
  asp_padding_layout->setSpacing(6);
  asp_padded_width_spin_ = create_spin_box(asp_padding_controls_, 1, kLargeSpinMax, 1024);
  asp_padded_width_spin_->setPrefix("W ");
  asp_padding_layout->addWidget(asp_padded_width_spin_);
  asp_padded_height_spin_ = create_spin_box(asp_padding_controls_, 1, kLargeSpinMax, 1024);
  asp_padded_height_spin_->setPrefix("H ");
  asp_padding_layout->addWidget(asp_padded_height_spin_);
  layout->addWidget(asp_padding_controls_, row, 1);
  ++row;

  auto time_transforms = QStringList{"None", "RFFT", "FFT", "Principal Component Analysis"};
  add_combo_row("Time Transform:", time_transform_combo_, time_transforms);
  add_spin_row("Time Window:", time_window_spin_, 1, kLargeSpinMax, 32);
  time_window_spin_->setToolTip(
      tr("Number of consecutive frames in each temporal-transform window. For 1D or 3D LSCI, "
         "this same value sets the temporal window; 2D LSCI ignores it."));
  add_spin_row("Lambda (nm):", lambda_spin_, 1, kLargeSpinMax, 852);
  add_spin_row("Focus (mm):", focus_spin_, 1, kLargeSpinMax, 380);

  focus_slider_ = new QSlider(Qt::Horizontal, this);
  focus_slider_->setRange(0, 1000);
  focus_slider_->setValue(focus_spin_->value());
  layout->addWidget(focus_slider_, row, 0, 1, 2);
  ++row;

  autofocus_widget_ = new AutoFocusWidget(this);

  layout->addWidget(new QLabel("Convolution:", this), row, 0, 1, 2);
  ++row;

  convolution_combo_ = create_combo_box(this, load_available_kernels());
  layout->addWidget(convolution_combo_, row, 0);
  convolution_divide_check_ = new QCheckBox("Divide", this);
  layout->addWidget(convolution_divide_check_, row, 1, 1, 1, Qt::AlignRight);
  ++row;

  layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), row, 0, 1,
                  2);
  update_asp_padding_visibility();
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
  connect(image_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this]() {
    emit settings_changed();
    emit spatial_contrast_display_inversion_changed(
        spatial_contrast_check_->isChecked() && get_image_mode() == "Processed" &&
        is_spatial_contrast_display_inverted());
  });
  connect(batch_size_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(time_stride_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(filter_2d_check_, &QCheckBox::toggled, this, &ImageRenderingWidget::settings_changed);
  connect(filter_2d_inner_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(filter_2d_outer_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(space_transform_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this]() {
    update_asp_padding_visibility();
    emit settings_changed();
  });
  connect(asp_padding_check_, &QCheckBox::toggled, this, [this](bool enabled) {
    asp_padding_controls_->setEnabled(enabled);
    emit settings_changed();
  });
  connect(asp_padded_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(asp_padded_height_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
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
  connect(spatial_contrast_check_, &QCheckBox::toggled, this, [this](bool enabled) {
    spatial_contrast_window_size_->setEnabled(enabled && get_spatial_contrast_dimensions() != 1);
    spatial_contrast_dimensions_->setEnabled(enabled);
    spatial_contrast_method_->setEnabled(enabled && get_spatial_contrast_dimensions() != 1);
    spatial_contrast_display_max_->setEnabled(enabled);
    spatial_contrast_invert_display_->setEnabled(enabled);
    update_time_window_limits();
    emit settings_changed();
    const float display_max = enabled ? static_cast<float>(spatial_contrast_display_max_->value())
                                      : 1.0f;
    emit spatial_contrast_display_range_changed(0.0f, display_max);
    emit spatial_contrast_display_inversion_changed(
        enabled && get_image_mode() == "Processed" &&
        is_spatial_contrast_display_inverted());
  });
  connect(spatial_contrast_window_size_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(spatial_contrast_dimensions_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          [this](int) {
            set_spatial_contrast_dimensions(get_spatial_contrast_dimensions());
            emit settings_changed();
          });
  connect(spatial_contrast_method_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ImageRenderingWidget::settings_changed);
  connect(spatial_contrast_display_max_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          [this](double value) {
            if (spatial_contrast_check_->isChecked())
              emit spatial_contrast_display_range_changed(0.0f, static_cast<float>(value));
          });
  connect(spatial_contrast_invert_display_, &QCheckBox::toggled, this, [this](bool inverted) {
    emit spatial_contrast_display_inversion_changed(
        spatial_contrast_check_->isChecked() && get_image_mode() == "Processed" && inverted);
  });
  connect(autofocus_widget_, &AutoFocusWidget::settings_changed, this, [this]() {
    set_angular_spectrum_enabled(!autofocus_widget_->is_enabled());
    emit settings_changed();
  });
}

void ImageRenderingWidget::update_asp_padding_visibility() {
  const bool visible = space_transform_combo_->currentText() == "Angular Spectrum";
  asp_padding_check_->setVisible(visible);
  asp_padding_controls_->setVisible(visible);
  asp_padding_controls_->setEnabled(asp_padding_check_->isChecked());
}

void ImageRenderingWidget::set_angular_spectrum_enabled(bool enabled) {
  constexpr auto kAngularSpectrum = "Angular Spectrum";
  constexpr auto kFresnel         = "Fresnel Diffraction";

  const int index = space_transform_combo_->findText(kAngularSpectrum);
  auto     *model = qobject_cast<QStandardItemModel *>(space_transform_combo_->model());
  if (index >= 0 && model != nullptr) {
    if (auto *item = model->item(index); item != nullptr) {
      item->setEnabled(enabled);
    }
  }

  if (!enabled && space_transform_combo_->currentText() == kAngularSpectrum) {
    space_transform_combo_->setCurrentText(kFresnel);
  }
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
  clear_validation_error(batch_size_spin_);
  clear_validation_error(time_stride_spin_);
  clear_validation_error(filter_2d_check_);
  clear_validation_error(filter_2d_inner_spin_);
  clear_validation_error(filter_2d_outer_spin_);
  clear_validation_error(time_window_spin_);
  clear_validation_error(space_transform_combo_);
  clear_validation_error(asp_padding_check_);
  clear_validation_error(asp_padded_width_spin_);
  clear_validation_error(asp_padded_height_spin_);
  clear_validation_error(time_transform_combo_);
  clear_validation_error(convolution_combo_);
  clear_validation_error(spatial_contrast_window_size_);
  clear_validation_error(spatial_contrast_dimensions_);
  clear_validation_error(spatial_contrast_method_);
}

} // namespace holovibes::ui
