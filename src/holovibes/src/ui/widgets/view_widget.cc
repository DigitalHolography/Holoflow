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

#include "ui/widgets/view_widget.hh"

#include <QGridLayout>
#include <QLabel>
#include <QSpacerItem>
#include <QVBoxLayout>

namespace holovibes::ui {

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

ViewWidget::ViewWidget(QWidget *parent) : QGroupBox("View", parent) {
  setup_ui();
  connect_signals();
}

QString ViewWidget::get_image_type() const { return image_type_combo_->currentText(); }
bool    ViewWidget::is_cuts_3d_enabled() const { return cuts_3d_check_->isChecked(); }
bool    ViewWidget::is_fft_shift_enabled() const { return fft_shift_check_->isChecked(); }
bool    ViewWidget::is_raw_view_enabled() const { return raw_view_check_->isChecked(); }
bool    ViewWidget::is_raw_spectrum_view_enabled() const {
  return raw_spectrum_view_check_->isChecked();
}
bool ViewWidget::is_process_spectrum_view_enabled() const {
  return process_spectrum_view_check_->isChecked();
}
int     ViewWidget::get_x_origin() const { return x_spin_->value(); }
int     ViewWidget::get_x_width() const { return x_width_spin_->value(); }
int     ViewWidget::get_y_origin() const { return y_spin_->value(); }
int     ViewWidget::get_y_width() const { return y_width_spin_->value(); }
int     ViewWidget::get_z_origin() const { return z_spin_->value(); }
int     ViewWidget::get_z_width() const { return z_width_spin_->value(); }
QString ViewWidget::get_view_kind() const { return kind_combo_->currentText(); }
int     ViewWidget::get_accumulation() const { return accumulation_spin_->value(); }
int     ViewWidget::get_range_start() const { return range_start_spin_->value(); }
int     ViewWidget::get_range_end() const { return range_end_spin_->value(); }
bool    ViewWidget::is_registration_enabled() const { return registration_check_->isChecked(); }
double  ViewWidget::get_registration_radius() const { return registration_radius_->value(); }
bool    ViewWidget::is_reticle_enabled() const { return reticle_check_->isChecked(); }
double  ViewWidget::get_reticle_radius() const { return reticle_radius_->value(); }
bool    ViewWidget::is_pct_enabled() const { return pct_check_->isChecked(); }
double  ViewWidget::get_pct_radius() const { return pct_radius_->value(); }

// Setters
void ViewWidget::set_x_origin(int value) { x_spin_->setValue(value); }
void ViewWidget::set_x_width(int value) { x_width_spin_->setValue(value); }
void ViewWidget::set_y_origin(int value) { y_spin_->setValue(value); }
void ViewWidget::set_y_width(int value) { y_width_spin_->setValue(value); }
void ViewWidget::set_z_origin(int value) { z_spin_->setValue(value); }
void ViewWidget::set_z_width(int value) { z_width_spin_->setValue(value); }
void ViewWidget::set_cuts_3d_enabled(bool enabled) { cuts_3d_check_->setChecked(enabled); }
void ViewWidget::set_fft_shift_enabled(bool enabled) { fft_shift_check_->setChecked(enabled); }
void ViewWidget::set_accumulation(int value) { accumulation_spin_->setValue(value); }
void ViewWidget::set_reticle_radius(int value) { reticle_radius_->setValue(value); }
void ViewWidget::set_registration_enabled(bool enabled) {
  registration_check_->setChecked(enabled);
}
void ViewWidget::set_registration_radius(int value) { registration_radius_->setValue(value); }
void ViewWidget::set_pct_enabled(bool enabled) { pct_check_->setChecked(enabled); }
void ViewWidget::set_pct_radius(double value) { pct_radius_->setValue(value); }

// Validation
void ViewWidget::mark_z_invalid() {
  z_spin_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
}
void ViewWidget::mark_z_width_invalid() {
  z_width_spin_->setStyleSheet("background-color: rgba(255, 0, 0, 50);");
}

// Access to widgets for connection setup
QComboBox      *ViewWidget::image_type_combo() { return image_type_combo_; }
QCheckBox      *ViewWidget::cuts_3d_check() { return cuts_3d_check_; }
QCheckBox      *ViewWidget::fft_shift_check() { return fft_shift_check_; }
QCheckBox      *ViewWidget::raw_view_check() { return raw_view_check_; }
QCheckBox      *ViewWidget::raw_spectrum_view_check() { return raw_spectrum_view_check_; }
QCheckBox      *ViewWidget::process_spectrum_view_check() { return process_spectrum_view_check_; }
QSpinBox       *ViewWidget::x_spin() { return x_spin_; }
QSpinBox       *ViewWidget::x_width_spin() { return x_width_spin_; }
QSpinBox       *ViewWidget::y_spin() { return y_spin_; }
QSpinBox       *ViewWidget::y_width_spin() { return y_width_spin_; }
QSpinBox       *ViewWidget::z_spin() { return z_spin_; }
QSpinBox       *ViewWidget::z_width_spin() { return z_width_spin_; }
QComboBox      *ViewWidget::kind_combo() { return kind_combo_; }
QSpinBox       *ViewWidget::accumulation_spin() { return accumulation_spin_; }
QSpinBox       *ViewWidget::range_start_spin() { return range_start_spin_; }
QSpinBox       *ViewWidget::range_end_spin() { return range_end_spin_; }
QCheckBox      *ViewWidget::registration_check() { return registration_check_; }
QDoubleSpinBox *ViewWidget::registration_radius() { return registration_radius_; }
QCheckBox      *ViewWidget::reticle_check() { return reticle_check_; }
QDoubleSpinBox *ViewWidget::reticle_radius() { return reticle_radius_; }
QCheckBox      *ViewWidget::pct_check() { return pct_check_; }
QDoubleSpinBox *ViewWidget::pct_radius() { return pct_radius_; }

void ViewWidget::update_3d_cut_controls(bool enabled) {
  x_spin_->setEnabled(enabled);
  x_width_spin_->setEnabled(enabled);
  y_spin_->setEnabled(enabled);
  y_width_spin_->setEnabled(enabled);
}

void ViewWidget::setup_ui() {
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

  add_combo_row("Image Type:", image_type_combo_, QStringList{"Magnitude", "Phase"});

  cuts_3d_check_ = new QCheckBox("3D Cuts", this);
  layout->addWidget(cuts_3d_check_, row, 0);

  fft_shift_check_ = new QCheckBox("FFT Shift", this);
  layout->addWidget(fft_shift_check_, row, 1);
  ++row;

  raw_view_check_ = new QCheckBox("Raw View", this);
  layout->addWidget(raw_view_check_, row, 0);
  ++row;

  raw_spectrum_view_check_ = new QCheckBox("Raw Spectrum View", this);
  layout->addWidget(raw_spectrum_view_check_, row, 0);

  process_spectrum_view_check_ = new QCheckBox("Processed Spectrum View", this);
  layout->addWidget(process_spectrum_view_check_, row, 1);
  ++row;

  auto *axes_layout  = new QGridLayout();
  int   axis_row     = 0;
  auto  add_axis_row = [&](const QString &axis, QSpinBox *&origin_spin, QSpinBox *&width_spin) {
    axes_layout->addWidget(new QLabel(axis + ":", this), axis_row, 0);
    origin_spin = create_spin_box(this, 0, kLargeSpinMax, 0);
    axes_layout->addWidget(origin_spin, axis_row, 1);
    axes_layout->addWidget(new QLabel("Width:", this), axis_row, 2);
    width_spin = create_spin_box(this, 1, kLargeSpinMax, 1);
    axes_layout->addWidget(width_spin, axis_row, 3);
    ++axis_row;
  };

  add_axis_row("X", x_spin_, x_width_spin_);
  add_axis_row("Y", y_spin_, y_width_spin_);
  add_axis_row("Z", z_spin_, z_width_spin_);
  layout->addLayout(axes_layout, row, 0, 1, 2);
  ++row;

  add_combo_row("View Kind:", kind_combo_, QStringList{"XY", "XZ", "YZ"});
  add_spin_row("Output image accumulation:", accumulation_spin_, 1, kLargeSpinMax, 1);

  auto *brightness_group = new QGroupBox("Brightness/Contrast", this);
  brightness_group->setCheckable(true);
  brightness_group->setChecked(true);
  auto *bright_layout = new QGridLayout(brightness_group);

  auto *range_layout = new QGridLayout();
  range_layout->addWidget(new QLabel("Range:", brightness_group), 0, 0);
  range_start_spin_ = create_spin_box(brightness_group, 1, kLargeSpinMax, 0);
  range_layout->addWidget(range_start_spin_, 0, 1);
  range_end_spin_ = create_spin_box(brightness_group, 1, kLargeSpinMax, 255);
  range_layout->addWidget(range_end_spin_, 0, 2);
  bright_layout->addLayout(range_layout, 1, 0, 1, 2);

  reticle_check_ = new QCheckBox("Display reticle", brightness_group);
  bright_layout->addWidget(reticle_check_, 2, 0);
  reticle_radius_ = create_double_spin_box(brightness_group, 0.05, 1.0, 0.05, 1.0);
  bright_layout->addWidget(reticle_radius_, 2, 1);
  pct_check_ = new QCheckBox("PCT", brightness_group);
  bright_layout->addWidget(pct_check_, 3, 0);
  pct_radius_ = create_double_spin_box(brightness_group, 0.05, 1.0, 0.05, 1.0);
  bright_layout->addWidget(pct_radius_, 3, 1);

  registration_check_ = new QCheckBox("Registration", brightness_group);
  bright_layout->addWidget(registration_check_, 4, 0);
  registration_radius_ = create_double_spin_box(brightness_group, 0.05, 1.0, 0.05, 1.0);
  bright_layout->addWidget(registration_radius_, 4, 1);

  layout->addWidget(brightness_group, row, 0, 1, 2);
  ++row;

  layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), row, 0, 1,
                  2);
  update_3d_cut_controls(cuts_3d_check_->isChecked());
}

void ViewWidget::connect_signals() {
  // Special signals for UI changes
  connect(cuts_3d_check_, &QCheckBox::toggled, this, &ViewWidget::cuts_3d_toggled);
  connect(raw_view_check_, &QCheckBox::toggled, this, &ViewWidget::raw_view_toggled);
  connect(raw_spectrum_view_check_, &QCheckBox::toggled, this,
          &ViewWidget::raw_spectrum_view_toggled);
  connect(process_spectrum_view_check_, &QCheckBox::toggled, this,
          &ViewWidget::process_spectrum_view_toggled);
  connect(reticle_check_, &QCheckBox::toggled, this, &ViewWidget::reticle_toggled);
  connect(reticle_radius_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          &ViewWidget::reticle_radius_changed);
  connect(cuts_3d_check_, &QCheckBox::toggled, this, &ViewWidget::update_3d_cut_controls);

  // Emit settings_changed for all control changes
  connect(image_type_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ViewWidget::settings_changed);
  connect(cuts_3d_check_, &QCheckBox::toggled, this, &ViewWidget::settings_changed);
  connect(fft_shift_check_, &QCheckBox::toggled, this, &ViewWidget::settings_changed);
  connect(raw_view_check_, &QCheckBox::toggled, this, &ViewWidget::settings_changed);
  connect(raw_spectrum_view_check_, &QCheckBox::toggled, this, &ViewWidget::settings_changed);
  connect(process_spectrum_view_check_, &QCheckBox::toggled, this, &ViewWidget::settings_changed);
  connect(x_spin_, qOverload<int>(&QSpinBox::valueChanged), this, &ViewWidget::settings_changed);
  connect(x_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::settings_changed);
  connect(y_spin_, qOverload<int>(&QSpinBox::valueChanged), this, &ViewWidget::settings_changed);
  connect(y_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::settings_changed);
  connect(z_spin_, qOverload<int>(&QSpinBox::valueChanged), this, &ViewWidget::settings_changed);
  connect(z_width_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::settings_changed);
  connect(kind_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          &ViewWidget::settings_changed);
  connect(accumulation_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::settings_changed);
  connect(range_start_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::settings_changed);
  connect(range_end_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &ViewWidget::settings_changed);
  connect(registration_check_, &QCheckBox::toggled, this, &ViewWidget::settings_changed);
  connect(registration_radius_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          &ViewWidget::settings_changed);
  connect(pct_check_, &QCheckBox::toggled, this, &ViewWidget::settings_changed);
  connect(pct_radius_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          &ViewWidget::settings_changed);
}

void ViewWidget::clear_validation_styles() {
  z_spin_->setStyleSheet("");
  z_width_spin_->setStyleSheet("");
}

} // namespace holovibes::ui