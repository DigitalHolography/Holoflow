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

#include "ui/widgets/auto_focus_widget.hh"

#include <algorithm>

#include <QGridLayout>
#include <QLabel>
#include <QSizePolicy>
#include <QSpacerItem>
#include <QVBoxLayout>

namespace holovibes::ui {

namespace {

constexpr int kFixedNbSubaps = 5;
constexpr int kFixedNbIter   = 1;

QSpinBox *create_fixed_spin_box(QWidget *parent, int value, const QString &tooltip) {
  auto *spin_box = new QSpinBox(parent);
  spin_box->setRange(value, value);
  spin_box->setValue(value);
  spin_box->setEnabled(false);
  spin_box->setToolTip(tooltip);
  return spin_box;
}

QDoubleSpinBox *create_display_double_spin_box(QWidget *parent, double value, int decimals = 6) {
  auto *spin_box = new QDoubleSpinBox(parent);
  spin_box->setRange(-1e9, 1e9);
  spin_box->setDecimals(decimals);
  spin_box->setButtonSymbols(QAbstractSpinBox::NoButtons);
  spin_box->setReadOnly(true);
  spin_box->setValue(value);
  spin_box->setSuffix(" rad");
  return spin_box;
}

QCheckBox *create_checkbox(QWidget *parent, bool checked = false, const QString &text = {}) {
  auto *checkbox = new QCheckBox(text, parent);
  checkbox->setChecked(checked);
  return checkbox;
}

} // namespace

AutoFocusWidget::AutoFocusWidget(QWidget *parent) : QGroupBox("Auto Focus", parent) {
  setCheckable(true);
  setup_ui();
  connect_signals();
  setChecked(false);
}

// Fixed configuration
int  AutoFocusWidget::get_nb_subaps() const { return nb_subaps_spin_->value(); }
int  AutoFocusWidget::get_nb_iter() const { return nb_iter_spin_->value(); }
bool AutoFocusWidget::is_enabled() const { return isChecked(); }

// Zernike values
double AutoFocusWidget::get_z2() const { return z2_spin_->value(); }
double AutoFocusWidget::get_z3() const { return z3_spin_->value(); }
double AutoFocusWidget::get_z4() const { return z4_spin_->value(); }
double AutoFocusWidget::get_z5() const { return z5_spin_->value(); }
double AutoFocusWidget::get_z6() const { return z6_spin_->value(); }
double AutoFocusWidget::get_z7() const { return z7_spin_->value(); }
double AutoFocusWidget::get_z8() const { return z8_spin_->value(); }
double AutoFocusWidget::get_z9() const { return z9_spin_->value(); }
double AutoFocusWidget::get_z10() const { return z10_spin_->value(); }

void AutoFocusWidget::set_z2(double value) { z2_spin_->setValue(value); }
void AutoFocusWidget::set_z3(double value) { z3_spin_->setValue(value); }
void AutoFocusWidget::set_z4(double value) { z4_spin_->setValue(value); }
void AutoFocusWidget::set_z5(double value) { z5_spin_->setValue(value); }
void AutoFocusWidget::set_z6(double value) { z6_spin_->setValue(value); }
void AutoFocusWidget::set_z7(double value) { z7_spin_->setValue(value); }
void AutoFocusWidget::set_z8(double value) { z8_spin_->setValue(value); }
void AutoFocusWidget::set_z9(double value) { z9_spin_->setValue(value); }
void AutoFocusWidget::set_z10(double value) { z10_spin_->setValue(value); }

void AutoFocusWidget::set_zernike_value(int noll_index, double value) {
  switch (noll_index) {
  case 2:
    set_z2(value);
    break;
  case 3:
    set_z3(value);
    break;
  case 4:
    set_z4(value);
    break;
  case 5:
    set_z5(value);
    break;
  case 6:
    set_z6(value);
    break;
  case 7:
    set_z7(value);
    break;
  case 8:
    set_z8(value);
    break;
  case 9:
    set_z9(value);
    break;
  case 10:
    set_z10(value);
    break;
  default:
    break;
  }
}

void AutoFocusWidget::set_zernike_values(const std::vector<int>   &noll_indexes,
                                         const std::vector<float> &values) {
  reset_zernike_values();

  const auto count = std::min(noll_indexes.size(), values.size());
  for (size_t i = 0; i < count; ++i) {
    set_zernike_value(noll_indexes[i], values[i]);
  }
}

void AutoFocusWidget::reset_zernike_values() {
  set_z2(0.0);
  set_z3(0.0);
  set_z4(0.0);
  set_z5(0.0);
  set_z6(0.0);
  set_z7(0.0);
  set_z8(0.0);
  set_z9(0.0);
  set_z10(0.0);
}

// Per-coefficient enable flags
bool AutoFocusWidget::is_z2_enabled() const { return z2_checkbox_->isChecked(); }
bool AutoFocusWidget::is_z3_enabled() const { return z3_checkbox_->isChecked(); }
bool AutoFocusWidget::is_z4_enabled() const { return z4_checkbox_->isChecked(); }
bool AutoFocusWidget::is_z5_enabled() const { return z5_checkbox_->isChecked(); }
bool AutoFocusWidget::is_z6_enabled() const { return z6_checkbox_->isChecked(); }
bool AutoFocusWidget::is_z7_enabled() const { return z7_checkbox_->isChecked(); }
bool AutoFocusWidget::is_z8_enabled() const { return z8_checkbox_->isChecked(); }
bool AutoFocusWidget::is_z9_enabled() const { return z9_checkbox_->isChecked(); }
bool AutoFocusWidget::is_z10_enabled() const { return z10_checkbox_->isChecked(); }

void AutoFocusWidget::set_z2_enabled(bool enabled) { z2_checkbox_->setChecked(enabled); }
void AutoFocusWidget::set_z3_enabled(bool enabled) { z3_checkbox_->setChecked(enabled); }
void AutoFocusWidget::set_z4_enabled(bool enabled) { z4_checkbox_->setChecked(enabled); }
void AutoFocusWidget::set_z5_enabled(bool enabled) { z5_checkbox_->setChecked(enabled); }
void AutoFocusWidget::set_z6_enabled(bool enabled) { z6_checkbox_->setChecked(enabled); }
void AutoFocusWidget::set_z7_enabled(bool enabled) { z7_checkbox_->setChecked(enabled); }
void AutoFocusWidget::set_z8_enabled(bool enabled) { z8_checkbox_->setChecked(enabled); }
void AutoFocusWidget::set_z9_enabled(bool enabled) { z9_checkbox_->setChecked(enabled); }
void AutoFocusWidget::set_z10_enabled(bool enabled) { z10_checkbox_->setChecked(enabled); }

// Visualization toggles
bool AutoFocusWidget::show_reconstructed_phase() const {
  return reconstructed_phase_checkbox_->isChecked();
}

bool AutoFocusWidget::show_shack_hartmann_sensor_view() const {
  return shack_hartmann_sensor_view_checkbox_->isChecked();
}

bool AutoFocusWidget::show_cross_correlation_view() const {
  return cross_correlation_view_checkbox_->isChecked();
}

void AutoFocusWidget::set_show_reconstructed_phase(bool checked) {
  reconstructed_phase_checkbox_->setChecked(checked);
}

void AutoFocusWidget::set_show_shack_hartmann_sensor_view(bool checked) {
  shack_hartmann_sensor_view_checkbox_->setChecked(checked);
}

void AutoFocusWidget::set_show_cross_correlation_view(bool checked) {
  cross_correlation_view_checkbox_->setChecked(checked);
}

void AutoFocusWidget::set_enabled(bool enabled) { setChecked(enabled); }

// Widget accessors
QCheckBox *AutoFocusWidget::z2_checkbox() { return z2_checkbox_; }
QCheckBox *AutoFocusWidget::z3_checkbox() { return z3_checkbox_; }
QCheckBox *AutoFocusWidget::z4_checkbox() { return z4_checkbox_; }
QCheckBox *AutoFocusWidget::z5_checkbox() { return z5_checkbox_; }
QCheckBox *AutoFocusWidget::z6_checkbox() { return z6_checkbox_; }
QCheckBox *AutoFocusWidget::z7_checkbox() { return z7_checkbox_; }
QCheckBox *AutoFocusWidget::z8_checkbox() { return z8_checkbox_; }
QCheckBox *AutoFocusWidget::z9_checkbox() { return z9_checkbox_; }
QCheckBox *AutoFocusWidget::z10_checkbox() { return z10_checkbox_; }

QCheckBox *AutoFocusWidget::reconstructed_phase_checkbox() { return reconstructed_phase_checkbox_; }

QCheckBox *AutoFocusWidget::shack_hartmann_sensor_view_checkbox() {
  return shack_hartmann_sensor_view_checkbox_;
}

QCheckBox *AutoFocusWidget::cross_correlation_view_checkbox() {
  return cross_correlation_view_checkbox_;
}

void AutoFocusWidget::setup_ui() {
  content_container_ = new QWidget(this);

  auto *outer_layout = new QVBoxLayout(this);
  auto *inner_layout = new QGridLayout(content_container_);

  outer_layout->addWidget(content_container_);

  int row = 0;

  const auto add_label_widget_row = [&](const QString &label, QWidget *widget) {
    inner_layout->addWidget(new QLabel(label, content_container_), row, 0);
    inner_layout->addWidget(widget, row, 1, 1, 2);
    ++row;
  };

  const auto add_zernike_row = [&](const QString &label, QCheckBox *&checkbox,
                                   QDoubleSpinBox *&spin_box) {
    inner_layout->addWidget(new QLabel(label, content_container_), row, 0);

    checkbox = create_checkbox(content_container_, false);
    inner_layout->addWidget(checkbox, row, 1);

    spin_box = create_display_double_spin_box(content_container_, 0.0);
    inner_layout->addWidget(spin_box, row, 2);

    ++row;
  };

  nb_subaps_spin_ = create_fixed_spin_box(content_container_, kFixedNbSubaps,
                                          "Only 5 subapertures are supported for now.");
  add_label_widget_row("Nb Subaps:", nb_subaps_spin_);

  nb_iter_spin_ = create_fixed_spin_box(content_container_, kFixedNbIter,
                                        "Only one iteration is supported for now.");
  add_label_widget_row("Nb Iter:", nb_iter_spin_);

  add_zernike_row("Z2 - Tilt X:", z2_checkbox_, z2_spin_);
  add_zernike_row("Z3 - Tilt Y:", z3_checkbox_, z3_spin_);
  add_zernike_row("Z4 - Defocus:", z4_checkbox_, z4_spin_);
  add_zernike_row("Z5 - Astigmatism 45°:", z5_checkbox_, z5_spin_);
  add_zernike_row("Z6 - Astigmatism 0°:", z6_checkbox_, z6_spin_);

  add_zernike_row("Z7 - Vertical trefoil:", z7_checkbox_, z7_spin_);
  add_zernike_row("Z8 - Vertical coma:", z8_checkbox_, z8_spin_);
  add_zernike_row("Z9 - Horizontal coma:", z9_checkbox_, z9_spin_);
  add_zernike_row("Z10 - Oblique trefoil:", z10_checkbox_, z10_spin_);

  reconstructed_phase_checkbox_ =
      create_checkbox(content_container_, false, "Display reconstructed phase");
  shack_hartmann_sensor_view_checkbox_ =
      create_checkbox(content_container_, false, "Display Shack-Hartmann sensor view");
  cross_correlation_view_checkbox_ =
      create_checkbox(content_container_, false, "Display cross-correlation view");

  inner_layout->addWidget(reconstructed_phase_checkbox_, row++, 0, 1, 3);
  inner_layout->addWidget(shack_hartmann_sensor_view_checkbox_, row++, 0, 1, 3);
  inner_layout->addWidget(cross_correlation_view_checkbox_, row++, 0, 1, 3);

  inner_layout->setColumnStretch(0, 1);
  inner_layout->setColumnStretch(1, 0);
  inner_layout->setColumnStretch(2, 1);

  inner_layout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), row,
                        0, 1, 3);
}

void AutoFocusWidget::connect_signals() {
  connect(this, &QGroupBox::toggled, this, &AutoFocusWidget::settings_changed);

  connect(z2_checkbox_, &QCheckBox::toggled, this, &AutoFocusWidget::settings_changed);
  connect(z3_checkbox_, &QCheckBox::toggled, this, &AutoFocusWidget::settings_changed);
  connect(z4_checkbox_, &QCheckBox::toggled, this, &AutoFocusWidget::settings_changed);
  connect(z5_checkbox_, &QCheckBox::toggled, this, &AutoFocusWidget::settings_changed);
  connect(z6_checkbox_, &QCheckBox::toggled, this, &AutoFocusWidget::settings_changed);

  connect(z7_checkbox_, &QCheckBox::toggled, this, &AutoFocusWidget::settings_changed);
  connect(z8_checkbox_, &QCheckBox::toggled, this, &AutoFocusWidget::settings_changed);
  connect(z9_checkbox_, &QCheckBox::toggled, this, &AutoFocusWidget::settings_changed);
  connect(z10_checkbox_, &QCheckBox::toggled, this, &AutoFocusWidget::settings_changed);

  connect(reconstructed_phase_checkbox_, &QCheckBox::toggled, this,
          &AutoFocusWidget::settings_changed);
  connect(shack_hartmann_sensor_view_checkbox_, &QCheckBox::toggled, this,
          &AutoFocusWidget::settings_changed);
  connect(cross_correlation_view_checkbox_, &QCheckBox::toggled, this,
          &AutoFocusWidget::settings_changed);
}

} // namespace holovibes::ui
