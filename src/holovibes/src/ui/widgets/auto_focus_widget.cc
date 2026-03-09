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
  spin_box->setSingleStep(0.0);
  spin_box->setValue(value);
  spin_box->setEnabled(false);
  spin_box->setSuffix(" rad");
  return spin_box;
}

QCheckBox *create_checkbox(QWidget *parent, const QString &text, bool checked = false) {
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

void AutoFocusWidget::set_z2(double value) { z2_spin_->setValue(value); }
void AutoFocusWidget::set_z3(double value) { z3_spin_->setValue(value); }
void AutoFocusWidget::set_z4(double value) { z4_spin_->setValue(value); }
void AutoFocusWidget::set_z5(double value) { z5_spin_->setValue(value); }
void AutoFocusWidget::set_z6(double value) { z6_spin_->setValue(value); }

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

QCheckBox *AutoFocusWidget::reconstructed_phase_checkbox() {
  return reconstructed_phase_checkbox_;
}

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
    inner_layout->addWidget(widget, row, 1);
    ++row;
  };

  nb_subaps_spin_ =
      create_fixed_spin_box(content_container_, kFixedNbSubaps,
                            "Only 5 subapertures are supported for now.");
  add_label_widget_row("Nb Subaps:", nb_subaps_spin_);

  nb_iter_spin_ =
      create_fixed_spin_box(content_container_, kFixedNbIter,
                            "Only one iteration is supported for now.");
  add_label_widget_row("Nb Iter:", nb_iter_spin_);

  z2_spin_ = create_display_double_spin_box(content_container_, 0.0);
  z3_spin_ = create_display_double_spin_box(content_container_, 0.0);
  z4_spin_ = create_display_double_spin_box(content_container_, 0.0);
  z5_spin_ = create_display_double_spin_box(content_container_, 0.0);
  z6_spin_ = create_display_double_spin_box(content_container_, 0.0);

  add_label_widget_row("Z2 - Tilt X:", z2_spin_);
  add_label_widget_row("Z3 - Tilt Y:", z3_spin_);
  add_label_widget_row("Z4 - Defocus:", z4_spin_);
  add_label_widget_row("Z5 - Astigmatism 45°:", z5_spin_);
  add_label_widget_row("Z6 - Astigmatism 0°:", z6_spin_);

  reconstructed_phase_checkbox_ =
      create_checkbox(content_container_, "Display reconstructed phase");
  shack_hartmann_sensor_view_checkbox_ =
      create_checkbox(content_container_, "Display Shack-Hartmann sensor view");
  cross_correlation_view_checkbox_ =
      create_checkbox(content_container_, "Display cross-correlation view");

  inner_layout->addWidget(reconstructed_phase_checkbox_, row++, 0, 1, 2);
  inner_layout->addWidget(shack_hartmann_sensor_view_checkbox_, row++, 0, 1, 2);
  inner_layout->addWidget(cross_correlation_view_checkbox_, row++, 0, 1, 2);

  inner_layout->addItem(
      new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), row, 0, 1, 2);
}

void AutoFocusWidget::connect_signals() {
  connect(this, &QGroupBox::toggled, this, &AutoFocusWidget::settings_changed);

  connect(reconstructed_phase_checkbox_, &QCheckBox::toggled, this,
          &AutoFocusWidget::settings_changed);
  connect(shack_hartmann_sensor_view_checkbox_, &QCheckBox::toggled, this,
          &AutoFocusWidget::settings_changed);
  connect(cross_correlation_view_checkbox_, &QCheckBox::toggled, this,
          &AutoFocusWidget::settings_changed);
}

} // namespace holovibes::ui