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
#include <QSpacerItem>
#include <QVBoxLayout>

#include "logger.hh"

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

} // namespace

AutoFocusWidget::AutoFocusWidget(QWidget *parent) : QGroupBox("Auto Focus", parent) {
  setCheckable(true);
  setup_ui();
  connect_signals();
  setChecked(false);
}

// Getters
int    AutoFocusWidget::get_nb_subaps() const { return nb_subaps_spin_->value(); }
int    AutoFocusWidget::get_max_iter() const { return max_iter_spin_->value(); }
double AutoFocusWidget::get_tolerance() const { return tolerance_spin_->value(); }
double AutoFocusWidget::get_gain() const { return gain_spin_->value(); }
bool   AutoFocusWidget::is_enabled() const { return isChecked(); }

// Setters
void AutoFocusWidget::set_nb_subaps(int value) { nb_subaps_spin_->setValue(value); }
void AutoFocusWidget::set_max_iter(int value) { max_iter_spin_->setValue(value); }
void AutoFocusWidget::set_tolerance(double value) { tolerance_spin_->setValue(value); }
void AutoFocusWidget::set_gain(double value) { gain_spin_->setValue(value); }
void AutoFocusWidget::set_enabled(bool enabled) { setChecked(enabled); }

// Access to widgets for connection setup
QSpinBox       *AutoFocusWidget::nb_subaps_spin() { return nb_subaps_spin_; }
QSpinBox       *AutoFocusWidget::max_iter_spin() { return max_iter_spin_; }
QDoubleSpinBox *AutoFocusWidget::tolerance_spin() { return tolerance_spin_; }
QDoubleSpinBox *AutoFocusWidget::gain_spin() { return gain_spin_; }

void AutoFocusWidget::setup_ui() {
  content_container_ = new QWidget(this);

  auto *inner  = new QGridLayout(content_container_);
  auto *layout = new QVBoxLayout(this);
  layout->addWidget(content_container_);

  int row = 0;

  auto add_spin_row = [&](const QString &label, QSpinBox *&spin, int minimum, int maximum,
                          int value) {
    inner->addWidget(new QLabel(label, content_container_), row, 0);
    spin = create_spin_box(content_container_, minimum, maximum, value);
    inner->addWidget(spin, row, 1);
    ++row;
  };

  auto add_double_spin_row = [&](const QString &label, QDoubleSpinBox *&spin, double minimum,
                                 double maximum, double step, double value, int decimals = 2) {
    inner->addWidget(new QLabel(label, content_container_), row, 0);
    spin = create_double_spin_box(content_container_, minimum, maximum, step, value, decimals);
    inner->addWidget(spin, row, 1);
    ++row;
  };

  add_spin_row("Nb Subaps:", nb_subaps_spin_, 1, kLargeSpinMax, 10);
  add_spin_row("Max Iterations:", max_iter_spin_, 1, kLargeSpinMax, 100);
  add_double_spin_row("Tolerance:", tolerance_spin_, 0.0, 1.0, 0.001, 0.001, 4);
  add_double_spin_row("Gain:", gain_spin_, 0.0, 10.0, 0.1, 1.0, 2);

  inner->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding), row, 0, 1,
                 2);
}

void AutoFocusWidget::connect_signals() {
  // Connect the checkable group box toggle
  connect(this, &QGroupBox::toggled, this, &AutoFocusWidget::set_enabled);
  connect(this, &QGroupBox::toggled, this, &AutoFocusWidget::settings_changed);

  // Emit settings_changed for all control changes
  connect(nb_subaps_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &AutoFocusWidget::settings_changed);
  connect(max_iter_spin_, qOverload<int>(&QSpinBox::valueChanged), this,
          &AutoFocusWidget::settings_changed);
  connect(tolerance_spin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          &AutoFocusWidget::settings_changed);
  connect(gain_spin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          &AutoFocusWidget::settings_changed);
}

} // namespace holovibes::ui