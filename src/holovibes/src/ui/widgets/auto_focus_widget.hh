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

#pragma once

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QSpinBox>
#include <QWidget>

namespace holovibes::ui {

class AutoFocusWidget : public QGroupBox {
  Q_OBJECT

public:
  explicit AutoFocusWidget(QWidget *parent = nullptr);

  // Fixed configuration
  int  get_nb_subaps() const;
  int  get_nb_iter() const;
  bool is_enabled() const;

  // Zernike values in radians
  double get_z2() const;
  double get_z3() const;
  double get_z4() const;
  double get_z5() const;
  double get_z6() const;

  void set_z2(double value);
  void set_z3(double value);
  void set_z4(double value);
  void set_z5(double value);
  void set_z6(double value);

  // Per-coefficient enable flags
  bool is_z2_enabled() const;
  bool is_z3_enabled() const;
  bool is_z4_enabled() const;
  bool is_z5_enabled() const;
  bool is_z6_enabled() const;

  void set_z2_enabled(bool enabled);
  void set_z3_enabled(bool enabled);
  void set_z4_enabled(bool enabled);
  void set_z5_enabled(bool enabled);
  void set_z6_enabled(bool enabled);

  // Visualization toggles
  bool show_reconstructed_phase() const;
  bool show_shack_hartmann_sensor_view() const;
  bool show_cross_correlation_view() const;

  void set_show_reconstructed_phase(bool checked);
  void set_show_shack_hartmann_sensor_view(bool checked);
  void set_show_cross_correlation_view(bool checked);

  void set_enabled(bool enabled);

  // Accessors for external connection/setup if needed
  QCheckBox *z2_checkbox();
  QCheckBox *z3_checkbox();
  QCheckBox *z4_checkbox();
  QCheckBox *z5_checkbox();
  QCheckBox *z6_checkbox();

  QCheckBox *reconstructed_phase_checkbox();
  QCheckBox *shack_hartmann_sensor_view_checkbox();
  QCheckBox *cross_correlation_view_checkbox();

signals:
  void settings_changed();

private:
  void setup_ui();
  void connect_signals();

  QSpinBox *nb_subaps_spin_;
  QSpinBox *nb_iter_spin_;

  QCheckBox      *z2_checkbox_;
  QCheckBox      *z3_checkbox_;
  QCheckBox      *z4_checkbox_;
  QCheckBox      *z5_checkbox_;
  QCheckBox      *z6_checkbox_;
  QDoubleSpinBox *z2_spin_;
  QDoubleSpinBox *z3_spin_;
  QDoubleSpinBox *z4_spin_;
  QDoubleSpinBox *z5_spin_;
  QDoubleSpinBox *z6_spin_;

  QCheckBox *reconstructed_phase_checkbox_;
  QCheckBox *shack_hartmann_sensor_view_checkbox_;
  QCheckBox *cross_correlation_view_checkbox_;

  QWidget *content_container_;
};

} // namespace holovibes::ui