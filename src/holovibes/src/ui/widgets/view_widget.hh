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
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QSpinBox>

namespace holovibes::ui {

class ViewWidget : public QGroupBox {
  Q_OBJECT

public:
  explicit ViewWidget(QWidget *parent = nullptr);

  // Getters
  QString get_image_type() const;
  bool is_cuts_3d_enabled() const;
  bool is_fft_shift_enabled() const;
  bool is_raw_view_enabled() const;
  int get_x_origin() const;
  int get_x_width() const;
  int get_y_origin() const;
  int get_y_width() const;
  int get_z_origin() const;
  int get_z_width() const;
  QString get_view_kind() const;
  int get_accumulation() const;
  bool is_auto_brightness() const;
  bool is_invert() const;
  int get_range_start() const;
  int get_range_end() const;
  bool is_registration_enabled() const;
  double get_registration_radius() const;
  bool is_reticle_enabled() const;
  double get_reticle_radius() const;

  // Setters
  void set_x_origin(int value);
  void set_x_width(int value);
  void set_y_origin(int value);
  void set_y_width(int value);
  void set_z_origin(int value);
  void set_z_width(int value);
  void set_cuts_3d_enabled(bool enabled);
  void set_fft_shift_enabled(bool enabled);
  void set_accumulation(int value);
  void set_reticle_radius(int value);
  void set_registration_enabled(bool enabled);
  void set_registration_radius(int value);

  // Validation
  void clear_validation_styles();
  void mark_z_invalid();
  void mark_z_width_invalid();

  // Access to widgets for connection setup
  QComboBox *image_type_combo();
  QCheckBox *cuts_3d_check();
  QCheckBox *fft_shift_check();
  QCheckBox *raw_view_check();
  QSpinBox *x_spin();
  QSpinBox *x_width_spin();
  QSpinBox *y_spin();
  QSpinBox *y_width_spin();
  QSpinBox *z_spin();
  QSpinBox *z_width_spin();
  QComboBox *kind_combo();
  QSpinBox *accumulation_spin();
  QCheckBox *auto_check();
  QCheckBox *invert_check();
  QSpinBox *range_start_spin();
  QSpinBox *range_end_spin();
  QCheckBox *registration_check();
  QDoubleSpinBox *registration_radius();
  QCheckBox *reticle_check();
  QDoubleSpinBox *reticle_radius();

signals:
  void settings_changed();
  void cuts_3d_toggled(bool enabled);
  void raw_view_toggled(bool enabled);
  void reticle_toggled(bool enabled);
  void reticle_radius_changed(double radius);

private:
  void setup_ui();
  void connect_signals();

  QComboBox *image_type_combo_;
  QCheckBox *cuts_3d_check_;
  QCheckBox *fft_shift_check_;
  QCheckBox *raw_view_check_;
  QSpinBox *x_spin_;
  QSpinBox *x_width_spin_;
  QSpinBox *y_spin_;
  QSpinBox *y_width_spin_;
  QSpinBox *z_spin_;
  QSpinBox *z_width_spin_;
  QComboBox *kind_combo_;
  QSpinBox *accumulation_spin_;
  QCheckBox *auto_check_;
  QCheckBox *invert_check_;
  QSpinBox *range_start_spin_;
  QSpinBox *range_end_spin_;
  QCheckBox *registration_check_;
  QDoubleSpinBox *registration_radius_;
  QCheckBox *reticle_check_;
  QDoubleSpinBox *reticle_radius_;
};

} // namespace holovibes::ui