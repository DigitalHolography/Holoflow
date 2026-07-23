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
  bool    is_flatfield_enabled() const;
  int     get_x_origin() const;
  int     get_x_width() const;
  int     get_y_origin() const;
  int     get_y_width() const;
  int     get_z_origin() const;
  int     get_z_width() const;
  int     get_accumulation() const;
  double  get_flatfield_cutoff_period_um() const;
  int     get_range_start() const;
  int     get_range_end() const;
  bool    is_registration_enabled() const;
  double  get_registration_radius() const;
  bool    is_reticle_enabled() const;
  double  get_reticle_radius() const;
  bool    is_pct_enabled() const;
  double  get_pct_radius() const;

  // Setters
  void set_xy_extent(int width, int height);
  void set_z_origin(int value);
  void set_z_width(int value);
  void set_flatfield_enabled(bool enabled);
  void set_accumulation(int value);
  void set_flatfield_cutoff_period_um(double value);
  void set_reticle_radius(int value);
  void set_registration_enabled(bool enabled);
  void set_registration_radius(int value);
  void set_pct_enabled(bool enabled);
  void set_pct_radius(double value);

  // Validation
  void clear_validation_styles();
  void mark_z_invalid();
  void mark_z_width_invalid();
  void mark_flatfield_cutoff_period_invalid();
  void mark_registration_invalid();

  // Access to widgets for connection setup
  QComboBox      *image_type_combo();
  QCheckBox      *flatfield_check();
  QGroupBox      *post_processing_group();
  QSpinBox       *z_spin();
  QSpinBox       *z_width_spin();
  QSpinBox       *accumulation_spin();
  QDoubleSpinBox *flatfield_cutoff_period_um();
  QSpinBox       *range_start_spin();
  QSpinBox       *range_end_spin();
  QCheckBox      *registration_check();
  QDoubleSpinBox *registration_radius();
  QCheckBox      *reticle_check();
  QDoubleSpinBox *reticle_radius();
  QCheckBox      *pct_check();
  QDoubleSpinBox *pct_radius();

signals:
  void settings_changed();
  void reticle_toggled(bool enabled);
  void reticle_radius_changed(double radius);

private:
  void setup_ui();
  void connect_signals();

  QComboBox      *image_type_combo_;
  QSpinBox       *x_spin_;
  QSpinBox       *x_width_spin_;
  QSpinBox       *y_spin_;
  QSpinBox       *y_width_spin_;
  QSpinBox       *z_spin_;
  QSpinBox       *z_width_spin_;
  QGroupBox      *post_processing_group_;
  QSpinBox       *accumulation_spin_;
  QCheckBox      *flatfield_check_;
  QDoubleSpinBox *flatfield_cutoff_period_um_;
  QSpinBox       *range_start_spin_;
  QSpinBox       *range_end_spin_;
  QCheckBox      *registration_check_;
  QDoubleSpinBox *registration_radius_;
  QCheckBox      *reticle_check_;
  QDoubleSpinBox *reticle_radius_;
  QCheckBox      *pct_check_;
  QDoubleSpinBox *pct_radius_;
};

} // namespace holovibes::ui
