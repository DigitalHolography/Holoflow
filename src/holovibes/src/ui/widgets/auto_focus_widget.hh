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

namespace holovibes::ui {

class AutoFocusWidget : public QGroupBox {
  Q_OBJECT

public:
  explicit AutoFocusWidget(QWidget *parent = nullptr);

  // Getters
  int    get_nb_subaps() const;
  int    get_max_iter() const;
  double get_tolerance() const;
  double get_gain() const;
  bool   is_enabled() const;

  // Setters
  void set_nb_subaps(int value);
  void set_max_iter(int value);
  void set_tolerance(double value);
  void set_gain(double value);
  void set_enabled(bool enabled);

  // Access to widgets for connection setup
  QSpinBox       *nb_subaps_spin();
  QSpinBox       *max_iter_spin();
  QDoubleSpinBox *tolerance_spin();
  QDoubleSpinBox *gain_spin();

signals:
  void settings_changed();

private:
  void setup_ui();
  void connect_signals();

  void update_visibility(bool enabled);

  QSpinBox       *nb_subaps_spin_;
  QSpinBox       *max_iter_spin_;
  QDoubleSpinBox *tolerance_spin_;
  QDoubleSpinBox *gain_spin_;
  QWidget        *content_container_;
};

} // namespace holovibes::ui