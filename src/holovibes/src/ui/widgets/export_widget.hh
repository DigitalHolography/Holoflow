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
#include <QGroupBox>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>

namespace holovibes::ui {

class ExportWidget : public QGroupBox {
  Q_OBJECT

public:
  explicit ExportWidget(QWidget *parent = nullptr);

  // Getters
  QString get_image_type() const;
  QString get_file_path() const;
  QString get_tag() const;
  bool    is_frame_count_enabled() const;
  int     get_frame_count() const;

  // Setters
  void set_file_path(const QString &path);
  void set_frame_count(int count);
  void set_image_type(const QString &type);

  // Control button state
  void set_record_enabled(bool enabled);
  void set_stop_enabled(bool enabled);

  // Validation
  void clear_validation_styles();
  void mark_file_invalid();
  void mark_frames_invalid();

  // Access to widgets for connection setup
  QComboBox   *image_type_combo();
  QLineEdit   *file_line_edit();
  QPushButton *browse_button();
  QComboBox   *tag_combo();
  QCheckBox   *frames_check();
  QSpinBox    *frames_spin();
  QPushButton *record_button();
  QPushButton *stop_button();
  QPushButton *stop_fan_button();

signals:
  void record_clicked();
  void stop_clicked();
  void stop_fan_clicked();
  void browse_clicked();
  void settings_changed();

private:
  void setup_ui();
  void connect_signals();

  QComboBox   *image_type_combo_;
  QLineEdit   *file_line_edit_;
  QPushButton *browse_button_;
  QComboBox   *tag_combo_;
  QCheckBox   *frames_check_;
  QSpinBox    *frames_spin_;
  QPushButton *record_button_;
  QPushButton *stop_button_;
  QPushButton *stop_fan_button_;
};

} // namespace holovibes::ui