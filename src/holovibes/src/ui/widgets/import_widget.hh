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
#include <QStringList>

class QStackedLayout;

namespace holovibes::ui {

class ImportWidget : public QGroupBox {
  Q_OBJECT

public:
  explicit ImportWidget(QWidget *parent = nullptr);

  // Getters for settings
  bool    is_camera_mode() const;
  QString get_file_path() const;
  int     get_fps() const;
  int     get_start_index() const;
  int     get_end_index() const;
  QString get_load_method() const;
  QString get_camera_type() const;
  QString get_camera_config() const;

  // Setters
  void set_file_path(const QString &path);
  void set_start_index(int value);
  void set_end_index(int value);
  void set_end_index_range(int min, int max);
  void set_load_method(const QString &method);
  void set_camera_type(const QString &type);
  void set_camera_mode(bool enabled);

  // Control button state
  void set_start_enabled(bool enabled);
  void set_stop_enabled(bool enabled);
  bool is_stop_enabled() const;

  // Style management for validation
  void clear_validation_styles();
  void mark_file_invalid();
  void mark_start_index_invalid();
  void mark_end_index_invalid();
  void mark_camera_config_invalid();

  // Access to widgets for connection setup
  QLineEdit   *file_line_edit();
  QPushButton *browse_button();
  QPushButton *start_button();
  QPushButton *stop_button();
  QSpinBox    *fps_spin();
  QSpinBox    *start_index_spin();
  QSpinBox    *end_index_spin();
  QComboBox   *load_method_combo();
  QCheckBox   *cam_check();
  QComboBox   *camera_combo();
  QComboBox   *camera_config_combo();

signals:
  void start_clicked();
  void stop_clicked();
  void browse_clicked();
  void settings_changed();

private:
  void        setup_ui();
  void        connect_signals();
  QWidget    *create_file_page();
  QWidget    *create_camera_page();
  QStringList load_available_camera_configs();

  // File mode widgets
  QLineEdit   *file_line_edit_;
  QPushButton *browse_button_;
  QSpinBox    *fps_spin_;
  QSpinBox    *start_index_spin_;
  QSpinBox    *end_index_spin_;
  QComboBox   *load_method_combo_;

  // Camera mode widgets
  QCheckBox *cam_check_;
  QComboBox *camera_combo_;
  QComboBox *camera_config_combo_;

  // Control buttons
  QPushButton *start_button_;
  QPushButton *stop_button_;

  // Layout management
  QStackedLayout *stack_;
};

} // namespace holovibes::ui