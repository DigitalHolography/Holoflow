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

  // clang-format off
  QString current_image_type() const { return image_type_combo_->currentText(); }
  QString current_file() const { return file_line_edit_->text(); }
  QString current_tag() const { return tag_combo_->currentText(); }
  bool current_frames_check() const { return frames_check_->isChecked(); }
  int current_frames_value() const { return frames_spin_->value(); }
  // clang-format on

signals:
  void export_image_type_changed(const QString &image_type);
  void export_file_selected(const QString &file_path);
  void export_tag_changed(const QString &tag);
  void export_frames_check_changed(bool enabled);
  void export_frames_value_changed(int value);
  void export_record_pressed();
  void export_stop_pressed();
  void export_stop_fan_pressed();

private:
  void setup_ui();

  // Image type
  QComboBox *image_type_combo_;

  // File selection
  QLineEdit *file_line_edit_;
  QPushButton *file_browse_button_;

  // Tag selection
  QComboBox *tag_combo_;

  // Frames
  QCheckBox *frames_check_;
  QSpinBox *frames_spin_;

  // Action buttons
  QPushButton *record_button_;
  QPushButton *stop_button_;
  QPushButton *stop_fan_button_;
};

} // namespace holovibes::ui