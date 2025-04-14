#pragma once

#include <QComboBox>
#include <QGroupBox>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>

namespace holovibes::ui {

class ImportWidget : public QGroupBox {
  Q_OBJECT

public:
  explicit ImportWidget(QWidget *parent = nullptr);

  // clang-format off
  QString current_file() const { return file_line_edit_->text(); }
  int current_fps() const { return fps_spin_->value(); }
  int current_start_index() const { return start_index_spin_->value(); }
  int current_end_index() const { return end_index_spin_->value(); }
  QString current_load_method() const { return load_method_combo_->currentText(); }
  // clang-format on

signals:
  void file_selected(const QString &file_path);
  void fps_changed(int value);
  void start_index_changed(int value);
  void end_index_changed(int value);
  void load_method_changed(const QString &method);
  void start_import();
  void stop_import();

private:
  void setup_ui();

  // File selection
  QLineEdit *file_line_edit_;
  QPushButton *file_browse_button_;

  // Numeric fields
  QSpinBox *fps_spin_;
  QSpinBox *start_index_spin_;
  QSpinBox *end_index_spin_;

  // Load method combo
  QComboBox *load_method_combo_;

  // Control buttons
  QPushButton *start_button_;
  QPushButton *stop_button_;
};

} // namespace holovibes::ui
