#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QSpinBox>

namespace holovibes::ui {

class ViewWidget : public QGroupBox {
  Q_OBJECT

public:
  explicit ViewWidget(QWidget *parent = nullptr);

  // clang-format off
  QString current_image_type() const { return image_type_combo_->currentText(); }
  bool current_cuts_3d() const { return cuts_3d_check_->isChecked(); }
  bool current_fft_shift() const { return fft_shift_check_->isChecked(); }
  bool current_lens_view() const { return lens_view_check_->isChecked(); }
  bool current_raw_view() const { return raw_view_check_->isChecked(); }
  int current_z_value() const { return z_spin_->value(); }
  int current_width_value() const { return width_spin_->value(); }
  QString current_view_kind() const { return view_kind_combo_->currentText(); }
  int current_accumulation() const { return accumulation_spin_->value(); }
  bool current_auto() const { return auto_check_->isChecked(); }
  bool current_invert() const { return invert_check_->isChecked(); }
  int current_range_start() const { return range_start_spin_->value(); }
  int current_range_end() const { return range_end_spin_->value(); }
  bool current_renormalize() const { return renormalize_check_->isChecked(); }
  // clang-format on

signals:
  // Signals for main view settings
  void image_type_changed(const QString &image_type);
  void cuts_3d_toggled(bool enabled);
  void fft_shift_toggled(bool enabled);
  void lens_view_toggled(bool enabled);
  void raw_view_toggled(bool enabled);
  void z_changed(int value);
  void width_changed(int value);
  void view_kind_changed(const QString &view_kind);
  void accumulation_changed(int value);

  // Signals for brightness/contrast group
  void auto_changed(bool enabled);
  void invert_changed(bool enabled);
  void range_start_changed(int value);
  void range_end_changed(int value);
  void renormalize_changed(bool enabled);

private:
  void setup_ui();

  // Main view widgets
  QComboBox *image_type_combo_;
  QCheckBox *cuts_3d_check_;
  QCheckBox *fft_shift_check_;
  QCheckBox *lens_view_check_;
  QCheckBox *raw_view_check_;
  QSpinBox *z_spin_;
  QSpinBox *width_spin_;
  QComboBox *view_kind_combo_;
  QSpinBox *accumulation_spin_;

  // Brightness/contrast sub-widgets
  QGroupBox *brightness_contrast_box_;
  QCheckBox *auto_check_;
  QCheckBox *invert_check_;
  QSpinBox *range_start_spin_;
  QSpinBox *range_end_spin_;
  QCheckBox *renormalize_check_;
};

} // namespace holovibes::ui