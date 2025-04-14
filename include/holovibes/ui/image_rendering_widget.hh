#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QSlider>
#include <QSpinBox>

namespace holovibes::ui {

class ImageRenderingWidget : public QGroupBox {
  Q_OBJECT
public:
  explicit ImageRenderingWidget(QWidget *parent = nullptr);

  // clang-format off
  QString current_image() const { return image_combo_->currentText(); }
  int current_batch_size() const { return batch_size_spin_->value(); }
  int current_time_stride() const { return time_stride_spin_->value(); }
  bool current_filter_2d() const { return filter_2d_check_->isChecked(); }
  QString current_space_transform() const { return space_transform_combo_->currentText(); }
  QString current_time_transform() const { return time_transform_combo_->currentText(); }
  int current_time_window() const { return time_window_spin_->value(); }
  int current_lambda() const { return lambda_spin_->value(); }
  int current_boundary() const { return boundary_spin_->value(); }
  int current_focus() const { return focus_spin_->value(); }
  int current_focus_slider() const { return focus_slider_->value(); }
  QString current_convolution() const { return convolution_combo_->currentText(); }
  bool current_convolution_divide() const { return convolution_check_->isChecked(); }
  // clang-format on

signals:
  void image_changed(const QString &image_type);
  void batch_size_changed(int value);
  void time_stride_changed(int value);
  void filter_2d_toggled(bool enabled);
  void space_transform_changed(const QString &transform);
  void time_transform_changed(const QString &transform);
  void time_window_changed(int value);
  void lambda_changed(int value);
  void boundary_changed(int value);
  void focus_changed(int value);
  void focus_slider_changed(int value);
  void convolution_changed(const QString &convolution_type, bool divide);

private:
  void setup_ui();
  void validate_settings();

  QComboBox *image_combo_;
  QSpinBox *batch_size_spin_;
  QSpinBox *time_stride_spin_;
  QCheckBox *filter_2d_check_;
  QComboBox *space_transform_combo_;
  QComboBox *time_transform_combo_;
  QSpinBox *time_window_spin_;
  QSpinBox *lambda_spin_;
  QSpinBox *boundary_spin_;
  QSpinBox *focus_spin_;
  QSlider *focus_slider_;
  QComboBox *convolution_combo_;
  QCheckBox *convolution_check_;
};

} // namespace holovibes::ui