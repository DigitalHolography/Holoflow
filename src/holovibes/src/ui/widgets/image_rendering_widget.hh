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
#include <QLabel>
#include <QSlider>
#include <QSpinBox>
#include <limits>

#include "ui/widgets/auto_focus_widget.hh"

namespace holovibes::ui {

class ImageRenderingWidget : public QGroupBox {
  Q_OBJECT

public:
  explicit ImageRenderingWidget(QWidget *parent = nullptr);

  // Getters
  QString get_image_mode() const;
  int     get_batch_size() const;
  int     get_time_stride() const;
  bool    is_filter_2d_enabled() const;
  int     get_filter_inner() const;
  int     get_filter_outer() const;
  QString get_space_transform() const;
  QString get_time_transform() const;
  int     get_time_window() const;
  int     get_lambda() const;
  int     get_focus() const;
  bool    is_asp_padding_enabled() const;
  int     get_asp_padded_width() const;
  int     get_asp_padded_height() const;
  QString get_convolution() const;
  bool    is_convolution_divide() const;
  bool    is_spatial_contrast_enabled() const;
  int     get_spatial_contrast_window_size() const;
  int     get_spatial_contrast_dimensions() const;
  QString get_spatial_contrast_method() const;
  double  get_spatial_contrast_display_max() const;
  bool    is_spatial_contrast_display_inverted() const;

  // Setters
  void set_batch_size(int value);
  void set_time_stride(int value);
  void set_filter_2d_enabled(bool enabled);
  void set_filter_inner(int value);
  void set_filter_outer(int value);
  void set_space_transform(const QString &method);
  void set_time_transform(const QString &method);
  void set_time_window(int value);
  void set_lambda(int value);
  void set_focus(int value);
  void set_asp_padding_enabled(bool enabled);
  void set_asp_padded_width(int value);
  void set_asp_padded_height(int value);
  void set_convolution(const QString &kernel);
  void set_convolution_divide(bool enabled);
  void set_spatial_contrast_mode(bool enabled, int dimensions);
  void set_spatial_contrast_window_size(int value);
  void set_time_window_max_for_temporal_lscsi(int value);
  void set_spatial_contrast_method(const QString &method);
  void set_spatial_contrast_display_max(double value);
  void set_spatial_contrast_display_inverted(bool inverted);
  void set_spatial_contrast_appearance_locked(bool locked);

  // Validation
  void clear_validation_styles();
  void mark_batch_size_invalid();
  void mark_time_stride_invalid();
  void mark_filter_2d_invalid();
  void mark_filter_inner_invalid();
  void mark_filter_outer_invalid();
  void mark_time_window_invalid();
  void mark_space_transform_invalid();
  void mark_asp_padding_invalid();
  void mark_time_transform_invalid();
  void mark_convolution_invalid();
  void mark_spatial_contrast_window_invalid();
  void mark_spatial_contrast_dimensions_invalid();
  void mark_spatial_contrast_method_invalid();

  // Access to widgets for connection setup
  QComboBox       *image_combo();
  QSpinBox        *batch_size_spin();
  QSpinBox        *time_stride_spin();
  QCheckBox       *filter_2d_check();
  QSpinBox        *filter_2d_inner_spin();
  QSpinBox        *filter_2d_outer_spin();
  QComboBox       *space_transform_combo();
  QComboBox       *time_transform_combo();
  QSpinBox        *time_window_spin();
  QSpinBox        *lambda_spin();
  QSpinBox        *focus_spin();
  QSlider         *focus_slider();
  QCheckBox       *asp_padding_check();
  QSpinBox        *asp_padded_width_spin();
  QSpinBox        *asp_padded_height_spin();
  QComboBox       *convolution_combo();
  QCheckBox       *convolution_divide_check();
  QSpinBox        *spatial_contrast_window_size();
  QComboBox       *spatial_contrast_method();
  QDoubleSpinBox  *spatial_contrast_display_max();
  AutoFocusWidget *autofocus_widget();

signals:
  void settings_changed();
  void spatial_contrast_enabled_changed(bool enabled);
  void spatial_contrast_display_range_changed(float minimum, float maximum);

private:
  void        setup_ui();
  void        connect_signals();
  QStringList load_available_kernels();
  void        set_angular_spectrum_enabled(bool enabled);
  void        update_asp_padding_visibility();
  void        update_spatial_contrast_controls();
  void        update_spatial_contrast_display_range();
  void        update_time_window_limits();

  QComboBox       *image_combo_;
  QSpinBox        *batch_size_spin_;
  QSpinBox        *time_stride_spin_;
  QCheckBox       *filter_2d_check_;
  QSpinBox        *filter_2d_inner_spin_;
  QSpinBox        *filter_2d_outer_spin_;
  QComboBox       *space_transform_combo_;
  QComboBox       *time_transform_combo_;
  QSpinBox        *time_window_spin_;
  QSpinBox        *lambda_spin_;
  QSpinBox        *focus_spin_;
  QSlider         *focus_slider_;
  QCheckBox       *asp_padding_check_;
  QSpinBox        *asp_padded_width_spin_;
  QSpinBox        *asp_padded_height_spin_;
  QWidget         *asp_padding_controls_;
  QComboBox       *convolution_combo_;
  QCheckBox       *convolution_divide_check_;
  QLabel          *spatial_contrast_window_label_;
  QSpinBox        *spatial_contrast_window_size_;
  QLabel          *spatial_contrast_method_label_;
  QComboBox       *spatial_contrast_method_;
  QLabel          *spatial_contrast_display_max_label_;
  QDoubleSpinBox  *spatial_contrast_display_max_;
  QCheckBox       *spatial_contrast_invert_display_;
  bool             spatial_contrast_appearance_locked_ = false;
  AutoFocusWidget *autofocus_widget_;
  int              max_temporal_lscsi_frames_ = std::numeric_limits<int>::max();
};

} // namespace holovibes::ui
