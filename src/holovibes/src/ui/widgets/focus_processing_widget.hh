// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "holotask/pipelines/focus_stack.hh"
#include <QWidget>
#include <array>
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QSpinBox;
class QLabel;

namespace holovibes::ui {
class FocusProcessingWidget : public QWidget {
  Q_OBJECT
public:
  explicit FocusProcessingWidget(QWidget *parent = nullptr);
  holotask::pipelines::HoloFocusSettings settings() const;
  void set_settings(const holotask::pipelines::HoloFocusSettings &settings);
  void set_playback_position(int first, int last, int total);
  void load_calibration(const std::filesystem::path &path, bool apply_estimates);
signals:
  void settings_changed();

private:
  void                            refresh_method();
  QComboBox                      *method_;
  QCheckBox                      *playback_;
  QSpinBox                       *playback_fps_;
  QLabel                         *playback_position_;
  QSpinBox                       *window_, *filter_inner_, *filter_outer_;
  QDoubleSpinBox                 *temporal_focus_, *temporal_pitch_;
  QCheckBox                      *temporal_filter_;
  QLabel                         *frame_label_;
  QSpinBox                       *frame_, *planes_, *carrier_x_, *carrier_y_, *radius_;
  std::array<QDoubleSpinBox *, 5> optics_{};
  QCheckBox                      *padding_, *isolate_;
  QLabel                         *calibration_error_;
};
} // namespace holovibes::ui
