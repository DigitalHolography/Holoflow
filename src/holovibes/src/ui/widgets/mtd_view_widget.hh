// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pipeline/settings.hh"
#include <QWidget>
#include <array>
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QSlider;
class QSpinBox;
namespace holovibes::ui {
class MTDViewWidget : public QWidget {
  Q_OBJECT
public:
  explicit MTDViewWidget(QWidget *parent = nullptr);
  pipeline::MTDViewSettings settings() const;
  void                      set_settings(const pipeline::MTDViewSettings &settings);
  void                      set_extent(int dimension, double voxel_pitch_m = 0);
signals:
  void settings_changed();

private:
  std::array<QSlider *, 3>  sliders_{};
  std::array<QSpinBox *, 3> positions_{};
  QComboBox                *component_;
  QDoubleSpinBox           *minimum_, *maximum_;
  QLabel                   *geometry_;
};
} // namespace holovibes::ui
