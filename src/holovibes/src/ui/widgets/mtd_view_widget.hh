// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pipeline/settings.hh"
#include <QWidget>
#include <array>
class QComboBox;
class QCheckBox;
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
  void                      set_slice_positions(int x, int y, int z);
  void                      set_extent(int dimension, double voxel_pitch_m = 0);
  void                      set_focus_mode();
  void set_focus_extent(int width, int height, int depth, double dx, double dy, double z_min,
                        double z_max, bool temporal = false);
signals:
  void settings_changed();

private:
  std::array<QSlider *, 3>  sliders_{};
  std::array<QSpinBox *, 3> positions_{};
  QComboBox                *component_;
  QDoubleSpinBox           *minimum_, *maximum_;
  QLabel                   *geometry_;
  QLabel                   *z_position_ = nullptr;
  std::array<QSpinBox *, 3> widths_{};
  std::array<QLabel *, 3>   axis_labels_{};
  QCheckBox                *auto_contrast_, *fit_panel_;
  bool                      temporal_ = false;
  double                    z_min_ = 0, z_step_ = 0;
};
} // namespace holovibes::ui
