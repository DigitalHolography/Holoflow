// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "ui/widgets/mtd_view_widget.hh"
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QLabel>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <algorithm>
namespace holovibes::ui {
MTDViewWidget::MTDViewWidget(QWidget *parent) : QWidget(parent) {
  auto *layout = new QGridLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  const QStringList labels{tr("X — YZ slice"), tr("Y — XZ slice"), tr("Z — XY slice")};
  for (int axis = 0; axis < 3; ++axis) {
    auto *slider = sliders_[axis] = new QSlider(Qt::Horizontal, this);
    auto *position = positions_[axis] = new QSpinBox(this);
    slider->setRange(0, 511);
    position->setRange(0, 511);
    axis_labels_[axis] = new QLabel(labels[axis], this);
    layout->addWidget(axis_labels_[axis], axis, 0);
    layout->addWidget(slider, axis, 1);
    layout->addWidget(position, axis, 2);
    connect(slider, &QSlider::valueChanged, position, &QSpinBox::setValue);
    connect(position, qOverload<int>(&QSpinBox::valueChanged), this, [this, slider](int value) {
      const QSignalBlocker blocker(slider);
      slider->setValue(value);
      emit settings_changed();
    });
  }
  component_ = new QComboBox(this);
  component_->addItem(tr("Refractive index difference (Δn)"), "real");
  component_->addItem(tr("Absorption (κ)"), "imaginary");
  component_->addItem(tr("Magnitude"), "magnitude");
  layout->addWidget(new QLabel(tr("Component"), this), 3, 0);
  layout->addWidget(component_, 3, 1, 1, 2);
  minimum_ = new QDoubleSpinBox(this);
  maximum_ = new QDoubleSpinBox(this);
  for (auto *spin : {minimum_, maximum_}) {
    spin->setDecimals(6);
    spin->setRange(-1e6, 1e6);
    spin->setSingleStep(0.001);
    connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this] { emit settings_changed(); });
  }
  layout->addWidget(new QLabel(tr("Display min / max"), this), 4, 0);
  layout->addWidget(minimum_, 4, 1);
  layout->addWidget(maximum_, 4, 2);
  z_position_ = new QLabel(this);
  layout->addWidget(z_position_, 6, 0, 1, 3);
  connect(positions_[2], qOverload<int>(&QSpinBox::valueChanged), this, [this](int index) {
    z_position_->setText(
        temporal_ ? tr("Temporal bin %1 (not calibrated depth)").arg(index)
                  : tr("Z propagation = %1 mm").arg((z_min_ + index * z_step_) * 1e3, 0, 'g', 8));
  });
  z_position_->hide();
  geometry_ = new QLabel(this);
  geometry_->setWordWrap(true);
  layout->addWidget(geometry_, 5, 0, 1, 3);
  connect(component_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          [this] { emit settings_changed(); });
  for (int axis = 0; axis < 3; ++axis) {
    widths_[axis] = new QSpinBox(this);
    widths_[axis]->setRange(0, 2047);
    widths_[axis]->setKeyboardTracking(false);
    auto *label = new QLabel(
        axis == 0 ? tr("X width") : (axis == 1 ? tr("Y width") : tr("XY band width")), this);
    layout->addWidget(label, 7 + axis, 0);
    layout->addWidget(widths_[axis], 7 + axis, 1, 1, 2);
    label->hide();
    widths_[axis]->hide();
    widths_[axis]->setToolTip(
        tr("Inclusive extent, matching Holovibes: width 0 averages one plane; width 20 averages up "
           "to 21 planes. Clipped at the volume edge."));
    connect(widths_[axis], qOverload<int>(&QSpinBox::valueChanged), this,
            [this] { emit settings_changed(); });
  }
  auto_contrast_ = new QCheckBox(tr("Auto contrast per view (0.5-99.5%)"), this);
  fit_panel_     = new QCheckBox(tr("Fit images to panels"), this);
  layout->addWidget(auto_contrast_, 10, 0, 1, 3);
  layout->addWidget(fit_panel_, 11, 0, 1, 3);
  auto_contrast_->hide();
  fit_panel_->hide();
  connect(auto_contrast_, &QCheckBox::toggled, this, [this](bool enabled) {
    minimum_->setEnabled(!enabled);
    maximum_->setEnabled(!enabled);
    emit settings_changed();
  });
  connect(fit_panel_, &QCheckBox::toggled, this, [this] { emit settings_changed(); });
  set_settings({});
  set_extent(512);
}
pipeline::MTDViewSettings MTDViewWidget::settings() const {
  return {.x             = positions_[0]->value(),
          .y             = positions_[1]->value(),
          .z             = positions_[2]->value(),
          .component     = component_->currentData().toString().toStdString(),
          .minimum       = minimum_->value(),
          .maximum       = maximum_->value(),
          .band_width    = widths_[2]->value(),
          .x_width       = widths_[0]->value(),
          .y_width       = widths_[1]->value(),
          .auto_contrast = auto_contrast_->isChecked(),
          .fit_panel     = fit_panel_->isChecked()};
}
void MTDViewWidget::set_settings(const pipeline::MTDViewSettings &settings) {
  const QSignalBlocker blocker(this);
  const std::array     values{settings.x, settings.y, settings.z};
  for (int axis = 0; axis < 3; ++axis)
    positions_[axis]->setValue(values[axis]);
  component_->setCurrentIndex(
      std::max(0, component_->findData(QString::fromStdString(settings.component))));
  minimum_->setValue(settings.minimum);
  maximum_->setValue(settings.maximum);
  widths_[0]->setValue(settings.x_width);
  widths_[1]->setValue(settings.y_width);
  widths_[2]->setValue(settings.band_width);
  auto_contrast_->setChecked(settings.auto_contrast);
  fit_panel_->setChecked(settings.fit_panel);
}
void MTDViewWidget::set_extent(int dimension, double voxel_pitch_m) {
  const QSignalBlocker blocker(this);
  const int            maximum = std::max(0, dimension - 1);
  for (int axis = 0; axis < 3; ++axis) {
    sliders_[axis]->setRange(0, maximum);
    positions_[axis]->setRange(0, maximum);
    sliders_[axis]->setValue(positions_[axis]->value());
  }
  geometry_->setText(
      voxel_pitch_m > 0
          ? tr("%1³ voxels · %2 μm per voxel. Positions are voxel indices.")
                .arg(dimension)
                .arg(voxel_pitch_m * 1e6, 0, 'g', 6)
          : tr("Positions are voxel indices. Shared contrast stays fixed while navigating."));
}
void MTDViewWidget::set_focus_mode() {
  const QSignalBlocker blocker(this);
  component_->clear();
  component_->addItem(tr("Normalized holographic intensity"), "real");
  component_->setEnabled(false);
  z_position_->show();
  auto_contrast_->show();
  fit_panel_->show();
  for (int axis = 0; axis < 3; ++axis) {
    widths_[axis]->show();
    auto *grid = qobject_cast<QGridLayout *>(layout());
    grid->itemAtPosition(7 + axis, 0)->widget()->show();
  }
}
void MTDViewWidget::set_focus_extent(int width, int height, int depth, double dx, double dy,
                                     double z_min, double z_max, bool temporal) {
  const QSignalBlocker blocker(this);
  temporal_  = temporal;
  auto *grid = qobject_cast<QGridLayout *>(layout());
  for (int axis = 0; axis < 3; ++axis) {
    grid->itemAtPosition(7 + axis, 0)->widget()->setVisible(temporal);
    widths_[axis]->setVisible(temporal);
  }
  axis_labels_[2]->setText(temporal ? tr("P - XY band start") : tr("Z - XY slice"));
  axis_labels_[0]->setText(temporal ? tr("X - YP cut") : tr("X - YZ slice"));
  axis_labels_[1]->setText(temporal ? tr("Y - XP cut") : tr("Y - XZ slice"));
  component_->setItemText(0,
                          temporal ? tr("STFT magnitude") : tr("Normalized holographic intensity"));
  const std::array sizes{width, height, depth};
  for (int axis = 0; axis < 3; ++axis) {
    widths_[axis]->setMaximum(std::max(0, sizes[axis] - 1));
    sliders_[axis]->setRange(0, std::max(0, sizes[axis] - 1));
    positions_[axis]->setRange(0, std::max(0, sizes[axis] - 1));
    sliders_[axis]->setValue(positions_[axis]->value());
  }
  if (temporal) {
    geometry_->setText(tr("%1 x %2 pixels, %3 temporal bins. XY averages the selected P band; "
                          "XP/YP average the selected Y/X widths. P is a temporal-frequency bin, "
                          "not anatomical Z. Width 0 means one plane.")
                           .arg(width)
                           .arg(height)
                           .arg(depth));
    z_position_->setText(tr("Temporal bin %1 (not calibrated depth)").arg(positions_[2]->value()));
    return;
  }
  z_min_  = z_min;
  z_step_ = depth > 1 ? (z_max - z_min) / (depth - 1) : 0;
  geometry_->setText(
      tr("%1 x %2 x %3 (X,Y,Z). Sensor pitches %4 / %5 um; Z step %6 mm. Positions are indices; "
         "display aspect is controlled by Fit images to panels. Propagation distance is not "
         "anatomical depth.")
          .arg(width)
          .arg(height)
          .arg(depth)
          .arg(dx * 1e6, 0, 'g', 6)
          .arg(dy * 1e6, 0, 'g', 6)
          .arg(z_step_ * 1e3, 0, 'g', 6));
  z_position_->setText(tr("Z propagation = %1 mm")
                           .arg((z_min_ + positions_[2]->value() * z_step_) * 1e3, 0, 'g', 8));
}
void MTDViewWidget::set_slice_positions(int x, int y, int z) {
  const auto before = settings();
  {
    const QSignalBlocker blocker(this);
    const std::array     values{x, y, z};
    for (size_t i = 0; i < values.size(); ++i)
      positions_[i]->setValue(values[i]);
  }
  const auto after = settings();
  if (before.x != after.x || before.y != after.y || before.z != after.z)
    emit settings_changed();
}
} // namespace holovibes::ui
