// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "ui/widgets/mtd_view_widget.hh"
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
    layout->addWidget(new QLabel(labels[axis], this), axis, 0);
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
  geometry_ = new QLabel(this);
  geometry_->setWordWrap(true);
  layout->addWidget(geometry_, 5, 0, 1, 3);
  connect(component_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          [this] { emit settings_changed(); });
  set_settings({});
  set_extent(512);
}
pipeline::MTDViewSettings MTDViewWidget::settings() const {
  return {.x         = positions_[0]->value(),
          .y         = positions_[1]->value(),
          .z         = positions_[2]->value(),
          .component = component_->currentData().toString().toStdString(),
          .minimum   = minimum_->value(),
          .maximum   = maximum_->value()};
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
} // namespace holovibes::ui
