// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "ui/widgets/mtd_processing_widget.hh"
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QSignalBlocker>
#include <QSpinBox>
namespace holovibes::ui {
MTDProcessingWidget::MTDProcessingWidget(QWidget *parent) : QWidget(parent) {
  auto *layout = new QFormLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  dimension_ = new QSpinBox(this);
  dimension_->setRange(2, 1024);
  dimension_->setSingleStep(2);
  iterations_ = new QSpinBox(this);
  iterations_->setRange(1, 1000);
  override_medium_ = new QCheckBox(tr("Override medium index"), this);
  medium_index_    = new QDoubleSpinBox(this);
  medium_index_->setDecimals(6);
  medium_index_->setRange(0.000001, 10.0);
  medium_index_->setValue(1.49);
  medium_index_->setEnabled(false);
  summary_ = new QLabel(tr("Select an ACQUIS folder to load the optical settings."), this);
  summary_->setWordWrap(true);
  layout->addRow(tr("Volume dimension"), dimension_);
  layout->addRow(tr("GPS iterations"), iterations_);
  layout->addRow(override_medium_, medium_index_);
  layout->addRow(summary_);
  connect(dimension_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this] { emit settings_changed(); });
  connect(iterations_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this] { emit settings_changed(); });
  connect(medium_index_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          [this] { emit settings_changed(); });
  connect(override_medium_, &QCheckBox::toggled, this, [this](bool enabled) {
    medium_index_->setEnabled(enabled);
    emit settings_changed();
  });
  set_settings({});
}
pipeline::MTDReconstructionSettings MTDProcessingWidget::settings() const {
  pipeline::MTDReconstructionSettings result{.dimension  = dimension_->value(),
                                             .iterations = iterations_->value()};
  if (override_medium_->isChecked())
    result.medium_index = medium_index_->value();
  return result;
}
void MTDProcessingWidget::set_settings(const pipeline::MTDReconstructionSettings &settings) {
  const QSignalBlocker blocker(this);
  dimension_->setValue(settings.dimension);
  iterations_->setValue(settings.iterations);
  override_medium_->setChecked(settings.medium_index.has_value());
  medium_index_->setEnabled(settings.medium_index.has_value());
  if (settings.medium_index)
    medium_index_->setValue(*settings.medium_index);
}
void MTDProcessingWidget::set_dataset_summary(const QString &summary) {
  summary_->setText(summary);
}
} // namespace holovibes::ui
