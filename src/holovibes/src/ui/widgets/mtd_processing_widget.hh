// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pipeline/settings.hh"
#include <QWidget>
class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QSpinBox;
namespace holovibes::ui {
class MTDProcessingWidget : public QWidget {
  Q_OBJECT
public:
  explicit MTDProcessingWidget(QWidget *parent = nullptr);
  pipeline::MTDReconstructionSettings settings() const;
  void set_settings(const pipeline::MTDReconstructionSettings &settings);
  void set_dataset_summary(const QString &summary);
signals:
  void settings_changed();

private:
  QSpinBox       *dimension_, *iterations_;
  QCheckBox      *override_medium_;
  QDoubleSpinBox *medium_index_;
  QLabel         *summary_;
};
} // namespace holovibes::ui
