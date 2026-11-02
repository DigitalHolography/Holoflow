// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <QRectF>
#include <QWidget>
#include <cstddef>
#include <cstdint>
#include <vector>

class QMouseEvent;

namespace holovibes::ui {

struct MesiExposureSample {
  std::vector<int>    exposure_frames;
  std::vector<double> exposure_seconds;
  std::vector<double> measured_k2;
  std::vector<double> model_exposure_seconds;
  std::vector<double> model_k2;
  double              beta              = 0.0;
  double              rho               = 0.0;
  double              tau_seconds       = 0.0;
  double              rmse              = 0.0;
  double              exposure_fraction = 1.0;
  size_t              valid_pixels      = 0;
  size_t              roi_pixels        = 0;
  uint64_t            window_index      = 0;
  bool                fit_valid         = false;
  bool                tau_at_bound      = false;
  bool                fixed_beta_rho    = false;
  bool                single_pixel      = false;
  double              pixel_tau_seconds = 0.0;
  double              pixel_beta        = 0.0;
  double              pixel_rho         = 0.0;
};

class MesiExposurePlotWidget : public QWidget {
  Q_OBJECT

public:
  explicit MesiExposurePlotWidget(QWidget *parent = nullptr);
  void  reset();
  void  set_sample(MesiExposureSample sample);
  QSize sizeHint() const override;

signals:
  void samplesDisplayed();

protected:
  void paintEvent(QPaintEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void leaveEvent(QEvent *event) override;

private:
  MesiExposureSample sample_;
  bool               has_sample_ = false;
  QRectF             graph_rect_;
  double             x_min_ = 0.0;
  double             x_max_ = 1.0;
  double             y_min_ = 0.0;
  double             y_max_ = 1.0;
};

} // namespace holovibes::ui
