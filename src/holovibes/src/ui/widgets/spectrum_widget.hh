// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <QWidget>

#include "spectrum.hh"

class QCheckBox;

namespace holovibes::ui {

class SpectrumWidget : public QWidget {
  Q_OBJECT

public:
  explicit SpectrumWidget(QWidget *parent = nullptr);

  void  present_spectrum(std::vector<SpectrumSample> samples, double sampling_frequency_hz,
                         bool full_fft);
  void  show_waiting_placeholder();
  QSize sizeHint() const override;

signals:
  void spectrumDisplayed();

protected:
  void paintEvent(QPaintEvent *event) override;

private:
  std::vector<SpectrumSample> samples_;
  double                      sampling_frequency_hz_ = 1.0;
  bool                        full_fft_              = false;
  QWidget                    *controls_              = nullptr;
  QCheckBox                  *log10_scale_           = nullptr;
  QCheckBox                  *exclude_dc_            = nullptr;
};

} // namespace holovibes::ui
