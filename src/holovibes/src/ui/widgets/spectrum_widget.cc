// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "spectrum_widget.hh"

#include <QPainter>
#include <QPainterPath>

#include <algorithm>
#include <cmath>
#include <utility>

namespace holovibes::ui {

SpectrumWidget::SpectrumWidget(QWidget *parent) : QWidget(parent) { setMinimumSize(240, 180); }

void SpectrumWidget::present_spectrum(std::vector<SpectrumSample> samples,
                                      double sampling_frequency_hz, bool full_fft) {
  samples_               = std::move(samples);
  sampling_frequency_hz_ = sampling_frequency_hz;
  full_fft_              = full_fft;
  update();
  emit spectrumDisplayed();
}

void SpectrumWidget::show_waiting_placeholder() {
  samples_.clear();
  update();
}

QSize SpectrumWidget::sizeHint() const { return {640, 360}; }

void SpectrumWidget::paintEvent(QPaintEvent *) {
  QPainter painter(this);
  painter.fillRect(rect(), palette().brush(QPalette::Base));
  const QColor text_color = palette().color(QPalette::Text);
  painter.setPen(text_color);
  if (samples_.empty()) {
    painter.drawText(rect(), Qt::AlignCenter, tr("Waiting for spectrum data..."));
    return;
  }

  const QRectF plot = QRectF(rect()).adjusted(85.0, 28.0, -30.0, -50.0);
  if (plot.width() <= 0.0 || plot.height() <= 0.0) {
    return;
  }
  // Symmetric FFT limits keep DC at the center, including even-length transforms whose
  // positive-frequency endpoint is one bin short of Nyquist.
  const double x_min = full_fft_ ? -sampling_frequency_hz_ / 2.0 : 0.0;
  const double x_max = sampling_frequency_hz_ / 2.0;
  double       y_max = 0.0;
  for (const auto &sample : samples_) {
    if (std::isfinite(sample.power)) {
      y_max = std::max(y_max, sample.power);
    }
  }
  y_max = y_max > 0.0 ? y_max * 1.05 : 1.0;

  painter.drawText(QRectF(plot.left(), 4.0, plot.width(), 20.0), Qt::AlignCenter,
                   tr("Mean power spectrum"));
  painter.drawText(QRectF(plot.left(), plot.bottom() + 28.0, plot.width(), 20.0), Qt::AlignCenter,
                   tr("Frequency (Hz)"));
  painter.save();
  painter.translate(16.0, plot.center().y());
  painter.rotate(-90.0);
  painter.drawText(QRectF(-plot.height() / 2.0, -10.0, plot.height(), 20.0), Qt::AlignCenter,
                   tr("Power (a.u.)"));
  painter.restore();

  QColor grid_color = text_color;
  grid_color.setAlpha(40);
  for (int tick = 0; tick <= 4; ++tick) {
    const double fraction = tick / 4.0;
    const double x        = plot.left() + fraction * plot.width();
    const double y        = plot.bottom() - fraction * plot.height();
    painter.setPen(grid_color);
    painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
    painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
    painter.setPen(text_color);
    painter.drawText(QRectF(x - 40.0, plot.bottom() + 4.0, 80.0, 20.0), Qt::AlignCenter,
                     QString::number(x_min + fraction * (x_max - x_min), 'g', 4));
    painter.drawText(QRectF(23.0, y - 10.0, 56.0, 20.0), Qt::AlignRight | Qt::AlignVCenter,
                     QString::number(fraction * y_max, 'g', 3));
  }
  painter.drawRect(plot);

  painter.save();
  painter.setClipRect(plot);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setPen(QPen(palette().color(QPalette::Highlight), 1.5));
  QPainterPath curve;
  bool         start_segment = true;
  for (const auto &sample : samples_) {
    if (!std::isfinite(sample.power)) {
      start_segment = true;
      continue;
    }
    const QPointF point(plot.left() +
                            (sample.frequency_hz - x_min) / (x_max - x_min) * plot.width(),
                        plot.bottom() - sample.power / y_max * plot.height());
    if (start_segment) {
      curve.moveTo(point);
      painter.drawPoint(point);
      start_segment = false;
    } else {
      curve.lineTo(point);
    }
  }
  painter.drawPath(curve);
  painter.restore();
}

} // namespace holovibes::ui
