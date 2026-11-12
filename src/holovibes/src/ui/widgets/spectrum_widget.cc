// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "spectrum_widget.hh"

#include <QCheckBox>
#include <QHBoxLayout>
#include <QPainter>
#include <QPainterPath>
#include <QVBoxLayout>

#include <utility>

namespace holovibes::ui {

SpectrumWidget::SpectrumWidget(QWidget *parent) : QWidget(parent) {
  setMinimumSize(240, 180);

  controls_             = new QWidget(this);
  auto *controls_layout = new QHBoxLayout(controls_);
  controls_layout->setContentsMargins(0, 0, 0, 0);
  log10_scale_ = new QCheckBox(tr("Log10 power"), controls_);
  log10_scale_->setToolTip(tr("Plot log10(power); zero power values are omitted."));
  exclude_dc_ = new QCheckBox(tr("Exclude DC"), controls_);
  exclude_dc_->setToolTip(
      tr("Hide the zero-frequency bin and fit Y limits to the remaining bins."));
  controls_layout->addWidget(log10_scale_);
  controls_layout->addWidget(exclude_dc_);
  controls_layout->addStretch();

  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(8, 4, 8, 0);
  layout->addWidget(controls_);
  layout->addStretch();

  connect(log10_scale_, &QCheckBox::toggled, this, [this] { update(); });
  connect(exclude_dc_, &QCheckBox::toggled, this, [this] { update(); });
}

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

  const double header_top = controls_->geometry().bottom() + 4.0;
  const QRectF plot       = QRectF(rect()).adjusted(85.0, header_top + 24.0, -30.0, -50.0);
  if (plot.width() <= 0.0 || plot.height() <= 0.0) {
    return;
  }
  // Symmetric FFT limits keep DC at the center, including even-length transforms whose
  // positive-frequency endpoint is one bin short of Nyquist.
  const double x_min       = full_fft_ ? -sampling_frequency_hz_ / 2.0 : 0.0;
  const double x_max       = sampling_frequency_hz_ / 2.0;
  const bool   log10_scale = log10_scale_->isChecked();
  const bool   exclude_dc  = exclude_dc_->isChecked();
  const auto   y_range     = spectrum_power_range(samples_, log10_scale, exclude_dc);

  painter.drawText(QRectF(plot.left(), header_top, plot.width(), 20.0), Qt::AlignCenter,
                   tr("Mean power spectrum"));
  painter.drawText(QRectF(plot.left(), plot.bottom() + 28.0, plot.width(), 20.0), Qt::AlignCenter,
                   tr("Frequency (Hz)"));
  painter.save();
  painter.translate(16.0, plot.center().y());
  painter.rotate(-90.0);
  painter.drawText(QRectF(-plot.height() / 2.0, -10.0, plot.height(), 20.0), Qt::AlignCenter,
                   log10_scale ? tr("log10(Power / a.u.)") : tr("Power (a.u.)"));
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
    painter.drawText(
        QRectF(23.0, y - 10.0, 56.0, 20.0), Qt::AlignRight | Qt::AlignVCenter,
        QString::number(y_range.minimum + fraction * (y_range.maximum - y_range.minimum), 'g', 3));
  }
  painter.drawRect(plot);

  painter.save();
  painter.setClipRect(plot);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setPen(QPen(palette().color(QPalette::Highlight), 1.5));
  QPainterPath curve;
  bool         start_segment = true;
  for (const auto &sample : samples_) {
    const auto value = spectrum_plot_value(sample, log10_scale, exclude_dc);
    if (!value) {
      start_segment = true;
      continue;
    }
    const QPointF point(plot.left() +
                            (sample.frequency_hz - x_min) / (x_max - x_min) * plot.width(),
                        plot.bottom() - (*value - y_range.minimum) /
                                            (y_range.maximum - y_range.minimum) * plot.height());
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
