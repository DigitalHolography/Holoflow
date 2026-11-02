// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "mesi_exposure_plot_widget.hh"

#include <QEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QToolTip>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace holovibes::ui {
namespace {

QString exposure_label(double seconds) {
  if (seconds < 1.0e-3)
    return QStringLiteral("%1 µs").arg(QString::number(seconds * 1.0e6, 'g', 3));
  if (seconds < 1.0)
    return QStringLiteral("%1 ms").arg(QString::number(seconds * 1.0e3, 'g', 3));
  return QStringLiteral("%1 s").arg(QString::number(seconds, 'g', 3));
}

} // namespace

MesiExposurePlotWidget::MesiExposurePlotWidget(QWidget *parent) : QWidget(parent) {
  setMinimumSize(440, 300);
  setMouseTracking(true);
  setAutoFillBackground(true);
}

void MesiExposurePlotWidget::reset() {
  sample_     = {};
  has_sample_ = false;
  graph_rect_ = {};
  update();
}

void MesiExposurePlotWidget::set_sample(MesiExposureSample sample) {
  sample_     = std::move(sample);
  has_sample_ = true;
  graph_rect_ = {};
  update();
  emit samplesDisplayed();
}

QSize MesiExposurePlotWidget::sizeHint() const { return {720, 500}; }

void MesiExposurePlotWidget::paintEvent(QPaintEvent *) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.fillRect(rect(), palette().color(QPalette::Window));
  const QColor text  = palette().color(QPalette::WindowText);
  const QColor muted = palette().color(QPalette::PlaceholderText);
  painter.setPen(text);
  painter.drawText(QRect(18, 10, width() - 36, 24), Qt::AlignLeft | Qt::AlignVCenter,
                   tr("ROI mean K² (unitless) versus virtual exposure"));

  if (!has_sample_) {
    painter.setPen(muted);
    painter.drawText(rect().adjusted(20, 40, -20, -20), Qt::AlignCenter,
                     tr("Start MESI processing to see the next frame window."));
    return;
  }

  painter.setPen(muted);
  painter.drawText(QRect(18, 36, width() - 36, 22), Qt::AlignLeft | Qt::AlignVCenter,
                   tr("Window %1  ·  K²-complete ROI pixels %2 / %3  ·  exposure/frame period %4")
                       .arg(QString::number(sample_.window_index + 1))
                       .arg(QString::number(static_cast<qulonglong>(sample_.valid_pixels)))
                       .arg(QString::number(static_cast<qulonglong>(sample_.roi_pixels)))
                       .arg(QString::number(sample_.exposure_fraction, 'g', 3)));

  if (sample_.exposure_seconds.empty() || (sample_.valid_pixels == 0 && !sample_.single_pixel)) {
    painter.drawText(rect().adjusted(20, 70, -20, -20), Qt::AlignCenter,
                     tr("No ROI pixel has valid K² values at every exposure."));
    return;
  }

  if (sample_.single_pixel) {
    const bool valid_tau =
        std::isfinite(sample_.pixel_tau_seconds) && sample_.pixel_tau_seconds > 0.0;
    painter.setPen(valid_tau ? text : muted);
    painter.drawText(QRect(18, 62, width() - 36, 22), Qt::AlignLeft | Qt::AlignVCenter,
                     tr("Selected pixel: τ = %1  ·  β = %2  ·  ρ = %3")
                         .arg(valid_tau ? exposure_label(sample_.pixel_tau_seconds) : tr("invalid"))
                         .arg(QString::number(sample_.pixel_beta, 'g', 5))
                         .arg(QString::number(sample_.pixel_rho, 'g', 5)));
  }

  if (sample_.fit_valid) {
    painter.setPen(text);
    painter.drawText(QRect(18, sample_.single_pixel ? 84 : 62, width() - 36, 22),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     tr("ROI model fit: τ = %1  ·  β = %2  ·  ρ = %3  ·  RMSE = %4%5%6")
                         .arg(exposure_label(sample_.tau_seconds))
                         .arg(QString::number(sample_.beta, 'g', 4))
                         .arg(QString::number(sample_.rho, 'g', 4))
                         .arg(QString::number(sample_.rmse, 'g', 4))
                         .arg(sample_.fixed_beta_rho ? tr("  ·  β/ρ fixed")
                                                     : tr("  ·  separate from pixel REMI"))
                         .arg(sample_.tau_at_bound ? tr("  ·  τ at fit bound") : QString{}));
  } else {
    painter.setPen(muted);
    const bool negative_k2 =
        std::any_of(sample_.measured_k2.begin(), sample_.measured_k2.end(),
                    [](double value) { return std::isfinite(value) && value < 0.0; });
    painter.drawText(QRect(18, sample_.single_pixel ? 84 : 62, width() - 36, 22),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     negative_k2 ? tr("No ROI model fit; negative measured K² is shown below zero.")
                                 : tr("No ROI model fit; available measured K² remains visible."));
  }

  const int graph_top = sample_.single_pixel ? 128 : 106;
  graph_rect_ =
      QRectF(74, graph_top, std::max(1, width() - 106), std::max(1, height() - graph_top - 88));
  x_min_ = std::log(sample_.exposure_seconds.front());
  x_max_ = std::log(sample_.exposure_seconds.back());
  if (!(x_max_ > x_min_))
    x_max_ = x_min_ + 1.0;
  y_min_ = std::numeric_limits<double>::infinity();
  y_max_ = -std::numeric_limits<double>::infinity();
  for (double value : sample_.measured_k2) {
    if (std::isfinite(value)) {
      y_min_ = std::min(y_min_, value);
      y_max_ = std::max(y_max_, value);
    }
  }
  for (double value : sample_.model_k2) {
    if (std::isfinite(value)) {
      y_min_ = std::min(y_min_, value);
      y_max_ = std::max(y_max_, value);
    }
  }
  if (!std::isfinite(y_min_) || !std::isfinite(y_max_)) {
    painter.drawText(rect().adjusted(20, graph_top, -20, -65), Qt::AlignCenter,
                     tr("No finite measured K² value at this pixel."));
    return;
  }
  const double padding = std::max((y_max_ - y_min_) * 0.12, 1.0e-6);
  y_min_               = y_min_ < 0.0 ? y_min_ - padding : std::max(0.0, y_min_ - padding);
  y_max_ += padding;

  const auto px = [this](double seconds) {
    return graph_rect_.left() +
           (std::log(seconds) - x_min_) / (x_max_ - x_min_) * graph_rect_.width();
  };
  const auto py = [this](double k2) {
    return graph_rect_.bottom() - (k2 - y_min_) / (y_max_ - y_min_) * graph_rect_.height();
  };

  painter.setPen(QPen(muted, 1));
  painter.drawLine(graph_rect_.bottomLeft(), graph_rect_.topLeft());
  painter.drawLine(graph_rect_.bottomLeft(), graph_rect_.bottomRight());
  for (int tick = 0; tick <= 4; ++tick) {
    const double y_value = y_min_ + (y_max_ - y_min_) * tick / 4.0;
    const double y       = py(y_value);
    painter.setPen(QPen(muted, 1, Qt::DotLine));
    painter.drawLine(QPointF(graph_rect_.left(), y), QPointF(graph_rect_.right(), y));
    painter.setPen(text);
    painter.drawText(QRectF(2, y - 10, 66, 20), Qt::AlignRight | Qt::AlignVCenter,
                     QString::number(y_value, 'g', 4));
  }

  if (sample_.fit_valid && sample_.model_k2.size() == sample_.model_exposure_seconds.size()) {
    QPainterPath curve;
    bool         started = false;
    for (size_t index = 0; index < sample_.model_k2.size(); ++index) {
      const double seconds = sample_.model_exposure_seconds[index];
      const double value   = sample_.model_k2[index];
      if (!(seconds > 0.0) || !std::isfinite(value))
        continue;
      const QPointF point(px(seconds), py(value));
      if (!started) {
        curve.moveTo(point);
        started = true;
      } else {
        curve.lineTo(point);
      }
    }
    painter.setPen(QPen(QColor(69, 190, 255), 2));
    painter.drawPath(curve);
  }

  painter.setPen(QPen(QColor(255, 185, 54), 2));
  painter.setBrush(QColor(255, 185, 54));
  const double tick_width = std::min(
      86.0, graph_rect_.width() / std::max<size_t>(sample_.exposure_seconds.size() - 1, 1) * 0.9);
  for (size_t index = 0; index < sample_.exposure_seconds.size(); ++index) {
    const double seconds = sample_.exposure_seconds[index];
    const double value   = sample_.measured_k2[index];
    const double x       = px(seconds);
    if (std::isfinite(value))
      painter.drawEllipse(QPointF(x, py(value)), 4, 4);
    painter.setPen(text);
    painter.drawText(QRectF(x - tick_width / 2, graph_rect_.bottom() + 5, tick_width, 18),
                     Qt::AlignCenter, QString::number(sample_.exposure_frames[index]));
    if (tick_width >= 60.0)
      painter.drawText(QRectF(x - tick_width / 2, graph_rect_.bottom() + 23, tick_width, 18),
                       Qt::AlignCenter, exposure_label(seconds));
    painter.setPen(QPen(QColor(255, 185, 54), 2));
  }

  painter.setPen(text);
  painter.drawText(QRectF(graph_rect_.left(), height() - 37, graph_rect_.width(), 20),
                   Qt::AlignCenter,
                   tr("Nominal exposure span T = frames × frame period (log scale)"));
  painter.drawText(QRectF(2, graph_top - 18, 65, 18), Qt::AlignRight, tr("K²"));
  painter.setPen(QColor(255, 185, 54));
  painter.drawText(QRectF(graph_rect_.right() - 220, graph_top - 18, 105, 18), Qt::AlignRight,
                   tr("● measured"));
  painter.setPen(QColor(69, 190, 255));
  painter.drawText(QRectF(graph_rect_.right() - 105, graph_top - 18, 100, 18), Qt::AlignRight,
                   tr("— ROI fit"));
}

void MesiExposurePlotWidget::mouseMoveEvent(QMouseEvent *event) {
  if (!has_sample_ || !graph_rect_.contains(event->position()) ||
      sample_.exposure_seconds.empty()) {
    QToolTip::hideText();
    return;
  }
  const double x        = event->position().x();
  size_t       nearest  = 0;
  double       distance = std::numeric_limits<double>::infinity();
  for (size_t index = 0; index < sample_.exposure_seconds.size(); ++index) {
    const double point_x =
        graph_rect_.left() + (std::log(sample_.exposure_seconds[index]) - x_min_) /
                                 (x_max_ - x_min_) * graph_rect_.width();
    if (std::abs(point_x - x) < distance) {
      distance = std::abs(point_x - x);
      nearest  = index;
    }
  }
  if (distance > 18.0) {
    QToolTip::hideText();
    return;
  }
  QString   tooltip     = tr("%1 frames (%2)\nMean measured K²: %3")
                              .arg(sample_.exposure_frames[nearest])
                              .arg(exposure_label(sample_.exposure_seconds[nearest]))
                              .arg(std::isfinite(sample_.measured_k2[nearest])
                                       ? QString::number(sample_.measured_k2[nearest], 'g', 6)
                                       : tr("nonfinite"));
  const int frame_index = sample_.exposure_frames[nearest] - 1;
  if (sample_.fit_valid && frame_index >= 0 &&
      static_cast<size_t>(frame_index) < sample_.model_k2.size()) {
    const double predicted = sample_.model_k2[frame_index];
    tooltip += tr("\nROI model fit K²: %1\nResidual: %2")
                   .arg(QString::number(predicted, 'g', 6))
                   .arg(QString::number(sample_.measured_k2[nearest] - predicted, 'g', 6));
  }
  QToolTip::showText(event->globalPosition().toPoint(), tooltip, this);
}

void MesiExposurePlotWidget::leaveEvent(QEvent *event) {
  QToolTip::hideText();
  QWidget::leaveEvent(event);
}

} // namespace holovibes::ui
