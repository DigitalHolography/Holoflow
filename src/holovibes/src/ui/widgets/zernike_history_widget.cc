// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "zernike_history_widget.hh"

#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <limits>

namespace holovibes::ui {

namespace {

constexpr int kRefreshIntervalMs = 33;
constexpr int kTickCount         = 5;

std::optional<std::pair<double, double>> finite_extrema(const std::deque<SignalSample> &samples) {
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = -std::numeric_limits<double>::infinity();

  for (const auto &sample : samples) {
    if (!std::isfinite(sample.value)) {
      continue;
    }
    minimum = std::min(minimum, sample.value);
    maximum = std::max(maximum, sample.value);
  }

  if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
    return std::nullopt;
  }

  return std::pair{minimum, maximum};
}

AxisRange make_display_range(double minimum, double maximum) {
  if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum) {
    return {-1.0, 1.0};
  }

  const double span  = maximum - minimum;
  const double scale = std::max(std::abs(minimum), std::abs(maximum));
  if (span <= std::max(1e-12, scale * 1e-9)) {
    const double half_range = std::max(std::abs(minimum) * 0.1, 1e-3);
    return {minimum - half_range, maximum + half_range};
  }

  const double margin =
      std::max({span * 0.08, std::abs(minimum) * 0.02, std::abs(maximum) * 0.02, 1e-6});
  return {minimum - margin, maximum + margin};
}

bool is_valid_y_axis_scaling_mode(YAxisScalingMode mode) {
  switch (mode) {
  case YAxisScalingMode::VisibleWindow:
  case YAxisScalingMode::RecordedExtrema:
  case YAxisScalingMode::Manual:
    return true;
  }

  return false;
}

QString format_axis_value(double value) {
  const double magnitude = std::abs(value);
  if ((magnitude > 0.0 && magnitude < 1e-3) || magnitude >= 1e4) {
    return QString::number(value, 'e', 2);
  }
  return QString::number(value, 'f', magnitude < 1.0 ? 3 : 2);
}

} // namespace

ZernikeHistoryWidget::ZernikeHistoryWidget(QWidget *parent) : QWidget(parent) {
  setMinimumSize(320, 220);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  refresh_timer_ = new QTimer(this);
  refresh_timer_->setInterval(kRefreshIntervalMs);
  refresh_timer_->setTimerType(Qt::CoarseTimer);
  connect(refresh_timer_, &QTimer::timeout, this, [this]() {
    if (!refresh_dirty_) {
      return;
    }
    refresh_dirty_ = false;
    update();
  });
}

void ZernikeHistoryWidget::start_run(double time_window_seconds) {
  const bool time_window_changed = display_settings_.time_window_seconds != time_window_seconds;
  display_settings_.time_window_seconds = time_window_seconds;
  history_.set_time_window_seconds(display_settings_.time_window_seconds);
  history_.clear();
  recorded_minimum_.reset();
  recorded_maximum_.reset();
  active_ = true;
  refresh_timer_->start();
  request_refresh();
  if (time_window_changed) {
    emit display_settings_changed(display_settings_);
  }
}

void ZernikeHistoryWidget::resume_run() {
  active_ = true;
  refresh_timer_->start();
  request_refresh();
}

void ZernikeHistoryWidget::stop_run() {
  active_ = false;
  refresh_timer_->stop();
}

void ZernikeHistoryWidget::set_time_window_seconds(double time_window_seconds) {
  auto settings                = display_settings_;
  settings.time_window_seconds = time_window_seconds;
  set_display_settings(settings);
}

void ZernikeHistoryWidget::append_samples(std::vector<SignalSample> samples) {
  if (!active_) {
    return;
  }

  bool appended = false;
  for (const auto &sample : samples) {
    if (history_.append(sample)) {
      update_recorded_extrema(sample.value);
      appended = true;
    }
  }
  if (appended) {
    request_refresh();
  }
}

ZernikeHistoryDisplaySettings ZernikeHistoryWidget::display_settings() const {
  return display_settings_;
}

AxisRange ZernikeHistoryWidget::displayed_y_range() const {
  switch (display_settings_.y_scaling_mode) {
  case YAxisScalingMode::VisibleWindow:
    if (const auto extrema = finite_extrema(history_.samples()); extrema.has_value()) {
      return make_display_range(extrema->first, extrema->second);
    }
    return {-1.0, 1.0};
  case YAxisScalingMode::RecordedExtrema:
    if (recorded_minimum_.has_value() && recorded_maximum_.has_value()) {
      return make_display_range(*recorded_minimum_, *recorded_maximum_);
    }
    return {-1.0, 1.0};
  case YAxisScalingMode::Manual:
    return {display_settings_.manual_y_minimum, display_settings_.manual_y_maximum};
  }

  return {-1.0, 1.0};
}

bool ZernikeHistoryWidget::set_display_settings(const ZernikeHistoryDisplaySettings &settings) {
  if (!std::isfinite(settings.time_window_seconds) || settings.time_window_seconds <= 0.0 ||
      !is_valid_y_axis_scaling_mode(settings.y_scaling_mode) ||
      !std::isfinite(settings.manual_y_minimum) || !std::isfinite(settings.manual_y_maximum) ||
      settings.manual_y_minimum >= settings.manual_y_maximum) {
    return false;
  }

  if (settings == display_settings_) {
    return true;
  }

  display_settings_ = settings;
  history_.set_time_window_seconds(display_settings_.time_window_seconds);
  request_refresh();
  emit display_settings_changed(display_settings_);
  return true;
}

bool ZernikeHistoryWidget::set_manual_y_range(double minimum, double maximum) {
  auto settings             = display_settings_;
  settings.manual_y_minimum = minimum;
  settings.manual_y_maximum = maximum;
  return set_display_settings(settings);
}

void ZernikeHistoryWidget::set_y_axis_scaling_mode(YAxisScalingMode mode) {
  if (mode == display_settings_.y_scaling_mode) {
    return;
  }

  auto settings = display_settings_;
  if (mode == YAxisScalingMode::Manual) {
    const auto current_range  = displayed_y_range();
    settings.manual_y_minimum = current_range.minimum;
    settings.manual_y_maximum = current_range.maximum;
  }
  settings.y_scaling_mode = mode;
  set_display_settings(settings);
}

void ZernikeHistoryWidget::reset_recorded_range_state() {
  // This reset removes old outlier influence without changing display settings, samples, or time.
  recorded_minimum_.reset();
  recorded_maximum_.reset();
  initialize_recorded_extrema_from_visible_samples();
  request_refresh();
}

void ZernikeHistoryWidget::reset_display_settings() {
  // Display reset restores axis policy but deliberately preserves runtime recorded-range state.
  set_display_settings({});
}

QSize ZernikeHistoryWidget::sizeHint() const { return {520, 300}; }

void ZernikeHistoryWidget::paintEvent(QPaintEvent *) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.fillRect(rect(), palette().color(QPalette::Base));

  constexpr int left_margin   = 70;
  constexpr int top_margin    = 34;
  constexpr int right_margin  = 20;
  constexpr int bottom_margin = 54;
  const QRectF  plot =
      QRectF(rect()).adjusted(left_margin, top_margin, -right_margin, -bottom_margin);
  if (plot.width() <= 1.0 || plot.height() <= 1.0) {
    return;
  }

  const QColor text_color = palette().color(QPalette::Text);
  QColor       grid_color = text_color;
  grid_color.setAlpha(45);
  QColor curve_color = palette().color(QPalette::Highlight);
  if (!curve_color.isValid()) {
    curve_color = QColor(50, 170, 210);
  }

  painter.setPen(text_color);
  QFont title_font = painter.font();
  title_font.setBold(true);
  painter.setFont(title_font);
  painter.drawText(QRectF(0.0, 4.0, width(), 24.0), Qt::AlignCenter, tr("Zernike a4"));

  const auto     &samples     = history_.samples();
  const double    time_window = history_.time_window_seconds();
  const AxisRange y_range     = displayed_y_range();
  const double    y_min       = y_range.minimum;
  const double    y_max       = y_range.maximum;
  const double    newest_time = samples.empty() ? 0.0 : samples.back().time_seconds;
  const auto      map_to_plot = [&](double relative_time, double value) {
    const double x_fraction = (relative_time + time_window) / time_window;
    const double y_fraction = (value - y_min) / (y_max - y_min);
    return QPointF(plot.left() + x_fraction * plot.width(),
                   plot.bottom() - y_fraction * plot.height());
  };

  painter.setFont(QFont(painter.font().family(), 8));
  painter.setPen(QPen(grid_color, 1.0));
  for (int tick = 0; tick <= kTickCount; ++tick) {
    const double fraction = static_cast<double>(tick) / kTickCount;
    const double x        = plot.left() + fraction * plot.width();
    const double y        = plot.bottom() - fraction * plot.height();
    painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
    painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));

    painter.setPen(text_color);
    const double relative_time = -time_window + fraction * time_window;
    painter.drawText(QRectF(x - 34.0, plot.bottom() + 5.0, 68.0, 18.0), Qt::AlignHCenter,
                     QString::number(relative_time, 'f', time_window < 2.0 ? 2 : 1));
    const double y_value = y_min + fraction * (y_max - y_min);
    painter.drawText(QRectF(2.0, y - 9.0, left_margin - 9.0, 18.0),
                     Qt::AlignRight | Qt::AlignVCenter, format_axis_value(y_value));
    painter.setPen(QPen(grid_color, 1.0));
  }

  painter.setPen(QPen(text_color, 1.0));
  painter.drawRect(plot);
  painter.drawText(QRectF(plot.left(), height() - 26.0, plot.width(), 20.0), Qt::AlignCenter,
                   tr("Time (s)"));

  painter.save();
  painter.translate(17.0, plot.center().y());
  painter.rotate(-90.0);
  painter.drawText(QRectF(-plot.height() / 2.0, -10.0, plot.height(), 20.0), Qt::AlignCenter,
                   tr("a4 (rad)"));
  painter.restore();

  if (samples.empty()) {
    QColor no_data_color = text_color;
    no_data_color.setAlpha(150);
    painter.setPen(no_data_color);
    painter.drawText(plot, Qt::AlignCenter, tr("No data"));
    return;
  }

  QPainterPath curve;
  bool         started = false;
  for (const auto &sample : samples) {
    if (!std::isfinite(sample.value)) {
      continue;
    }
    const QPointF point = map_to_plot(sample.time_seconds - newest_time, sample.value);
    if (!started) {
      curve.moveTo(point);
      started = true;
    } else {
      curve.lineTo(point);
    }
  }

  painter.save();
  painter.setClipRect(plot);
  painter.setPen(QPen(curve_color, 2.0));
  painter.drawPath(curve);
  if (samples.size() == 1) {
    painter.setBrush(curve_color);
    painter.drawEllipse(map_to_plot(0.0, samples.back().value), 2.5, 2.5);
  }
  painter.restore();
}

void ZernikeHistoryWidget::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    emit configuration_requested(this);
  }
  QWidget::mousePressEvent(event);
}

void ZernikeHistoryWidget::update_recorded_extrema(double value) {
  if (!std::isfinite(value)) {
    return;
  }
  recorded_minimum_ = recorded_minimum_.has_value() ? std::min(*recorded_minimum_, value) : value;
  recorded_maximum_ = recorded_maximum_.has_value() ? std::max(*recorded_maximum_, value) : value;
}

void ZernikeHistoryWidget::initialize_recorded_extrema_from_visible_samples() {
  for (const auto &sample : history_.samples()) {
    update_recorded_extrema(sample.value);
  }
}

void ZernikeHistoryWidget::request_refresh() {
  refresh_dirty_ = true;
  if (!refresh_timer_->isActive()) {
    refresh_dirty_ = false;
    update();
  }
}

} // namespace holovibes::ui
