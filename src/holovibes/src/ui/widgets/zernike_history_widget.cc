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

#include <QFont>
#include <QImage>
#include <QLabel>
#include <QMetaObject>
#include <QMouseEvent>
#include <QPainter>
#include <QPalette>
#include <QPointer>
#include <QPolygon>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace holovibes::ui {

namespace {

constexpr int    kRefreshIntervalMs         = 33;
constexpr int    kTickCount                 = 2;
constexpr double kUnsafeRasterCoordinateAbs = 1.0e6;
constexpr double kPlotGap                   = 8.0;

const std::array<QColor, 9> kCurveColors{
    QColor("#3DAEE9"), QColor("#E67E22"), QColor("#2ECC71"), QColor("#E74C3C"), QColor("#9B59B6"),
    QColor("#F1C40F"), QColor("#1ABC9C"), QColor("#E84393"), QColor("#95A5A6"),
};

struct PlotFonts {
  QFont widget_title;
  QFont subplot_title;
  QFont statistics;
  QFont axis;
};

struct GridLayout {
  QRectF content;
  int    columns     = 0;
  int    rows        = 0;
  double cell_width  = 0.0;
  double cell_height = 0.0;

  [[nodiscard]] bool valid() const { return cell_width > 1.0 && cell_height > 1.0; }

  [[nodiscard]] QRectF cell(size_t index) const {
    const int column = static_cast<int>(index) % columns;
    const int row    = static_cast<int>(index) / columns;
    return {content.left() + column * (cell_width + kPlotGap),
            content.top() + row * (cell_height + kPlotGap), cell_width, cell_height};
  }
};

struct CurveGeometry {
  std::vector<QPolygon> segments;
  QRectF                clip_rect;
  QColor                color;
  bool                  needs_clipping = false;
};

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

PlotFonts make_plot_fonts(const QPainter &painter) {
  const QString family = painter.font().family();

  PlotFonts fonts;
  fonts.widget_title = QFont(family);
  fonts.widget_title.setBold(true);

  fonts.subplot_title = QFont(family);
  fonts.subplot_title.setBold(true);
  fonts.subplot_title.setPixelSize(11);

  fonts.statistics = QFont(family);
  fonts.statistics.setPixelSize(9);

  fonts.axis = QFont(family);
  fonts.axis.setPixelSize(7);
  return fonts;
}

GridLayout make_grid_layout(const QRect &widget_rect, int series_count) {
  GridLayout layout;
  layout.columns     = series_count <= 3 ? 1 : series_count <= 6 ? 2 : 3;
  layout.rows        = (series_count + layout.columns - 1) / layout.columns;
  layout.content     = QRectF(widget_rect).adjusted(8.0, 28.0, -8.0, -8.0);
  layout.cell_width  = (layout.content.width() - (layout.columns - 1) * kPlotGap) / layout.columns;
  layout.cell_height = (layout.content.height() - (layout.rows - 1) * kPlotGap) / layout.rows;
  return layout;
}

QRectF make_plot_rect(const QRectF &cell, bool show_statistics) {
  const double header_height = show_statistics ? 32.0 : 20.0;
  return cell.adjusted(50.0, header_height, -8.0, -30.0);
}

void draw_plot_frame(QPainter &painter, const QRectF &cell, const QRectF &plot, int noll_index,
                     const QString &statistics_text, double time_window, AxisRange y_range,
                     const QColor &text_color, const QColor &grid_color, const PlotFonts &fonts,
                     const QString &time_axis_label) {
  painter.setFont(fonts.subplot_title);
  painter.setPen(text_color);
  painter.drawText(QRectF(cell.left(), cell.top(), cell.width(), 16.0), Qt::AlignCenter,
                   QString("a%1 (rad)").arg(noll_index));

  if (!statistics_text.isEmpty()) {
    painter.setFont(fonts.statistics);
    painter.drawText(QRectF(cell.left(), cell.top() + 16.0, cell.width(), 14.0), Qt::AlignCenter,
                     statistics_text);
  }

  painter.setFont(fonts.axis);
  painter.setPen(QPen(grid_color, 1.0));

  for (int tick = 0; tick <= kTickCount; ++tick) {
    const double fraction = static_cast<double>(tick) / kTickCount;
    const double x        = plot.left() + fraction * plot.width();
    const double y        = plot.bottom() - fraction * plot.height();

    painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
    painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));

    const double  relative_time = -time_window + fraction * time_window;
    const double  y_value       = y_range.minimum + fraction * (y_range.maximum - y_range.minimum);
    const QString time_label    = QString::number(relative_time, 'f', time_window < 2.0 ? 2 : 1);
    const QString value_label   = format_axis_value(y_value);

    painter.setPen(text_color);
    painter.drawText(QRectF(x - 28.0, plot.bottom() + 2.0, 56.0, 13.0), Qt::AlignHCenter,
                     time_label);
    painter.drawText(QRectF(cell.left(), y - 7.0, 45.0, 14.0), Qt::AlignRight | Qt::AlignVCenter,
                     value_label);
    painter.setPen(QPen(grid_color, 1.0));
  }

  painter.setPen(QPen(text_color, 1.0));
  painter.drawRect(plot);
  painter.drawText(QRectF(plot.left(), plot.bottom() + 15.0, plot.width(), 13.0), Qt::AlignCenter,
                   time_axis_label);
}

CurveGeometry build_curve(const std::deque<SignalSample> &samples, const QRectF &plot,
                          AxisRange y_range, double oldest_time, double newest_time,
                          double time_window, const QColor &color) {
  CurveGeometry curve{
      .clip_rect = plot,
      .color     = color,
  };

  QPolygon segment;
  segment.reserve(static_cast<qsizetype>(samples.size()));

  const auto flush_segment = [&]() {
    if (segment.empty()) {
      return;
    }
    curve.segments.push_back(std::move(segment));
    segment = QPolygon{};
  };

  const auto map_to_plot = [&](const SignalSample &sample) {
    const double relative_time = sample.time_seconds - newest_time;
    const double x_fraction    = (relative_time + time_window) / time_window;
    const double y_fraction =
        (sample.value - y_range.minimum) / (y_range.maximum - y_range.minimum);
    return QPointF(plot.left() + x_fraction * plot.width(),
                   plot.bottom() - y_fraction * plot.height());
  };

  for (const auto &sample : samples) {
    if (!std::isfinite(sample.time_seconds) || !std::isfinite(sample.value) ||
        sample.time_seconds < oldest_time || sample.time_seconds > newest_time) {
      flush_segment();
      continue;
    }

    const QPointF point = map_to_plot(sample);
    if (!std::isfinite(point.x()) || !std::isfinite(point.y()) ||
        std::abs(point.x()) > kUnsafeRasterCoordinateAbs ||
        std::abs(point.y()) > kUnsafeRasterCoordinateAbs) {
      flush_segment();
      continue;
    }

    curve.needs_clipping |= !plot.contains(point);

    // Curves are rendered without antialiasing, so sub-pixel coordinates provide no benefit.
    const QPoint raster_point = point.toPoint();
    if (segment.empty() || segment.back() != raster_point) {
      segment.append(raster_point);
    }
  }

  flush_segment();
  return curve;
}

} // namespace

class ZernikeHistoryWidget::CurveRenderWorker {
public:
  struct Job {
    uint64_t                   revision = 0;
    QSize                      logical_size;
    qreal                      device_pixel_ratio = 1.0;
    std::vector<CurveGeometry> curves;
  };

  explicit CurveRenderWorker(ZernikeHistoryWidget *widget)
      : widget_(widget), thread_([this]() { run(); }) {}

  ~CurveRenderWorker() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
      pending_job_.reset();
    }
    condition_.notify_one();
    thread_.join();
  }

  void submit(Job job) {
    {
      std::lock_guard lock(mutex_);
      if (is_duplicate(job)) {
        return;
      }

      last_submitted_revision_           = job.revision;
      last_submitted_size_               = job.logical_size;
      last_submitted_device_pixel_ratio_ = job.device_pixel_ratio;
      pending_job_                       = std::move(job);
    }
    condition_.notify_one();
  }

  [[nodiscard]] std::optional<QImage> latest_image(const QSize &logical_size) const {
    std::lock_guard lock(mutex_);
    if (!ready_image_.has_value() ||
        ready_image_->deviceIndependentSize().toSize() != logical_size) {
      return std::nullopt;
    }
    return ready_image_;
  }

private:
  [[nodiscard]] bool is_duplicate(const Job &job) const {
    return last_submitted_revision_ == job.revision && last_submitted_size_ == job.logical_size &&
           qFuzzyCompare(last_submitted_device_pixel_ratio_, job.device_pixel_ratio);
  }

  [[nodiscard]] std::optional<Job> wait_for_job() {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [this]() { return stopping_ || pending_job_.has_value(); });
    if (stopping_) {
      return std::nullopt;
    }

    Job job = std::move(*pending_job_);
    pending_job_.reset();
    return job;
  }

  static QImage render(const Job &job) {
    const QSize image_size(std::max(1, qRound(job.logical_size.width() * job.device_pixel_ratio)),
                           std::max(1, qRound(job.logical_size.height() * job.device_pixel_ratio)));

    QImage image(image_size, QImage::Format_ARGB32_Premultiplied);
    image.setDevicePixelRatio(job.device_pixel_ratio);
    image.fill(Qt::transparent);

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, false);

    for (const auto &curve : job.curves) {
      painter.save();

      // A zero-width cosmetic pen maps directly to a one-device-pixel raster line and avoids
      // Qt's more expensive general wide-pen stroker.
      QPen pen(curve.color);
      pen.setWidth(0);
      pen.setCosmetic(true);
      pen.setStyle(Qt::SolidLine);
      pen.setCapStyle(Qt::FlatCap);
      pen.setJoinStyle(Qt::BevelJoin);
      painter.setPen(pen);
      painter.setBrush(Qt::NoBrush);

      if (curve.needs_clipping) {
        painter.setClipRect(curve.clip_rect, Qt::ReplaceClip);
      } else {
        painter.setClipping(false);
      }

      for (const auto &segment : curve.segments) {
        if (segment.size() >= 2) {
          painter.drawPolyline(segment);
        } else if (segment.size() == 1) {
          painter.drawPoint(segment.front());
        }
      }

      painter.restore();
    }

    painter.end();
    return image;
  }

  bool publish(QImage image) {
    std::lock_guard lock(mutex_);
    if (stopping_) {
      return false;
    }
    ready_image_ = std::move(image);
    return true;
  }

  void notify_widget() const {
    const QPointer<ZernikeHistoryWidget> widget = widget_;
    if (widget.isNull()) {
      return;
    }

    QMetaObject::invokeMethod(
        widget.data(),
        [widget]() {
          if (!widget.isNull()) {
            widget->update();
          }
        },
        Qt::QueuedConnection);
  }

  void run() {
    while (const auto job = wait_for_job()) {
      if (publish(render(*job))) {
        notify_widget();
      }
    }
  }

  QPointer<ZernikeHistoryWidget> widget_;

  mutable std::mutex      mutex_;
  std::condition_variable condition_;
  std::optional<Job>      pending_job_;
  std::optional<QImage>   ready_image_;
  bool                    stopping_ = false;

  uint64_t last_submitted_revision_ = std::numeric_limits<uint64_t>::max();
  QSize    last_submitted_size_;
  qreal    last_submitted_device_pixel_ratio_ = 0.0;

  std::thread thread_;
};

ZernikeHistoryWidget::ZernikeHistoryWidget(QWidget *parent) : QWidget(parent) {
  setMinimumSize(320, 220);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(24, 24, 24, 24);

  waiting_label_ = new QLabel(tr("Waiting for data..."), this);
  waiting_label_->setObjectName("visualizationWorkspacePlaceholder");
  waiting_label_->setAlignment(Qt::AlignCenter);
  waiting_label_->setWordWrap(true);
  waiting_label_->setAttribute(Qt::WA_TransparentForMouseEvents);
  layout->addWidget(waiting_label_, 1);

  curve_render_worker_ = std::make_unique<CurveRenderWorker>(this);

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

ZernikeHistoryWidget::~ZernikeHistoryWidget() = default;

void ZernikeHistoryWidget::start_run(double time_window_seconds, const std::vector<int> &indexes) {
  const bool time_window_changed = display_settings_.time_window_seconds != time_window_seconds;
  display_settings_.time_window_seconds = time_window_seconds;

  set_series(indexes);
  for (auto &series : series_) {
    series.history.set_time_window_seconds(display_settings_.time_window_seconds);
    series.history.clear();
    series.recorded_range.reset();
  }

  waiting_label_->show();
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

void ZernikeHistoryWidget::set_series(const std::vector<int> &indexes) {
  const bool unchanged =
      series_.size() == indexes.size() &&
      std::equal(series_.begin(), series_.end(), indexes.begin(),
                 [](const Series &series, int index) { return series.noll_index == index; });
  if (unchanged) {
    return;
  }

  series_.clear();
  series_.reserve(indexes.size());
  for (const int index : indexes) {
    series_.push_back({index, SignalHistory(display_settings_.time_window_seconds)});
  }

  waiting_label_->show();
  request_refresh();
}

void ZernikeHistoryWidget::append_samples(std::vector<ZernikeHistorySample> samples) {
  if (!active_) {
    return;
  }

  bool appended = false;
  for (const auto &sample : samples) {
    if (auto *series = find_series(sample.noll_index);
        series != nullptr && series->history.append(sample.sample)) {
      update_recorded_range(*series, sample.sample.value);
      appended = true;
    }
  }

  if (!appended) {
    return;
  }

  waiting_label_->hide();
  request_refresh();
  emit samplesDisplayed();
}

void ZernikeHistoryWidget::show_waiting_placeholder(const QString &message) {
  waiting_label_->setText(message.isEmpty() ? tr("Waiting for data...") : message);
  waiting_label_->show();
  request_refresh();
}

ZernikeHistoryDisplaySettings ZernikeHistoryWidget::display_settings() const {
  return display_settings_;
}

AxisRange ZernikeHistoryWidget::displayed_y_range() const {
  switch (display_settings_.y_scaling_mode) {
  case YAxisScalingMode::VisibleWindow: {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto &series : series_) {
      if (const auto extrema = finite_extrema(series.history.samples())) {
        minimum = std::min(minimum, extrema->first);
        maximum = std::max(maximum, extrema->second);
      }
    }
    return std::isfinite(minimum) && std::isfinite(maximum) ? make_display_range(minimum, maximum)
                                                            : AxisRange{-1.0, 1.0};
  }

  case YAxisScalingMode::RecordedExtrema: {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto &series : series_) {
      if (series.recorded_range) {
        minimum = std::min(minimum, series.recorded_range->minimum);
        maximum = std::max(maximum, series.recorded_range->maximum);
      }
    }
    return std::isfinite(minimum) && std::isfinite(maximum) ? make_display_range(minimum, maximum)
                                                            : AxisRange{-1.0, 1.0};
  }

  case YAxisScalingMode::Manual:
    return {display_settings_.manual_y_minimum, display_settings_.manual_y_maximum};
  }

  return {-1.0, 1.0};
}

bool ZernikeHistoryWidget::set_display_settings(const ZernikeHistoryDisplaySettings &settings) {
  const bool valid =
      std::isfinite(settings.time_window_seconds) && settings.time_window_seconds > 0.0 &&
      is_valid_y_axis_scaling_mode(settings.y_scaling_mode) &&
      std::isfinite(settings.manual_y_minimum) && std::isfinite(settings.manual_y_maximum) &&
      settings.manual_y_minimum < settings.manual_y_maximum;
  if (!valid) {
    return false;
  }

  if (settings == display_settings_) {
    return true;
  }

  display_settings_ = settings;
  for (auto &series : series_) {
    series.history.set_time_window_seconds(display_settings_.time_window_seconds);
  }

  request_refresh();
  emit display_settings_changed(display_settings_);
  return true;
}

void ZernikeHistoryWidget::set_time_window_seconds(double time_window_seconds) {
  auto settings                = display_settings_;
  settings.time_window_seconds = time_window_seconds;
  set_display_settings(settings);
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
    const AxisRange current_range = displayed_y_range();
    settings.manual_y_minimum     = current_range.minimum;
    settings.manual_y_maximum     = current_range.maximum;
  }
  settings.y_scaling_mode = mode;
  set_display_settings(settings);
}

void ZernikeHistoryWidget::reset_recorded_range_state() {
  for (auto &series : series_) {
    series.recorded_range.reset();
  }
  initialize_recorded_ranges();
  request_refresh();
}

void ZernikeHistoryWidget::reset_display_settings() { set_display_settings({}); }

QSize ZernikeHistoryWidget::sizeHint() const { return {520, 300}; }

void ZernikeHistoryWidget::paintEvent(QPaintEvent *) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, false);

  const bool has_samples = std::ranges::any_of(
      series_, [](const Series &series) { return !series.history.samples().empty(); });
  painter.fillRect(rect(), palette().color(has_samples ? QPalette::Base : QPalette::Window));
  if (!has_samples) {
    return;
  }

  const QColor text_color = palette().color(QPalette::Text);
  QColor       grid_color = text_color;
  grid_color.setAlpha(45);

  const PlotFonts fonts = make_plot_fonts(painter);
  painter.setPen(text_color);
  painter.setFont(fonts.widget_title);
  painter.drawText(QRectF(0.0, 2.0, width(), 22.0), Qt::AlignCenter, tr("Zernike metrics"));

  const GridLayout layout = make_grid_layout(rect(), static_cast<int>(series_.size()));
  if (!layout.valid()) {
    return;
  }

  double newest_time = 0.0;
  for (const auto &series : series_) {
    const auto &samples = series.history.samples();
    if (!samples.empty()) {
      newest_time = std::max(newest_time, samples.back().time_seconds);
    }
  }

  const double time_window = display_settings_.time_window_seconds;
  const double oldest_time = newest_time - time_window;

  std::vector<CurveGeometry> curves;
  curves.reserve(series_.size());

  for (size_t i = 0; i < series_.size(); ++i) {
    const QRectF cell = layout.cell(i);
    const QRectF plot = make_plot_rect(cell, display_settings_.show_statistics);
    if (plot.width() <= 1.0 || plot.height() <= 1.0) {
      continue;
    }

    const Series   &series  = series_[i];
    const auto     &samples = series.history.samples();
    const AxisRange y_range = displayed_y_range(series);

    QString statistics_text;
    if (display_settings_.show_statistics) {
      if (const auto statistics = series.history.statistics(oldest_time)) {
        statistics_text = tr("Mean %1   SD %2")
                              .arg(format_axis_value(statistics->mean),
                                   format_axis_value(statistics->standard_deviation));
      }
    }

    draw_plot_frame(painter, cell, plot, series.noll_index, statistics_text, time_window, y_range,
                    text_color, grid_color, fonts, tr("Time (s)"));

    if (samples.empty()) {
      painter.drawText(plot, Qt::AlignCenter, tr("Waiting for data"));
      continue;
    }

    curves.push_back(build_curve(samples, plot, y_range, oldest_time, newest_time, time_window,
                                 kCurveColors[i % kCurveColors.size()]));
  }

  curve_render_worker_->submit({
      .revision           = render_revision_,
      .logical_size       = size(),
      .device_pixel_ratio = devicePixelRatioF(),
      .curves             = std::move(curves),
  });

  if (const auto image = curve_render_worker_->latest_image(size())) {
    painter.drawImage(QPointF(0.0, 0.0), *image);
  }
}

void ZernikeHistoryWidget::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    emit configuration_requested(this);
  }
  QWidget::mousePressEvent(event);
}

ZernikeHistoryWidget::Series *ZernikeHistoryWidget::find_series(int noll_index) {
  const auto it = std::ranges::find(series_, noll_index, &Series::noll_index);
  return it == series_.end() ? nullptr : &*it;
}

AxisRange ZernikeHistoryWidget::displayed_y_range(const Series &series) const {
  switch (display_settings_.y_scaling_mode) {
  case YAxisScalingMode::VisibleWindow:
    if (const auto extrema = finite_extrema(series.history.samples())) {
      return make_display_range(extrema->first, extrema->second);
    }
    return {-1.0, 1.0};

  case YAxisScalingMode::RecordedExtrema:
    if (series.recorded_range) {
      return make_display_range(series.recorded_range->minimum, series.recorded_range->maximum);
    }
    return {-1.0, 1.0};

  case YAxisScalingMode::Manual:
    return {display_settings_.manual_y_minimum, display_settings_.manual_y_maximum};
  }

  return {-1.0, 1.0};
}

void ZernikeHistoryWidget::update_recorded_range(Series &series, double value) {
  if (!std::isfinite(value)) {
    return;
  }

  if (!series.recorded_range) {
    series.recorded_range = AxisRange{value, value};
    return;
  }

  series.recorded_range->minimum = std::min(series.recorded_range->minimum, value);
  series.recorded_range->maximum = std::max(series.recorded_range->maximum, value);
}

void ZernikeHistoryWidget::initialize_recorded_ranges() {
  for (auto &series : series_) {
    for (const auto &sample : series.history.samples()) {
      update_recorded_range(series, sample.value);
    }
  }
}

void ZernikeHistoryWidget::request_refresh() {
  ++render_revision_;
  refresh_dirty_ = true;

  if (!refresh_timer_->isActive()) {
    refresh_dirty_ = false;
    update();
  }
}

} // namespace holovibes::ui
