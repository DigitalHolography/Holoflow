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

#include <QDebug>
#include <QElapsedTimer>
#include <QImage>
#include <QLabel>
#include <QMetaObject>
#include <QMouseEvent>
#include <QPainter>
#include <QPalette>
#include <QPointer>
#include <QPolygonF>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

namespace holovibes::ui {

namespace {

constexpr int kRefreshIntervalMs = 33;
constexpr int kTickCount         = 2;
constexpr int kProfileReportMs   = 1000;

struct PaintProfileAccumulator {
  QElapsedTimer report_timer;
  qint64        paint_count       = 0;
  qint64        plot_count        = 0;
  qint64        sample_count      = 0;
  qint64        total_ns          = 0;
  qint64        range_ns          = 0;
  qint64        chrome_ns         = 0;
  qint64        polyline_build_ns = 0;
  qint64        polyline_draw_ns  = 0;

  void record(qint64 paint_total_ns, qint64 paint_range_ns, qint64 paint_chrome_ns,
              qint64 paint_polyline_build_ns, qint64 paint_polyline_draw_ns, qint64 painted_plots,
              qint64 painted_samples) {
    if (!report_timer.isValid()) {
      report_timer.start();
    }

    ++paint_count;
    plot_count += painted_plots;
    sample_count += painted_samples;
    total_ns += paint_total_ns;
    range_ns += paint_range_ns;
    chrome_ns += paint_chrome_ns;
    polyline_build_ns += paint_polyline_build_ns;
    polyline_draw_ns += paint_polyline_draw_ns;

    const qint64 elapsed_ms = report_timer.elapsed();
    if (elapsed_ms < kProfileReportMs) {
      return;
    }

    constexpr double ns_to_ms    = 1.0e-6;
    const qint64     measured_ns = range_ns + chrome_ns + polyline_build_ns + polyline_draw_ns;
    const qint64     other_ns    = std::max<qint64>(0, total_ns - measured_ns);
    qInfo().nospace() << "Zernike paint profile (" << elapsed_ms << " ms): frames=" << paint_count
                      << ", plots=" << plot_count << ", samples=" << sample_count
                      << ", total=" << total_ns * ns_to_ms
                      << " ms, average=" << total_ns * ns_to_ms / static_cast<double>(paint_count)
                      << " ms/frame, setup/other=" << other_ns * ns_to_ms
                      << " ms, ranges=" << range_ns * ns_to_ms
                      << " ms, grid/text=" << chrome_ns * ns_to_ms
                      << " ms, polyline build=" << polyline_build_ns * ns_to_ms
                      << " ms, polyline draw=" << polyline_draw_ns * ns_to_ms << " ms";

    paint_count       = 0;
    plot_count        = 0;
    sample_count      = 0;
    total_ns          = 0;
    range_ns          = 0;
    chrome_ns         = 0;
    polyline_build_ns = 0;
    polyline_draw_ns  = 0;
    report_timer.restart();
  }
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

} // namespace

// -------------------------------------------------------------------------------------------------
// Curve rendering worker
// -------------------------------------------------------------------------------------------------

class ZernikeHistoryWidget::CurveRenderWorker {
public:
  struct Curve {
    QPolygonF points;
    QRectF    clip_rect;
    QColor    color;
  };

  struct Job {
    uint64_t           revision;
    QSize              logical_size;
    qreal              device_pixel_ratio;
    std::vector<Curve> curves;
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
      if (last_submitted_revision_ == job.revision && last_submitted_size_ == job.logical_size &&
          qFuzzyCompare(last_submitted_device_pixel_ratio_, job.device_pixel_ratio)) {
        return;
      }

      last_submitted_revision_           = job.revision;
      last_submitted_size_               = job.logical_size;
      last_submitted_device_pixel_ratio_ = job.device_pixel_ratio;
      pending_job_                       = std::move(job);
    }
    condition_.notify_one();
  }

  [[nodiscard]] QImage latest_image(const QSize &logical_size) const {
    std::lock_guard lock(mutex_);
    if (ready_image_.isNull() || ready_image_.deviceIndependentSize().toSize() != logical_size) {
      return {};
    }
    return ready_image_;
  }

private:
  void run() {
    while (true) {
      Job job;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [this]() { return stopping_ || pending_job_.has_value(); });
        if (stopping_) {
          return;
        }
        job = std::move(*pending_job_);
        pending_job_.reset();
      }

      const QSize image_size(
          std::max(1, qRound(job.logical_size.width() * job.device_pixel_ratio)),
          std::max(1, qRound(job.logical_size.height() * job.device_pixel_ratio)));
      QImage image(image_size, QImage::Format_ARGB32_Premultiplied);
      image.setDevicePixelRatio(job.device_pixel_ratio);
      image.fill(Qt::transparent);

      QPainter painter(&image);
      painter.setRenderHint(QPainter::Antialiasing, false);
      for (const auto &curve : job.curves) {
        painter.save();
        painter.setClipRect(curve.clip_rect);
        painter.setPen(QPen(curve.color, 2.0));
        painter.setBrush(Qt::NoBrush);
        painter.drawPolyline(curve.points);
        if (curve.points.size() == 1) {
          painter.setBrush(curve.color);
          painter.drawEllipse(curve.points.back(), 2.5, 2.5);
        }
        painter.restore();
      }
      painter.end();

      {
        std::lock_guard lock(mutex_);
        if (stopping_) {
          return;
        }
        ready_image_ = std::move(image);
      }

      QPointer<ZernikeHistoryWidget> widget = widget_;
      if (!widget.isNull()) {
        QMetaObject::invokeMethod(
            widget.data(),
            [widget]() {
              if (!widget.isNull()) {
                widget->update();
              }
            },
            Qt::QueuedConnection);
      }
    }
  }

  QPointer<ZernikeHistoryWidget> widget_;
  mutable std::mutex             mutex_;
  std::condition_variable        condition_;
  std::optional<Job>             pending_job_;
  QImage                         ready_image_;
  uint64_t                       last_submitted_revision_ = std::numeric_limits<uint64_t>::max();
  QSize                          last_submitted_size_;
  qreal                          last_submitted_device_pixel_ratio_ = 0.0;
  bool                           stopping_                          = false;
  std::thread                    thread_;
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

  curve_render_worker_ = std::make_unique<CurveRenderWorker>(this);
}

ZernikeHistoryWidget::~ZernikeHistoryWidget() = default;

void ZernikeHistoryWidget::start_run(double time_window_seconds, const std::vector<int> &indexes) {
  const bool time_window_changed = display_settings_.time_window_seconds != time_window_seconds;
  display_settings_.time_window_seconds = time_window_seconds;
  set_series(indexes);
  for (auto &series : series_) {
    series.history.set_time_window_seconds(display_settings_.time_window_seconds);
    series.history.clear();
    series.recorded_minimum.reset();
    series.recorded_maximum.reset();
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

void ZernikeHistoryWidget::set_time_window_seconds(double time_window_seconds) {
  auto settings                = display_settings_;
  settings.time_window_seconds = time_window_seconds;
  set_display_settings(settings);
}

void ZernikeHistoryWidget::append_samples(std::vector<ZernikeHistorySample> samples) {
  if (!active_) {
    return;
  }

  bool appended = false;
  for (const auto &sample : samples) {
    auto *series = find_series(sample.noll_index);
    if (series != nullptr && series->history.append(sample.sample)) {
      update_recorded_extrema(*series, sample.sample.value);
      appended = true;
    }
  }
  if (appended) {
    waiting_label_->hide();
    request_refresh();
    emit samplesDisplayed();
  }
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
      if (const auto extrema = finite_extrema(series.history.samples()); extrema.has_value()) {
        minimum = std::min(minimum, extrema->first);
        maximum = std::max(maximum, extrema->second);
      }
    }
    if (std::isfinite(minimum) && std::isfinite(maximum)) {
      return make_display_range(minimum, maximum);
    }
  }
    return {-1.0, 1.0};
  case YAxisScalingMode::RecordedExtrema: {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto &series : series_) {
      if (series.recorded_minimum.has_value() && series.recorded_maximum.has_value()) {
        minimum = std::min(minimum, *series.recorded_minimum);
        maximum = std::max(maximum, *series.recorded_maximum);
      }
    }
    if (std::isfinite(minimum) && std::isfinite(maximum)) {
      return make_display_range(minimum, maximum);
    }
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
  for (auto &series : series_) {
    series.history.set_time_window_seconds(display_settings_.time_window_seconds);
  }
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
  for (auto &series : series_) {
    series.recorded_minimum.reset();
    series.recorded_maximum.reset();
  }
  initialize_recorded_extrema_from_visible_samples();
  request_refresh();
}

void ZernikeHistoryWidget::reset_display_settings() {
  // Display reset restores axis policy but deliberately preserves runtime recorded-range state.
  set_display_settings({});
}

QSize ZernikeHistoryWidget::sizeHint() const { return {520, 300}; }

void ZernikeHistoryWidget::paintEvent(QPaintEvent *) {
  QElapsedTimer paint_timer;
  QElapsedTimer phase_timer;
  paint_timer.start();

  qint64                                range_ns          = 0;
  qint64                                chrome_ns         = 0;
  qint64                                polyline_build_ns = 0;
  qint64                                polyline_draw_ns  = 0;
  qint64                                plotted           = 0;
  qint64                                samples_drawn     = 0;
  std::vector<CurveRenderWorker::Curve> curves;
  curves.reserve(series_.size());

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
  const std::array<QColor, 9> curve_colors{
      QColor("#3DAEE9"), QColor("#E67E22"), QColor("#2ECC71"), QColor("#E74C3C"), QColor("#9B59B6"),
      QColor("#F1C40F"), QColor("#1ABC9C"), QColor("#E84393"), QColor("#95A5A6"),
  };

  painter.setPen(text_color);
  QFont title_font = painter.font();
  title_font.setBold(true);
  painter.setFont(title_font);
  painter.drawText(QRectF(0.0, 2.0, width(), 22.0), Qt::AlignCenter, tr("Zernike metrics"));

  const int        series_count = static_cast<int>(series_.size());
  const int        columns      = series_count <= 3 ? 1 : series_count <= 6 ? 2 : 3;
  const int        rows         = (series_count + columns - 1) / columns;
  constexpr double gap          = 8.0;
  const QRectF     content      = QRectF(rect()).adjusted(8.0, 28.0, -8.0, -8.0);
  const double     cell_width   = (content.width() - (columns - 1) * gap) / columns;
  const double     cell_height  = (content.height() - (rows - 1) * gap) / rows;
  if (cell_width <= 1.0 || cell_height <= 1.0) {
    return;
  }

  const double time_window = display_settings_.time_window_seconds;
  double       newest_time = 0.0;
  for (const auto &series : series_) {
    const auto &samples = series.history.samples();
    if (!samples.empty()) {
      newest_time = std::max(newest_time, samples.back().time_seconds);
    }
  }

  for (size_t i = 0; i < series_.size(); ++i) {
    const int    column = static_cast<int>(i) % columns;
    const int    row    = static_cast<int>(i) / columns;
    const QRectF cell(content.left() + column * (cell_width + gap),
                      content.top() + row * (cell_height + gap), cell_width, cell_height);
    const double header_height = display_settings_.show_statistics ? 32.0 : 20.0;
    const QRectF plot          = cell.adjusted(50.0, header_height, -8.0, -30.0);
    if (plot.width() <= 1.0 || plot.height() <= 1.0) {
      continue;
    }

    const auto &series  = series_[i];
    const auto &samples = series.history.samples();
    phase_timer.restart();
    const AxisRange y_range = displayed_y_range(series);
    range_ns += phase_timer.nsecsElapsed();
    const double y_min       = y_range.minimum;
    const double y_max       = y_range.maximum;
    const auto   map_to_plot = [&](double relative_time, double value) {
      const double x_fraction = (relative_time + time_window) / time_window;
      const double y_fraction = (value - y_min) / (y_max - y_min);
      return QPointF(plot.left() + x_fraction * plot.width(),
                     plot.bottom() - y_fraction * plot.height());
    };

    phase_timer.restart();
    QFont subplot_title_font = painter.font();
    subplot_title_font.setBold(true);
    subplot_title_font.setPixelSize(11);
    painter.setFont(subplot_title_font);
    painter.setPen(text_color);
    painter.drawText(QRectF(cell.left(), cell.top(), cell.width(), 16.0), Qt::AlignCenter,
                     QString("a%1 (rad)").arg(series.noll_index));

    if (display_settings_.show_statistics) {
      const auto statistics = series.history.statistics(newest_time - time_window);
      if (statistics.has_value()) {
        QFont statistics_font = painter.font();
        statistics_font.setBold(false);
        statistics_font.setPixelSize(9);
        painter.setFont(statistics_font);
        painter.drawText(QRectF(cell.left(), cell.top() + 16.0, cell.width(), 14.0),
                         Qt::AlignCenter,
                         tr("Mean %1   SD %2")
                             .arg(format_axis_value(statistics->mean),
                                  format_axis_value(statistics->standard_deviation)));
      }
    }

    painter.setFont(QFont(painter.font().family(), 7));
    painter.setPen(QPen(grid_color, 1.0));
    for (int tick = 0; tick <= kTickCount; ++tick) {
      const double fraction = static_cast<double>(tick) / kTickCount;
      const double x        = plot.left() + fraction * plot.width();
      const double y        = plot.bottom() - fraction * plot.height();
      painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
      painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));

      painter.setPen(text_color);
      const double relative_time = -time_window + fraction * time_window;
      painter.drawText(QRectF(x - 28.0, plot.bottom() + 2.0, 56.0, 13.0), Qt::AlignHCenter,
                       QString::number(relative_time, 'f', time_window < 2.0 ? 2 : 1));
      const double y_value = y_min + fraction * (y_max - y_min);
      painter.drawText(QRectF(cell.left(), y - 7.0, 45.0, 14.0), Qt::AlignRight | Qt::AlignVCenter,
                       format_axis_value(y_value));
      painter.setPen(QPen(grid_color, 1.0));
    }

    painter.setPen(QPen(text_color, 1.0));
    painter.drawRect(plot);
    painter.drawText(QRectF(plot.left(), plot.bottom() + 15.0, plot.width(), 13.0), Qt::AlignCenter,
                     tr("Time (s)"));
    chrome_ns += phase_timer.nsecsElapsed();
    ++plotted;

    if (samples.empty()) {
      phase_timer.restart();
      painter.drawText(plot, Qt::AlignCenter, tr("Waiting for data"));
      chrome_ns += phase_timer.nsecsElapsed();
      continue;
    }

    phase_timer.restart();
    QPolygonF curve;
    curve.reserve(static_cast<qsizetype>(samples.size()));
    for (const auto &sample : samples) {
      curve.append(map_to_plot(sample.time_seconds - newest_time, sample.value));
    }
    polyline_build_ns += phase_timer.nsecsElapsed();
    samples_drawn += static_cast<qint64>(samples.size());

    const QColor curve_color = curve_colors[i % curve_colors.size()];
    curves.push_back({std::move(curve), plot, curve_color});
  }

  curve_render_worker_->submit({
      .revision           = render_revision_,
      .logical_size       = size(),
      .device_pixel_ratio = devicePixelRatioF(),
      .curves             = std::move(curves),
  });

  phase_timer.restart();
  const QImage curve_image = curve_render_worker_->latest_image(size());
  if (!curve_image.isNull()) {
    painter.drawImage(QPointF(0.0, 0.0), curve_image);
  }
  polyline_draw_ns += phase_timer.nsecsElapsed();

  painter.end();
  static PaintProfileAccumulator profile;
  profile.record(paint_timer.nsecsElapsed(), range_ns, chrome_ns, polyline_build_ns,
                 polyline_draw_ns, plotted, samples_drawn);
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
    if (const auto extrema = finite_extrema(series.history.samples()); extrema.has_value()) {
      return make_display_range(extrema->first, extrema->second);
    }
    return {-1.0, 1.0};
  case YAxisScalingMode::RecordedExtrema:
    if (series.recorded_minimum.has_value() && series.recorded_maximum.has_value()) {
      return make_display_range(*series.recorded_minimum, *series.recorded_maximum);
    }
    return {-1.0, 1.0};
  case YAxisScalingMode::Manual:
    return {display_settings_.manual_y_minimum, display_settings_.manual_y_maximum};
  }

  return {-1.0, 1.0};
}

void ZernikeHistoryWidget::update_recorded_extrema(Series &series, double value) {
  if (!std::isfinite(value)) {
    return;
  }
  series.recorded_minimum =
      series.recorded_minimum.has_value() ? std::min(*series.recorded_minimum, value) : value;
  series.recorded_maximum =
      series.recorded_maximum.has_value() ? std::max(*series.recorded_maximum, value) : value;
}

void ZernikeHistoryWidget::initialize_recorded_extrema_from_visible_samples() {
  for (auto &series : series_) {
    for (const auto &sample : series.history.samples()) {
      update_recorded_extrema(series, sample.value);
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
