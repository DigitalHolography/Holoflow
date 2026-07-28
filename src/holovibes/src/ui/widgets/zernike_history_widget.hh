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

#pragma once

#include <QString>
#include <QWidget>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "signal_history.hh"

class QTimer;
class QLabel;

namespace holovibes::ui {

enum class YAxisScalingMode {
  VisibleWindow,
  RecordedExtrema,
  Manual,
};

struct AxisRange {
  double minimum;
  double maximum;
};

struct ZernikeHistoryDisplaySettings {
  double           time_window_seconds = 8.0;
  YAxisScalingMode y_scaling_mode      = YAxisScalingMode::VisibleWindow;
  double           manual_y_minimum    = -1.0;
  double           manual_y_maximum    = 1.0;
  bool             show_statistics     = true;

  bool operator==(const ZernikeHistoryDisplaySettings &) const = default;
};

struct ZernikeHistorySample {
  int          noll_index;
  SignalSample sample;
};

class ZernikeHistoryWidget : public QWidget {
  Q_OBJECT

public:
  explicit ZernikeHistoryWidget(QWidget *parent = nullptr);
  ~ZernikeHistoryWidget() override;

  void start_run(double time_window_seconds, const std::vector<int> &indexes);
  void resume_run();
  void stop_run();
  void set_series(const std::vector<int> &indexes);
  void set_time_window_seconds(double time_window_seconds);
  void append_samples(std::vector<ZernikeHistorySample> samples);
  void show_waiting_placeholder(const QString &message = {});

  [[nodiscard]] ZernikeHistoryDisplaySettings display_settings() const;
  [[nodiscard]] AxisRange                     displayed_y_range() const;

  bool set_display_settings(const ZernikeHistoryDisplaySettings &settings);
  bool set_manual_y_range(double minimum, double maximum);
  void set_y_axis_scaling_mode(YAxisScalingMode mode);
  void reset_recorded_range_state();
  void reset_display_settings();

  [[nodiscard]] QSize sizeHint() const override;

signals:
  void configuration_requested(holovibes::ui::ZernikeHistoryWidget *widget);
  void display_settings_changed(const holovibes::ui::ZernikeHistoryDisplaySettings &settings);
  void samplesDisplayed();

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;

private:
  class CurveRenderWorker;

  struct Series {
    int                   noll_index;
    SignalHistory         history;
    std::optional<double> recorded_minimum;
    std::optional<double> recorded_maximum;
  };

  [[nodiscard]] Series   *find_series(int noll_index);
  [[nodiscard]] AxisRange displayed_y_range(const Series &series) const;
  void                    update_recorded_extrema(Series &series, double value);
  void                    initialize_recorded_extrema_from_visible_samples();
  void                    request_refresh();

  std::vector<Series>           series_;
  ZernikeHistoryDisplaySettings display_settings_;

  QTimer  *refresh_timer_   = nullptr;
  QLabel  *waiting_label_   = nullptr;
  bool     active_          = false;
  bool     refresh_dirty_   = true;
  uint64_t render_revision_ = 0;

  std::unique_ptr<CurveRenderWorker> curve_render_worker_;
};

} // namespace holovibes::ui
