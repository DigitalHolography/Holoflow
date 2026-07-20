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

#include <QWidget>

#include <optional>
#include <vector>

#include "signal_history.hh"

class QTimer;

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

  bool operator==(const ZernikeHistoryDisplaySettings &) const = default;
};

class ZernikeHistoryWidget : public QWidget {
  Q_OBJECT

public:
  explicit ZernikeHistoryWidget(QWidget *parent = nullptr);

  void start_run(double time_window_seconds);
  void resume_run();
  void stop_run();
  void set_time_window_seconds(double time_window_seconds);
  void append_samples(std::vector<SignalSample> samples);

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

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;

private:
  void update_recorded_extrema(double value);
  void initialize_recorded_extrema_from_visible_samples();
  void request_refresh();

  SignalHistory                 history_;
  ZernikeHistoryDisplaySettings display_settings_;

  // Recorded extrema outlive the rolling visible buffer so samples that scroll out cannot shrink
  // RecordedExtrema mode. Only exact data extrema are stored; display padding is computed later.
  std::optional<double> recorded_minimum_;
  std::optional<double> recorded_maximum_;

  QTimer *refresh_timer_ = nullptr;
  bool    active_        = false;
  bool    refresh_dirty_ = true;
};

} // namespace holovibes::ui
