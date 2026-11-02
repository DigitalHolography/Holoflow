// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesi_exposure_plot_widget.hh"
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QPoint>
#include <QPointF>
#include <QRect>
#include <QSlider>
#include <QSpinBox>
#include <QWidget>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace holovibes::ui {
class TensorDisplayWidget;

struct MesiPixelValues {
  QPoint             pixel;
  uint64_t           packet_index = 0;
  std::vector<float> k2;
  float              beta        = 0.0F;
  float              rho         = 0.0F;
  float              tau_seconds = 0.0F;
  float              map_value   = 0.0F;
};

class MesiDashboardWidget : public QWidget {
  Q_OBJECT

public:
  explicit MesiDashboardWidget(QWidget *parent = nullptr);
  [[nodiscard]] TensorDisplayWidget    *tau_map_widget() const;
  [[nodiscard]] MesiExposurePlotWidget *exposure_plot_widget() const;
  [[nodiscard]] QRect                   selected_roi() const;
  void                                  set_map_size(int width, int height);
  void                                  set_tau_range(float minimum_seconds, float maximum_seconds);
  [[nodiscard]] int                     display_exposure_frames() const;
  void                                  set_display_exposure_frames(int frames);
  void                                  set_exposure_levels(const std::vector<int> &frames);
  void                                  set_display_selector_enabled(bool enabled);
  [[nodiscard]] QPoint                  hovered_pixel() const;
  [[nodiscard]] QPoint                  pixel_to_sample() const;
  void                                  set_pixel_values(const MesiPixelValues &values);
  std::vector<uint64_t>  prepare_curve_cache(const std::string &key, size_t packet_count);
  [[nodiscard]] uint64_t curve_cache_revision() const { return curve_cache_revision_.load(); }
  uint64_t               begin_map_run() { return ++map_revision_; }
  [[nodiscard]] uint64_t map_revision() const { return map_revision_.load(); }
  void                   reset_curve();
  void                   append_curve_sample(MesiExposureSample sample);

signals:
  void samplesDisplayed();
  void displayExposureChanged(int exposure_frames);

private:
  TensorDisplayWidget    *tau_map_widget_       = nullptr;
  MesiExposurePlotWidget *exposure_plot_widget_ = nullptr;
  class RoiOverlay;
  RoiOverlay                    *roi_overlay_    = nullptr;
  QSpinBox                      *roi_x_          = nullptr;
  QSpinBox                      *roi_y_          = nullptr;
  QSpinBox                      *roi_width_      = nullptr;
  QSpinBox                      *roi_height_     = nullptr;
  QLabel                        *roi_label_      = nullptr;
  QLabel                        *tau_low_label_  = nullptr;
  QLabel                        *tau_high_label_ = nullptr;
  QLabel                        *map_label_      = nullptr;
  QLabel                        *map_help_       = nullptr;
  QLabel                        *pixel_label_    = nullptr;
  QComboBox                     *map_selector_   = nullptr;
  QCheckBox                     *hold_curve_     = nullptr;
  QSlider                       *packet_slider_  = nullptr;
  QLabel                        *packet_label_   = nullptr;
  std::deque<MesiExposureSample> curve_samples_;
  std::string                    curve_cache_key_;
  std::vector<int>               exposure_levels_;
  QPoint                         pending_click_pixel_{-1, -1};
  QPointF                        pending_click_position_;
  int                            requested_display_exposure_frames_ = 0;
  size_t                         source_packet_count_               = 0;
  int                            map_width_                         = 0;
  int                            map_height_                        = 0;
  std::atomic<uint64_t>          curve_cache_revision_{0};
  std::atomic<uint64_t>          map_revision_{0};

  void show_selected_curve();
  void log_clicked_pixel(const MesiPixelValues &values);
  void update_tau_legend();
};

} // namespace holovibes::ui
