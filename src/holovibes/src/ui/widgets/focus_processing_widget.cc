// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "ui/widgets/focus_processing_widget.hh"
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QSignalBlocker>
#include <QSpinBox>
#include <algorithm>

namespace holovibes::ui {
FocusProcessingWidget::FocusProcessingWidget(QWidget *parent) : QWidget(parent) {
  auto *layout = new QFormLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  auto add_int = [this, layout](const QString &label, int minimum, int maximum) {
    auto *spin = new QSpinBox(this);
    spin->setRange(minimum, maximum);
    spin->setKeyboardTracking(false);
    layout->addRow(label, spin);
    connect(spin, qOverload<int>(&QSpinBox::valueChanged), this,
            [this] { emit settings_changed(); });
    return spin;
  };
  method_ = new QComboBox(this);
  method_->addItem(tr("Holovibes STFT volume"), "holovibes_stft");
  method_->addItem(tr("Single-frame propagation stack"), "propagation");
  layout->addRow(tr("Reconstruction"), method_);
  playback_ = new QCheckBox(tr("Continuous playback (loop)"), this);
  layout->addRow(playback_);
  playback_fps_      = add_int(tr("Playback FPS limit"), 1, 240);
  playback_position_ = new QLabel(tr("No window displayed"), this);
  playback_position_->setWordWrap(true);
  layout->addRow(playback_position_);
  window_         = add_int(tr("STFT window (frames)"), 2, 2048);
  temporal_focus_ = new QDoubleSpinBox(this);
  temporal_pitch_ = new QDoubleSpinBox(this);
  for (auto *spin : {temporal_focus_, temporal_pitch_}) {
    spin->setDecimals(6);
    spin->setRange(spin == temporal_pitch_ ? 0.000001 : -100000, 100000);
    spin->setKeyboardTracking(false);
    connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this] { emit settings_changed(); });
  }
  layout->addRow(tr("STFT spatial focus (mm)"), temporal_focus_);
  layout->addRow(tr("STFT sensor pitch (um)"), temporal_pitch_);
  temporal_pitch_->setToolTip(
      tr("Reference preset: 12 um, inferred from the shown Holovibes boundary N*d*d/lambda = 86.5 "
         "mm. This differs from the stale footer pitch of 20 um."));
  temporal_filter_ = new QCheckBox(tr("STFT spatial filter"), this);
  layout->addRow(temporal_filter_);
  filter_inner_ = add_int(tr("Filter inner radius (pixels)"), 0, 32768);
  filter_outer_ = add_int(tr("Filter outer radius (pixels)"), 1, 32768);
  frame_        = add_int(tr("Time frame (0-based)"), 0, 2147483646);
  frame_label_  = qobject_cast<QLabel *>(layout->labelForField(frame_));
  planes_       = add_int(tr("Z planes"), 2, 2048);
  const QStringList labels{tr("Wavelength (nm)"), tr("Sensor pitch X (um)"),
                           tr("Sensor pitch Y (um)"), tr("Z minimum (mm)"), tr("Z maximum (mm)")};
  for (size_t i = 0; i < optics_.size(); ++i) {
    auto *spin = optics_[i] = new QDoubleSpinBox(this);
    spin->setDecimals(6);
    spin->setRange(i < 3 ? 0.000001 : -100000, 100000);
    spin->setSingleStep(i < 3 ? 1 : 0.1);
    spin->setKeyboardTracking(false);
    layout->addRow(labels[static_cast<int>(i)], spin);
    connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this] { emit settings_changed(); });
  }
  padding_ = new QCheckBox(tr("2x zero padding (reduces wraparound)"), this);
  isolate_ = new QCheckBox(tr("Isolate off-axis Fourier order"), this);
  layout->addRow(padding_);
  layout->addRow(isolate_);
  carrier_x_ = add_int(tr("Carrier X offset (Fourier pixels)"), -16384, 16384);
  carrier_y_ = add_int(tr("Carrier Y offset (Fourier pixels)"), -16384, 16384);
  radius_    = add_int(tr("Order radius (Fourier pixels)"), 1, 8192);
  for (auto *spin : {carrier_x_, carrier_y_, radius_})
    spin->setEnabled(false);
  isolate_->setToolTip(
      tr("Use only for an isolated off-axis order. Inspect the spectrum first; a smaller radius "
         "removes detail. Offsets use the original image FFT grid."));
  calibration_error_ = new QLabel(this);
  calibration_error_->setWordWrap(true);
  layout->addRow(calibration_error_);
  calibration_error_->hide();
  connect(padding_, &QCheckBox::toggled, this, [this] { emit settings_changed(); });
  connect(isolate_, &QCheckBox::toggled, this, [this](bool enabled) {
    for (auto *spin : {carrier_x_, carrier_y_, radius_})
      spin->setEnabled(enabled);
    emit settings_changed();
  });
  connect(playback_, &QCheckBox::toggled, this, [this] {
    refresh_method();
    emit settings_changed();
  });
  connect(method_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] {
    refresh_method();
    emit settings_changed();
  });
  connect(temporal_filter_, &QCheckBox::toggled, this, [this] {
    refresh_method();
    emit settings_changed();
  });
  set_settings({});
}
holotask::pipelines::HoloFocusSettings FocusProcessingWidget::settings() const {
  holotask::pipelines::HoloFocusSettings result;
  result.method           = method_->currentData().toString().toStdString();
  result.window           = window_->value();
  result.playback         = playback_->isChecked();
  result.playback_fps     = playback_fps_->value();
  result.temporal_focus_m = temporal_focus_->value() * 1e-3;
  result.temporal_pitch_m = temporal_pitch_->value() * 1e-6;
  result.temporal_filter  = temporal_filter_->isChecked();
  result.filter_inner     = filter_inner_->value();
  result.filter_outer     = filter_outer_->value();
  result.frame            = frame_->value();
  auto &s                 = result.optics;
  s.wavelength_m          = optics_[0]->value() * 1e-9;
  s.pixel_pitch_x_m       = optics_[1]->value() * 1e-6;
  s.pixel_pitch_y_m       = optics_[2]->value() * 1e-6;
  s.z_min_m               = optics_[3]->value() * 1e-3;
  s.z_max_m               = optics_[4]->value() * 1e-3;
  s.planes                = planes_->value();
  s.padding               = padding_->isChecked();
  s.isolate_order         = isolate_->isChecked();
  s.carrier_x             = carrier_x_->value();
  s.carrier_y             = carrier_y_->value();
  s.order_radius          = radius_->value();
  return result;
}
void FocusProcessingWidget::set_settings(const holotask::pipelines::HoloFocusSettings &settings) {
  const QSignalBlocker blocker(this);
  method_->setCurrentIndex(std::max(0, method_->findData(QString::fromStdString(settings.method))));
  window_->setValue(settings.window);
  playback_->setChecked(settings.playback);
  playback_fps_->setValue(settings.playback_fps);
  temporal_focus_->setValue(settings.temporal_focus_m * 1e3);
  temporal_pitch_->setValue(settings.temporal_pitch_m * 1e6);
  temporal_filter_->setChecked(settings.temporal_filter);
  filter_inner_->setValue(settings.filter_inner);
  filter_outer_->setValue(settings.filter_outer);
  frame_->setValue(settings.frame);
  const auto      &s = settings.optics;
  const std::array values{s.wavelength_m * 1e9, s.pixel_pitch_x_m * 1e6, s.pixel_pitch_y_m * 1e6,
                          s.z_min_m * 1e3, s.z_max_m * 1e3};
  for (size_t i = 0; i < values.size(); ++i)
    optics_[i]->setValue(values[i]);
  planes_->setValue(s.planes);
  padding_->setChecked(s.padding);
  isolate_->setChecked(s.isolate_order);
  carrier_x_->setValue(s.carrier_x);
  carrier_y_->setValue(s.carrier_y);
  radius_->setValue(s.order_radius);
  refresh_method();
}
void FocusProcessingWidget::load_calibration(const std::filesystem::path &path,
                                             bool                         apply_estimates) {
  const QSignalBlocker blocker(this);
  try {
    const auto c = holotask::pipelines::inspect_holo_calibration(path);
    auto       s = settings();
    if (apply_estimates) {
      if (c.wavelength_m)
        s.optics.wavelength_m = *c.wavelength_m;
      if (c.pixel_pitch_x_m)
        s.optics.pixel_pitch_x_m = *c.pixel_pitch_x_m;
      if (c.pixel_pitch_y_m)
        s.optics.pixel_pitch_y_m = *c.pixel_pitch_y_m;
      if (c.propagation_m) {
        s.optics.z_min_m = *c.propagation_m - 0.005;
        s.optics.z_max_m = *c.propagation_m + 0.005;
      }
      s.frame = 0;
    }
    frame_->setMaximum(c.frames - 1);
    s.frame = std::clamp(s.frame, 0, c.frames - 1);
    set_settings(s);
    calibration_error_->clear();
    calibration_error_->hide();
  } catch (const std::exception &e) {
    calibration_error_->setText(QString::fromUtf8(e.what()));
    calibration_error_->show();
  }
}
void FocusProcessingWidget::set_playback_position(int first, int last, int total) {
  playback_position_->setText(
      tr("Displayed frames %1-%2 / %3 (0-based)").arg(first).arg(last).arg(total));
}
void FocusProcessingWidget::refresh_method() {
  const bool temporal = method_->currentData().toString() == "holovibes_stft";
  auto      *form     = qobject_cast<QFormLayout *>(layout());
  for (auto *widget : std::array<QWidget *, 9>{window_, temporal_focus_, temporal_pitch_,
                                               temporal_filter_, filter_inner_, filter_outer_,
                                               playback_, playback_fps_, playback_position_})
    form->setRowVisible(widget, temporal);
  for (auto *widget :
       std::array<QWidget *, 10>{planes_, optics_[1], optics_[2], optics_[3], optics_[4], padding_,
                                 isolate_, carrier_x_, carrier_y_, radius_})
    form->setRowVisible(widget, !temporal);
  frame_label_->setText(temporal ? tr("Window start frame (0-based)") : tr("Time frame (0-based)"));
  playback_fps_->setEnabled(playback_->isChecked());
  filter_inner_->setEnabled(temporal_filter_->isChecked());
  filter_outer_->setEnabled(temporal_filter_->isChecked());
}
} // namespace holovibes::ui
