// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "mesi_dashboard_widget.hh"

#include <QColor>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedLayout>
#include <QStringList>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <functional>
#include <mutex>
#include <utility>

#include "logger.hh"
#include "mesi_exposure_plot_widget.hh"
#include "tensor_display_widget.hh"

namespace holovibes::ui {
namespace {

QString tau_label(double seconds) {
  if (seconds < 1.0e-3)
    return QStringLiteral("%1 µs").arg(QString::number(seconds * 1.0e6, 'g', 3));
  if (seconds < 1.0)
    return QStringLiteral("%1 ms").arg(QString::number(seconds * 1.0e3, 'g', 3));
  return QStringLiteral("%1 s").arg(QString::number(seconds, 'g', 3));
}

} // namespace

class MesiDashboardWidget::RoiOverlay : public QWidget {
public:
  explicit RoiOverlay(QWidget *parent = nullptr) : QWidget(parent) {
    setAttribute(Qt::WA_TranslucentBackground);
    setMouseTracking(true);
    setCursor(Qt::CrossCursor);
  }

  void set_image_size(int width, int height) {
    std::lock_guard lock(mutex_);
    image_width_  = std::max(width, 0);
    image_height_ = std::max(height, 0);
  }

  QRect selected_roi() const {
    std::lock_guard lock(mutex_);
    return roi_;
  }

  void select_roi(const QRect &roi) {
    {
      std::lock_guard lock(mutex_);
      roi_ = roi;
    }
    if (roi_callback_)
      roi_callback_(roi);
    update();
  }

  void set_roi_callback(std::function<void(const QRect &)> callback) {
    roi_callback_ = std::move(callback);
  }

  QPoint hovered_pixel() const {
    std::lock_guard lock(mutex_);
    return hovered_pixel_;
  }

  QPoint pixel_to_sample() const {
    std::lock_guard lock(mutex_);
    return clicked_sample_pixel_.x() >= 0 ? clicked_sample_pixel_ : hovered_pixel_;
  }

  void request_click_sample(const QPoint &pixel) {
    std::lock_guard lock(mutex_);
    clicked_sample_pixel_ = pixel;
  }

  void clear_click_sample(const QPoint &pixel) {
    std::lock_guard lock(mutex_);
    if (clicked_sample_pixel_ == pixel)
      clicked_sample_pixel_ = QPoint(-1, -1);
  }

  void set_hover_callback(std::function<void(const QPoint &)> callback) {
    hover_callback_ = std::move(callback);
  }

  void set_click_callback(std::function<void(const QPoint &, const QPointF &)> callback) {
    click_callback_ = std::move(callback);
  }

  void watch_image(QWidget *image) {
    image->setMouseTracking(true);
    image->installEventFilter(this);
    watched_image_ = image;
  }

  void clear_roi() {
    {
      std::lock_guard lock(mutex_);
      roi_ = {};
    }
    dragging_ = false;
    if (roi_callback_)
      roi_callback_({});
    update();
  }

protected:
  bool eventFilter(QObject *watched, QEvent *event) override {
    if (watched != watched_image_)
      return QWidget::eventFilter(watched, event);
    if (event->type() == QEvent::Leave) {
      leaveEvent(event);
      return false;
    }
    if (event->type() != QEvent::MouseMove && event->type() != QEvent::MouseButtonPress &&
        event->type() != QEvent::MouseButtonRelease)
      return QWidget::eventFilter(watched, event);
    auto       *mouse = static_cast<QMouseEvent *>(event);
    QMouseEvent forwarded(mouse->type(), mapFrom(watched_image_, mouse->position()),
                          mouse->globalPosition(), mouse->button(), mouse->buttons(),
                          mouse->modifiers());
    if (event->type() == QEvent::MouseMove)
      mouseMoveEvent(&forwarded);
    else if (event->type() == QEvent::MouseButtonPress)
      mousePressEvent(&forwarded);
    else
      mouseReleaseEvent(&forwarded);
    return forwarded.isAccepted();
  }

  void mousePressEvent(QMouseEvent *event) override {
    if (event->button() != Qt::LeftButton || image_width_ <= 0 || image_height_ <= 0 ||
        !QRectF(image_rect()).contains(event->position())) {
      event->ignore();
      return;
    }
    dragging_   = true;
    drag_start_ = image_point(event->position());
    drag_end_   = drag_start_;
    update();
    event->accept();
  }

  void mouseMoveEvent(QMouseEvent *event) override {
    if (!dragging_) {
      const QPoint pixel = QRectF(image_rect()).contains(event->position())
                               ? image_point(event->position())
                               : QPoint(-1, -1);
      {
        std::lock_guard lock(mutex_);
        hovered_pixel_ = pixel;
      }
      if (hover_callback_)
        hover_callback_(pixel);
      event->accept();
      return;
    }
    drag_end_ = image_point(event->position());
    update();
    event->accept();
  }

  void leaveEvent(QEvent *event) override {
    {
      std::lock_guard lock(mutex_);
      hovered_pixel_ = QPoint(-1, -1);
    }
    if (hover_callback_)
      hover_callback_(QPoint(-1, -1));
    QWidget::leaveEvent(event);
  }

  void mouseReleaseEvent(QMouseEvent *event) override {
    if (!dragging_ || event->button() != Qt::LeftButton) {
      event->ignore();
      return;
    }
    drag_end_ = image_point(event->position());
    if (drag_start_ == drag_end_) {
      if (click_callback_)
        click_callback_(drag_end_, event->position());
    }
    const int left   = std::min(drag_start_.x(), drag_end_.x());
    const int top    = std::min(drag_start_.y(), drag_end_.y());
    const int right  = std::max(drag_start_.x(), drag_end_.x());
    const int bottom = std::max(drag_start_.y(), drag_end_.y());
    {
      std::lock_guard lock(mutex_);
      roi_ = QRect(left, top, right - left + 1, bottom - top + 1);
    }
    dragging_ = false;
    if (roi_callback_)
      roi_callback_(selected_roi());
    update();
    event->accept();
  }

  void paintEvent(QPaintEvent *) override {
    QRect roi = selected_roi();
    if (dragging_) {
      const int left = std::min(drag_start_.x(), drag_end_.x());
      const int top  = std::min(drag_start_.y(), drag_end_.y());
      roi            = QRect(left, top, std::abs(drag_start_.x() - drag_end_.x()) + 1,
                             std::abs(drag_start_.y() - drag_end_.y()) + 1);
    }
    if (roi.isEmpty() || image_width_ <= 0 || image_height_ <= 0)
      return;
    const QRect display_rect = image_rect();
    const int   x = display_rect.left() + roi.left() * display_rect.width() / image_width_;
    const int   y = display_rect.top() + display_rect.height() -
                    (roi.bottom() + 1) * display_rect.height() / image_height_;
    const int   w = std::max(1, roi.width() * display_rect.width() / image_width_);
    const int   h = std::max(1, roi.height() * display_rect.height() / image_height_);
    QPainter    painter(this);
    painter.setPen(QPen(QColor(0, 0, 0), 4));
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(QRect(x, y, w, h));
    painter.setPen(QPen(QColor(255, 220, 0), 2));
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(QRect(x, y, w, h));
  }

private:
  QRect image_rect() const {
    if (image_width_ <= 0 || image_height_ <= 0 || width() <= 0 || height() <= 0)
      return {};
    const double aspect        = static_cast<double>(image_width_) / image_height_;
    const double widget_aspect = static_cast<double>(width()) / height();
    if (widget_aspect > aspect) {
      const int image_w = static_cast<int>(height() * aspect);
      return {(width() - image_w) / 2, 0, image_w, height()};
    }
    const int image_h = static_cast<int>(width() / aspect);
    return {0, (height() - image_h) / 2, width(), image_h};
  }

  QPoint image_point(const QPointF &point) const {
    if (image_width_ <= 0 || image_height_ <= 0 || width() <= 0 || height() <= 0)
      return {};
    const QRect display_rect = image_rect();
    const int   x = std::clamp(static_cast<int>((point.x() - display_rect.left()) * image_width_ /
                                                std::max(display_rect.width(), 1)),
                               0, image_width_ - 1);
    const int   y =
        std::clamp(static_cast<int>((display_rect.top() + display_rect.height() - point.y()) *
                                    image_height_ / std::max(display_rect.height(), 1)),
                   0, image_height_ - 1);
    return {x, y};
  }

  mutable std::mutex                                   mutex_;
  int                                                  image_width_  = 0;
  int                                                  image_height_ = 0;
  QRect                                                roi_;
  QPoint                                               hovered_pixel_{-1, -1};
  QPoint                                               clicked_sample_pixel_{-1, -1};
  QPoint                                               drag_start_;
  QPoint                                               drag_end_;
  bool                                                 dragging_      = false;
  QWidget                                             *watched_image_ = nullptr;
  std::function<void(const QRect &)>                   roi_callback_;
  std::function<void(const QPoint &)>                  hover_callback_;
  std::function<void(const QPoint &, const QPointF &)> click_callback_;
};

MesiDashboardWidget::MesiDashboardWidget(QWidget *parent) : QWidget(parent) {
  auto *splitter   = new QSplitter(Qt::Horizontal, this);
  auto *map_panel  = new QWidget(splitter);
  auto *map_layout = new QVBoxLayout(map_panel);
  map_layout->setContentsMargins(4, 4, 4, 4);
  map_label_    = new QLabel(tr("Live τ map — drag for ROI"), map_panel);
  map_selector_ = new QComboBox(map_panel);
  map_selector_->addItem(tr("τ map"), 0);
  map_selector_->setToolTip(tr("Choose the live τ map or a pooled K² exposure map."));
  auto *reset_roi_button = new QPushButton(tr("Full image ROI"), map_panel);
  auto *map_toolbar      = new QHBoxLayout();
  map_toolbar->setContentsMargins(0, 0, 0, 0);
  map_toolbar->addWidget(map_label_, 1);
  map_toolbar->addWidget(map_selector_);
  map_toolbar->addWidget(reset_roi_button);
  tau_map_widget_ = new TensorDisplayWidget(map_panel);
  tau_map_widget_->set_colormap(Colormap::Grayscale);
  tau_map_widget_->set_auto_value_range(true);
  tau_map_widget_->set_nonpositive_invalid(true);
  auto *map_stack    = new QWidget(map_panel);
  auto *stack_layout = new QStackedLayout(map_stack);
  stack_layout->setStackingMode(QStackedLayout::StackAll);
  tau_map_widget_->setParent(map_stack);
  roi_overlay_ = new RoiOverlay(map_stack);
  stack_layout->addWidget(tau_map_widget_);
  stack_layout->addWidget(roi_overlay_);
  stack_layout->setCurrentWidget(roi_overlay_);
  roi_overlay_->raise();
  roi_overlay_->watch_image(tau_map_widget_);
  map_layout->addLayout(map_toolbar);
  auto *tau_legend   = new QHBoxLayout();
  tau_low_label_     = new QLabel(map_panel);
  tau_high_label_    = new QLabel(map_panel);
  auto *tau_gradient = new QLabel(map_panel);
  tau_gradient->setFixedHeight(12);
  tau_gradient->setMinimumWidth(80);
  tau_gradient->setStyleSheet(QStringLiteral(
      "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 black, stop:1 white);"));
  tau_legend->addWidget(tau_low_label_);
  tau_legend->addWidget(tau_gradient, 1);
  tau_legend->addWidget(tau_high_label_);
  map_help_ = new QLabel(
      tr("Black: τ at or below the left value. White: τ at or above the right value. "
         "Gray: between them. Magenta: no valid τ estimate (zero or nonfinite). "
         "The display range adjusts to each live image; τ is in seconds, not a direct flow speed."),
      map_panel);
  map_help_->setWordWrap(true);
  pixel_label_ = new QLabel(tr("Hover for K², τ, β and ρ; click a pixel to log it."), map_panel);
  pixel_label_->setWordWrap(true);
  map_layout->addWidget(pixel_label_);
  map_layout->addWidget(map_stack, 1);
  map_layout->addLayout(tau_legend);
  map_layout->addWidget(map_help_);
  update_tau_legend();
  connect(tau_map_widget_, &TensorDisplayWidget::tensorDisplayed, this,
          &MesiDashboardWidget::update_tau_legend);
  connect(map_selector_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
    const int exposure = map_selector_->itemData(index).toInt();
    set_display_exposure_frames(exposure);
    emit displayExposureChanged(exposure);
  });
  auto *curve_panel = new QWidget(splitter);
  splitter->addWidget(map_panel);
  splitter->addWidget(curve_panel);
  splitter->setStretchFactor(0, 1);
  splitter->setStretchFactor(1, 2);

  auto *controls = new QHBoxLayout();
  controls->setContentsMargins(8, 6, 8, 0);
  roi_label_ = new QLabel(tr("ROI: full image"), curve_panel);
  controls->addWidget(roi_label_, 1);
  const auto add_coordinate = [controls, curve_panel](const QString &label, QSpinBox *&spin) {
    controls->addWidget(new QLabel(label, curve_panel));
    spin = new QSpinBox(curve_panel);
    spin->setRange(0, 1);
    controls->addWidget(spin);
  };
  add_coordinate(tr("x"), roi_x_);
  add_coordinate(tr("y"), roi_y_);
  add_coordinate(tr("width"), roi_width_);
  add_coordinate(tr("height"), roi_height_);
  auto *full_image_button = new QPushButton(tr("Full image"), curve_panel);
  controls->addWidget(full_image_button);
  hold_curve_ = new QCheckBox(tr("Hold curve"), curve_panel);
  controls->addWidget(hold_curve_);

  auto *packet_controls = new QHBoxLayout();
  packet_controls->setContentsMargins(8, 0, 8, 0);
  packet_label_ = new QLabel(tr("No frame packets yet"), curve_panel);
  packet_controls->addWidget(packet_label_);
  packet_slider_ = new QSlider(Qt::Horizontal, curve_panel);
  packet_slider_->setRange(0, 0);
  packet_slider_->setEnabled(false);
  packet_slider_->setToolTip(tr("Browse stored packet curves for this ROI. Imported files keep one "
                                "curve per packet position."));
  packet_controls->addWidget(packet_slider_, 1);

  exposure_plot_widget_ = new MesiExposurePlotWidget(curve_panel);
  auto *curve_layout    = new QVBoxLayout(curve_panel);
  curve_layout->setContentsMargins(0, 0, 0, 0);
  curve_layout->addLayout(controls);
  curve_layout->addLayout(packet_controls);
  curve_layout->addWidget(exposure_plot_widget_, 1);
  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->addWidget(splitter);
  connect(exposure_plot_widget_, &MesiExposurePlotWidget::samplesDisplayed, this,
          &MesiDashboardWidget::samplesDisplayed);
  connect(hold_curve_, &QCheckBox::toggled, this, [this](bool held) {
    if (!held && !curve_samples_.empty()) {
      const QSignalBlocker blocker(packet_slider_);
      packet_slider_->setValue(static_cast<int>(curve_samples_.size() - 1));
    }
    show_selected_curve();
  });
  connect(packet_slider_, &QSlider::valueChanged, this, [this](int index) {
    if (curve_samples_.empty())
      return;
    const QSignalBlocker blocker(hold_curve_);
    hold_curve_->setChecked(index != packet_slider_->maximum());
    show_selected_curve();
  });
  connect(reset_roi_button, &QPushButton::clicked, roi_overlay_, &RoiOverlay::clear_roi);
  connect(full_image_button, &QPushButton::clicked, roi_overlay_, &RoiOverlay::clear_roi);
  roi_overlay_->set_roi_callback([this](const QRect &roi) {
    ++curve_cache_revision_;
    reset_curve();
    if (roi.isEmpty()) {
      const QSignalBlocker block_x(roi_x_);
      const QSignalBlocker block_y(roi_y_);
      const QSignalBlocker block_width(roi_width_);
      const QSignalBlocker block_height(roi_height_);
      roi_x_->setValue(0);
      roi_y_->setValue(0);
      roi_width_->setMaximum(std::max(1, roi_x_->maximum() + 1));
      roi_height_->setMaximum(std::max(1, roi_y_->maximum() + 1));
      roi_width_->setValue(std::max(1, roi_x_->maximum() + 1));
      roi_height_->setValue(std::max(1, roi_y_->maximum() + 1));
    }
    roi_label_->setText(roi.isEmpty() ? tr("ROI: full image")
                                      : tr("ROI: x=%1, y=%2, %3 × %4 px")
                                            .arg(roi.x())
                                            .arg(roi.y())
                                            .arg(roi.width())
                                            .arg(roi.height()));
  });
  roi_overlay_->set_hover_callback([this](const QPoint &pixel) {
    pixel_label_->setText(
        pixel.x() < 0
            ? tr("Hover for K², τ, β and ρ; click a pixel to log it.")
            : tr("Pixel (%1, %2): waiting for the next packet…").arg(pixel.x()).arg(pixel.y()));
  });
  roi_overlay_->set_click_callback([this](const QPoint &pixel, const QPointF &position) {
    pending_click_pixel_    = pixel;
    pending_click_position_ = position;
    roi_overlay_->request_click_sample(pixel);
    pixel_label_->setText(
        tr("Pixel (%1, %2): logging the next displayed packet…").arg(pixel.x()).arg(pixel.y()));
  });
  const auto update_roi = [this]() {
    const QSignalBlocker block_width(roi_width_);
    const QSignalBlocker block_height(roi_height_);
    roi_width_->setMaximum(std::max(1, roi_x_->maximum() + 1 - roi_x_->value()));
    roi_height_->setMaximum(std::max(1, roi_y_->maximum() + 1 - roi_y_->value()));
    roi_overlay_->select_roi(
        QRect(roi_x_->value(), roi_y_->value(), roi_width_->value(), roi_height_->value()));
  };
  connect(roi_x_, qOverload<int>(&QSpinBox::valueChanged), this, update_roi);
  connect(roi_y_, qOverload<int>(&QSpinBox::valueChanged), this, update_roi);
  connect(roi_width_, qOverload<int>(&QSpinBox::valueChanged), this, update_roi);
  connect(roi_height_, qOverload<int>(&QSpinBox::valueChanged), this, update_roi);
}

TensorDisplayWidget    *MesiDashboardWidget::tau_map_widget() const { return tau_map_widget_; }
MesiExposurePlotWidget *MesiDashboardWidget::exposure_plot_widget() const {
  return exposure_plot_widget_;
}
QRect MesiDashboardWidget::selected_roi() const { return roi_overlay_->selected_roi(); }
void  MesiDashboardWidget::set_map_size(int width, int height) {
  if (map_width_ == width && map_height_ == height)
    return;
  map_width_               = width;
  map_height_              = height;
  const QRect previous_roi = roi_overlay_->selected_roi();
  roi_overlay_->set_image_size(width, height);
  const QSignalBlocker block_x(roi_x_);
  const QSignalBlocker block_y(roi_y_);
  const QSignalBlocker block_width(roi_width_);
  const QSignalBlocker block_height(roi_height_);
  roi_x_->setRange(0, std::max(0, width - 1));
  roi_y_->setRange(0, std::max(0, height - 1));
  roi_width_->setRange(1, std::max(1, width));
  roi_height_->setRange(1, std::max(1, height));
  if (previous_roi.isEmpty()) {
    roi_overlay_->clear_roi();
  } else {
    const QRect bounded = previous_roi.intersected(QRect(0, 0, width, height));
    if (bounded.isEmpty()) {
      roi_overlay_->clear_roi();
    } else {
      roi_x_->setValue(bounded.x());
      roi_y_->setValue(bounded.y());
      roi_width_->setValue(bounded.width());
      roi_height_->setValue(bounded.height());
      roi_overlay_->select_roi(bounded);
    }
  }
}
void MesiDashboardWidget::set_tau_range(float minimum_seconds, float maximum_seconds) {
  tau_map_widget_->set_value_range(display_exposure_frames() == 0 ? minimum_seconds : 0.0F,
                                   display_exposure_frames() == 0 ? maximum_seconds : 1.0F);
  tau_map_widget_->set_auto_value_range(true);
  update_tau_legend();
}
int MesiDashboardWidget::display_exposure_frames() const {
  return requested_display_exposure_frames_;
}
void MesiDashboardWidget::set_display_exposure_frames(int frames) {
  requested_display_exposure_frames_ = frames;
  const int index                    = map_selector_->findData(frames);
  if (index < 0)
    return;
  const QSignalBlocker blocker(map_selector_);
  map_selector_->setCurrentIndex(index);
  map_label_->setText(
      frames == 0 ? tr("Live τ map — drag for ROI")
                  : tr("Live pooled K² map · %1-frame exposure — drag for ROI").arg(frames));
  map_help_->setText(
      frames == 0
          ? tr("Black to white: increasing τ. Magenta: zero or nonfinite τ. "
               "The grayscale range adjusts to each image; τ is not a direct flow speed.")
          : tr("Black to white: increasing pooled K² (unitless). Magenta: K² at or below zero "
               "or nonfinite. The range adjusts to each image."));
  if (frames != 0) {
    tau_map_widget_->set_value_range(0.0F, 1.0F);
    tau_map_widget_->set_auto_value_range(true);
  }
  update_tau_legend();
}
void MesiDashboardWidget::set_exposure_levels(const std::vector<int> &frames) {
  const int previous = display_exposure_frames();
  exposure_levels_   = frames;
  const QSignalBlocker blocker(map_selector_);
  map_selector_->clear();
  map_selector_->addItem(tr("τ map"), 0);
  for (const int count : frames)
    map_selector_->addItem(tr("K² · %1 frames").arg(count), count);
  set_display_exposure_frames(map_selector_->findData(previous) >= 0 ? previous : 0);
}
void MesiDashboardWidget::set_display_selector_enabled(bool enabled) {
  map_selector_->setEnabled(enabled);
}
QPoint MesiDashboardWidget::hovered_pixel() const { return roi_overlay_->hovered_pixel(); }
QPoint MesiDashboardWidget::pixel_to_sample() const { return roi_overlay_->pixel_to_sample(); }
void   MesiDashboardWidget::set_pixel_values(const MesiPixelValues &values) {
  if (values.pixel == pending_click_pixel_) {
    log_clicked_pixel(values);
    pending_click_pixel_ = QPoint(-1, -1);
    roi_overlay_->clear_click_sample(values.pixel);
  }
  if (values.pixel != hovered_pixel())
    return;
  const auto number = [](float value) {
    return std::isfinite(value) ? QString::number(value, 'g', 5) : QStringLiteral("invalid");
  };
  QStringList exposures;
  for (size_t i = 0; i < values.k2.size() && i < exposure_levels_.size(); ++i)
    exposures << tr("%1:%2").arg(exposure_levels_[i]).arg(number(values.k2[i]));
  const QString map_value = display_exposure_frames() == 0
                                ? (values.map_value > 0.0F && std::isfinite(values.map_value)
                                       ? tau_label(values.map_value)
                                       : tr("invalid"))
                                : number(values.map_value);
  pixel_label_->setText(tr("Packet %1 · pixel (%2, %3) · K² [frames:value] %4 · τ %5 · β %6 · ρ %7 "
                           "· displayed map pixel %8")
                            .arg(QString::number(values.packet_index + 1))
                            .arg(values.pixel.x())
                            .arg(values.pixel.y())
                            .arg(exposures.join(QStringLiteral("  ")))
                            .arg(values.tau_seconds > 0.0F && std::isfinite(values.tau_seconds)
                                     ? tau_label(values.tau_seconds)
                                     : tr("invalid"))
                            .arg(number(values.beta))
                            .arg(number(values.rho))
                            .arg(map_value));
}
void MesiDashboardWidget::log_clicked_pixel(const MesiPixelValues &values) {
  const QImage image = tau_map_widget_->grabFramebuffer();
  QString      color = QStringLiteral("unavailable");
  int          red   = -1;
  int          green = -1;
  int          blue  = -1;
  if (!image.isNull() && tau_map_widget_->width() > 0 && tau_map_widget_->height() > 0) {
    const QPointF widget_position = tau_map_widget_->mapFrom(roi_overlay_, pending_click_position_);
    const int     x =
        std::clamp(static_cast<int>(widget_position.x() * image.width() / tau_map_widget_->width()),
                   0, image.width() - 1);
    const int y = std::clamp(
        static_cast<int>(widget_position.y() * image.height() / tau_map_widget_->height()), 0,
        image.height() - 1);
    const QColor sampled = image.pixelColor(x, y);
    color                = sampled.name(QColor::HexRgb);
    red                  = sampled.red();
    green                = sampled.green();
    blue                 = sampled.blue();
  }
  QStringList exposures;
  for (size_t i = 0; i < values.k2.size() && i < exposure_levels_.size(); ++i)
    exposures << QStringLiteral("%1:%2")
                     .arg(exposure_levels_[i])
                     .arg(QString::number(values.k2[i], 'g', 9));
  const auto [minimum, maximum] = tau_map_widget_->display_range();
  const char *expected =
      std::isfinite(values.map_value) && values.map_value > 0.0F ? "grayscale" : "magenta";
  logger()->info("[MESI pixel click] packet={} pixel=({}, {}) map_exposure_frames={} "
                 "rendered_color={} rgb=({},{},{}) expected_color={} map_value={} "
                 "display_range=({}, {}) tau_seconds={} tau_microseconds={} beta={} rho={} K2={}",
                 values.packet_index + 1, values.pixel.x(), values.pixel.y(),
                 display_exposure_frames(), color.toStdString(), red, green, blue, expected,
                 values.map_value, minimum, maximum, values.tau_seconds, values.tau_seconds * 1.0e6,
                 values.beta, values.rho, exposures.join(QStringLiteral(", ")).toStdString());
}
std::vector<uint64_t> MesiDashboardWidget::prepare_curve_cache(const std::string &key,
                                                               size_t             packet_count) {
  if (key != curve_cache_key_ || packet_count != source_packet_count_) {
    curve_cache_key_     = key;
    source_packet_count_ = packet_count;
    reset_curve();
  }
  std::vector<uint64_t> cached;
  cached.reserve(curve_samples_.size());
  for (const auto &sample : curve_samples_)
    cached.push_back(sample.window_index);
  if (!curve_samples_.empty())
    show_selected_curve();
  return cached;
}
void MesiDashboardWidget::update_tau_legend() {
  const auto [minimum, maximum] = tau_map_widget_->display_range();
  tau_low_label_->setText(display_exposure_frames() == 0 ? tau_label(minimum)
                                                         : QString::number(minimum, 'g', 3));
  tau_high_label_->setText(display_exposure_frames() == 0 ? tau_label(maximum)
                                                          : QString::number(maximum, 'g', 3));
}
void MesiDashboardWidget::reset_curve() {
  ++curve_cache_revision_;
  curve_samples_.clear();
  const QSignalBlocker block_hold(hold_curve_);
  const QSignalBlocker block_slider(packet_slider_);
  hold_curve_->setChecked(false);
  packet_slider_->setRange(0, 0);
  packet_slider_->setEnabled(false);
  packet_label_->setText(tr("No frame packets yet"));
  exposure_plot_widget_->reset();
}

void MesiDashboardWidget::append_curve_sample(MesiExposureSample sample) {
  if (source_packet_count_ != 0 &&
      std::any_of(curve_samples_.begin(), curve_samples_.end(), [&sample](const auto &stored) {
        return stored.window_index == sample.window_index;
      }))
    return;
  constexpr size_t max_packets = 10000;
  int              selected    = packet_slider_->value();
  const uint64_t   held_packet = hold_curve_->isChecked() && !curve_samples_.empty()
                                     ? curve_samples_[static_cast<size_t>(selected)].window_index
                                     : 0;
  if (source_packet_count_ == 0 && curve_samples_.size() == max_packets) {
    curve_samples_.pop_front();
    selected = std::max(0, selected - 1);
  }
  curve_samples_.push_back(std::move(sample));
  if (source_packet_count_ != 0)
    std::sort(curve_samples_.begin(), curve_samples_.end(),
              [](const auto &a, const auto &b) { return a.window_index < b.window_index; });
  if (hold_curve_->isChecked() && source_packet_count_ != 0) {
    const auto found = std::find_if(
        curve_samples_.begin(), curve_samples_.end(),
        [held_packet](const auto &stored) { return stored.window_index == held_packet; });
    selected = static_cast<int>(found - curve_samples_.begin());
  }
  {
    const QSignalBlocker blocker(packet_slider_);
    packet_slider_->setRange(0, static_cast<int>(curve_samples_.size() - 1));
    packet_slider_->setEnabled(curve_samples_.size() > 1);
    packet_slider_->setValue(
        hold_curve_->isChecked() ? selected : static_cast<int>(curve_samples_.size() - 1));
  }
  show_selected_curve();
}

void MesiDashboardWidget::show_selected_curve() {
  if (curve_samples_.empty())
    return;
  const int   selected = packet_slider_->value();
  const auto &sample   = curve_samples_[static_cast<size_t>(selected)];
  packet_label_->setText(tr("%1 · packet %2 · stored %3 / %4")
                             .arg(hold_curve_->isChecked() ? tr("Held") : tr("Live"))
                             .arg(QString::number(sample.window_index + 1))
                             .arg(selected + 1)
                             .arg(QString::number(static_cast<qulonglong>(curve_samples_.size()))));
  exposure_plot_widget_->set_sample(sample);
}

} // namespace holovibes::ui
