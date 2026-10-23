// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0
#include "ui/widgets/slice_navigation_overlay.hh"
#include <QPainter>
#include <algorithm>
namespace holovibes::ui {
SliceNavigationOverlay::SliceNavigationOverlay(QWidget *parent) : QWidget(parent) {
  setAttribute(Qt::WA_TransparentForMouseEvents);
  setAttribute(Qt::WA_NoSystemBackground);
  setFocusPolicy(Qt::NoFocus);
  hide();
}
void SliceNavigationOverlay::set_state(const QRect &image, const std::vector<QRectF> &bands,
                                       const QString &hint) {
  if (image_ == image && bands_ == bands && hint_ == hint)
    return;
  image_ = image;
  bands_ = bands;
  hint_  = hint;
  update();
}
void SliceNavigationOverlay::paintEvent(QPaintEvent *) {
  QPainter painter(this);
  painter.setClipRect(image_);
  painter.setPen(QPen(QColor(255, 65, 65, 225), 1));
  painter.setBrush(QColor(240, 40, 40, 45));
  for (const auto &band : bands_) {
    QRectF pixels(image_.x() + band.x() * image_.width(), image_.y() + band.y() * image_.height(),
                  std::max(1.0, band.width() * image_.width()),
                  std::max(1.0, band.height() * image_.height()));
    painter.drawRect(pixels);
  }
  auto font = painter.font();
  font.setPointSize(9);
  painter.setFont(font);
  const QRect area = image_.adjusted(6, 6, -6, -6);
  const QRect text =
      painter.boundingRect(area, Qt::AlignLeft | Qt::AlignBottom | Qt::TextWordWrap, hint_);
  painter.fillRect(text.adjusted(-4, -2, 4, 2), QColor(0, 0, 0, 185));
  painter.setPen(Qt::white);
  painter.drawText(area, Qt::AlignLeft | Qt::AlignBottom | Qt::TextWordWrap, hint_);
}
} // namespace holovibes::ui
