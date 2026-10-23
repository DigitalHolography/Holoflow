// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <QRectF>
#include <QWidget>
#include <vector>
namespace holovibes::ui {
/// Transparent child layer: painting guides never changes the image's OpenGL state.
class SliceNavigationOverlay : public QWidget {
public:
  explicit SliceNavigationOverlay(QWidget *parent);
  void set_state(const QRect &image, const std::vector<QRectF> &bands, const QString &hint);

protected:
  void paintEvent(QPaintEvent *) override;

private:
  QRect               image_;
  std::vector<QRectF> bands_;
  QString             hint_;
};
} // namespace holovibes::ui
