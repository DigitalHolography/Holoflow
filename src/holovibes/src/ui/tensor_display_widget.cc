#include "tensor_display_widget.hh"

#include <QDebug>
#include <QMetaType>
#include <QPaintEvent>
#include <QPainter>
#include <QStyleOption>
#include <cstring>

#include "logger.hh"

namespace holovibes::ui {

TensorDisplayWidget::TensorDisplayWidget(QWidget *parent) : QWidget(parent) {
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  setMinimumSize(160, 120);

  qRegisterMetaType<QImage>("QImage");
  qRegisterMetaType<QByteArray>("QByteArray");
}

QSize TensorDisplayWidget::sizeHint() const { return {320, 240}; }

void TensorDisplayWidget::presentTensor(const QByteArray &bytes, int width, int height) {
  logger()->debug("TensorDisplayWidget received tensor {}x{} ({} bytes)", width, height,
                  bytes.size());
  if (width <= 0 || height <= 0) {
    logger()->warn("TensorDisplayWidget received non-positive dimensions {} {}", width, height);
    return;
  }

  const qsizetype expected = static_cast<qsizetype>(width) * static_cast<qsizetype>(height);
  if (bytes.size() < expected) {
    logger()->warn("TensorDisplayWidget received insufficient data {} expected {}", bytes.size(),
                   expected);
    return;
  }

  QImage               image(width, height, QImage::Format_Grayscale8);
  const unsigned char *src    = reinterpret_cast<const unsigned char *>(bytes.constData());
  const int            stride = image.bytesPerLine();

  for (int row = 0; row < height; ++row) {
    auto *dst = image.scanLine(row);
    std::memcpy(dst, src + static_cast<qsizetype>(row) * width, width);
    if (stride > width) {
      std::memset(dst + width, 0, stride - width);
    }
  }

  updatePixmap(image);
  logger()->debug("TensorDisplayWidget updated pixmap {}x{}", width, height);
  emit tensorDisplayed();
}

void TensorDisplayWidget::presentImage(const QImage &image) {
  if (image.isNull()) {
    logger()->warn("TensorDisplayWidget received null image");
    return;
  }

  updatePixmap(image);
  emit tensorDisplayed();
}

void TensorDisplayWidget::paintEvent(QPaintEvent *event) {
  Q_UNUSED(event);

  QPainter painter(this);
  painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

  QStyleOption opt;
  opt.initFrom(this);
  style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);

  if (pixmap_.isNull()) {
    return;
  }

  const QRect   target_rect = rect();
  const QPixmap scaled =
      pixmap_.scaled(target_rect.size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);

  const int x = target_rect.center().x() - scaled.width() / 2;
  const int y = target_rect.center().y() - scaled.height() / 2;
  painter.drawPixmap(x, y, scaled);
}

void TensorDisplayWidget::updatePixmap(const QImage &image) {
  pixmap_ = QPixmap::fromImage(image, Qt::NoFormatConversion);
  update();
}

} // namespace holovibes::ui
