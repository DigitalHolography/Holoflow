#include "holovibes/ui/tensor_display_widget.hh"

#include <QCoreApplication>
#include <QImage>
#include <QLabel>
#include <QPainter>
#include <QPixmap>
#include <QPoint>
#include <QWidget>
#include <cassert>
#include <spdlog/spdlog.h>

#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     TensorDisplayWidget Implementation
// ==========================================================================

TensorDisplayWidget::TensorDisplayWidget(int width, int height, QWidget *parent)
    : QMainWindow(parent), width_(width), height_(height),
      image_(new QLabel(this)), display_reticle_(false), reticle_radius_(1.0) {
  image_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  image_->setAlignment(Qt::AlignCenter);
  image_->setScaledContents(true);
  image_->resize(width_, height_);
  setCentralWidget(image_);
}

void TensorDisplayWidget::set_display_reticle(bool on) {
  display_reticle_ = on;
}

void TensorDisplayWidget::set_reticle_radius(double r) { reticle_radius_ = r; }

void TensorDisplayWidget::show_tensor(TensorView tens) {
  holovibes_logger()->trace("showing tensor");

  if (tens.data_type() != DataType::U8) {
    holovibes_logger()->warn("Tensor data type is not U8.");
    emit frame_displayed();
    return;
  }

  if (tens.memory_location() != MemoryLocation::HOST) {
    holovibes_logger()->warn("Tensor memory location is not HOST.");
    emit frame_displayed();
    return;
  }

  if (tens.shape().size() != 2) {
    holovibes_logger()->warn("Tensor shape size is not 2.");
    emit frame_displayed();
    return;
  }

  if (tens.strides().at(1) != 1) {
    holovibes_logger()->warn("Tensor stride at index 1 is not 1.");
    emit frame_displayed();
    return;
  }

  if (tens.strides().at(0) != tens.shape().at(1)) {
    holovibes_logger()->warn(
        "Tensor stride at index 0 does not match tensor shape at index 1.");
    emit frame_displayed();
    return;
  }

  uint8_t *tens_data = static_cast<uint8_t *>(tens.data());
  int tens_width = static_cast<int>(tens.shape().at(1));
  int tens_height = static_cast<int>(tens.shape().at(0));
  QImage image(tens_data, tens_width, tens_height, QImage::Format_Grayscale8);
  QPixmap pixmap = QPixmap::fromImage(image);

  if (display_reticle_ && reticle_radius_ > 0.0) {
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    QPen pen(Qt::red);
    pen.setWidth(2);
    painter.setPen(pen);

    const int w = pixmap.width();
    const int h = pixmap.height();

    const double sx = static_cast<double>(width_) / w;
    const double sy = static_cast<double>(height_) / h;

    const double r_screen = 0.5 * std::min(width_, height_) * reticle_radius_;
    const int rx = static_cast<int>(r_screen / sx + 0.5);
    const int ry = static_cast<int>(r_screen / sy + 0.5);

    painter.drawEllipse(QPoint(w / 2, h / 2), rx, ry);
  }

  QPixmap scaledPixmap = pixmap.scaled(width_, height_, Qt::IgnoreAspectRatio,
                                       Qt::SmoothTransformation);

  // holovibes_logger()->info("frame displayed");
  image_->setPixmap(scaledPixmap);
  emit frame_displayed();
}

} // namespace dh
