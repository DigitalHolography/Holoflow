#include "holovibes/ui/tensor_display_widget.hh"

#include <QImage>
#include <QLabel>
#include <QPixmap>
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
      image_(new QLabel(this)) {
  image_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  image_->setAlignment(Qt::AlignCenter);
  image_->setScaledContents(true);
  image_->resize(width_, height_);
  setCentralWidget(image_);
}

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
  QPixmap scaledPixmap = pixmap.scaled(width_, height_, Qt::IgnoreAspectRatio,
                                       Qt::SmoothTransformation);
  image_->setPixmap(scaledPixmap);

  holovibes_logger()->trace("frame displayed");
  emit frame_displayed();
}

} // namespace dh
