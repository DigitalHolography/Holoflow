#include "holovibes/ui/tensor_display_widget.hh"

#include <QImage>
#include <QLabel>
#include <QPixmap>
#include <QWidget>
#include <glog/logging.h>

namespace dh {

// ==========================================================================
//                     TensorDisplayWidget Implementation
// ==========================================================================

TensorDisplayWidget::TensorDisplayWidget(int width, int height, QWidget *parent)
    : QWidget(parent), width_(width), height_(height),
      image_(new QLabel(this)) {
  image_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  image_->setAlignment(Qt::AlignCenter);
  image_->setScaledContents(true);
  image_->resize(width_, height_);
}

void TensorDisplayWidget::show_tensor(TensorView tens) {
  CHECK(tens.data_type() == DataType::U8);
  CHECK(tens.memory_location() == MemoryLocation::HOST);
  CHECK(tens.shape().size() == 2);
  CHECK(tens.strides().at(1) == 1);
  CHECK(tens.strides().at(0) == tens.shape().at(1));

  uint8_t *tens_data = static_cast<uint8_t *>(tens.data());
  QImage image(tens_data, width_, height_, QImage::Format_Grayscale8);

  image_->setPixmap(QPixmap::fromImage(image).scaled(
      image_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

} // namespace dh