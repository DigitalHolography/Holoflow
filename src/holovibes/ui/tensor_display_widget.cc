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
    : QMainWindow(parent), width_(width), height_(height),
      image_(new QLabel(this)) {
  image_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  image_->setAlignment(Qt::AlignCenter);
  image_->setScaledContents(true);
  image_->resize(width_, height_);
  setCentralWidget(image_);
}

void TensorDisplayWidget::show_tensor(TensorView tens) {
  VLOG(2) << "showing tensor";
  CHECK(tens.data_type() == DataType::U8);
  CHECK(tens.memory_location() == MemoryLocation::HOST);
  CHECK(tens.shape().size() == 2);
  CHECK(tens.strides().at(1) == 1);
  CHECK(tens.strides().at(0) == tens.shape().at(1));

  uint8_t *tens_data = static_cast<uint8_t *>(tens.data());
  int tens_width = (int)tens.shape().at(1);
  int tens_height = (int)tens.shape().at(0);
  QImage image(tens_data, tens_width, tens_height, QImage::Format_Grayscale8);

  QPixmap pixmap = QPixmap::fromImage(image);
  QPixmap scaledPixmap =
      pixmap.scaled((int)width_, (int)height_, Qt::IgnoreAspectRatio,
                    Qt::SmoothTransformation);
  image_->setPixmap(scaledPixmap);

  // image_->setPixmap(QPixmap::fromImage(image).scaled(
  //     image_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

  VLOG(2) << "finished showing tensor";
}

} // namespace dh