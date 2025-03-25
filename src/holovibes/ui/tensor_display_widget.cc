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
  assert(tens.data_type() == DataType::U8);
  assert(tens.memory_location() == MemoryLocation::HOST);
  assert(tens.shape().size() == 2);
  assert(tens.strides().at(1) == 1);
  assert(tens.strides().at(0) == tens.shape().at(1));

  uint8_t *tens_data = static_cast<uint8_t *>(tens.data());
  int tens_width = static_cast<int>(tens.shape().at(1));
  int tens_height = static_cast<int>(tens.shape().at(0));
  QImage image(tens_data, tens_width, tens_height, QImage::Format_Grayscale8);

  QPixmap pixmap = QPixmap::fromImage(image);
  QPixmap scaledPixmap = pixmap.scaled(width_, height_, Qt::IgnoreAspectRatio,
                                       Qt::SmoothTransformation);
  image_->setPixmap(scaledPixmap);
}

void TensorDisplayWidget::paintEvent(QPaintEvent *event) {
  QMainWindow::paintEvent(event);
  holovibes_logger()->trace("frame displayed");
  emit frame_displayed();
}

} // namespace dh
