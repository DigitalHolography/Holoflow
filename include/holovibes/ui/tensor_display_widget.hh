#pragma once

#include <QLabel>
#include <QWidget>

#include "holoflow/tensor.hh"

namespace dh {

class TensorDisplayWidget : public QWidget {
  Q_OBJECT

public:
  explicit TensorDisplayWidget(int width, int height,
                               QWidget *parent = nullptr);

public slots:
  void show_tensor(TensorView tens);

private:
  int width_;
  int height_;
  QLabel *image_;
};

} // namespace dh