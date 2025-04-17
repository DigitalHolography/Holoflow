#pragma once

#include <QLabel>
#include <QMainWindow>
#include <QWidget>

#include "holoflow/tensor.hh"

namespace dh {

class TensorDisplayWidget : public QMainWindow {
  Q_OBJECT

public:
  explicit TensorDisplayWidget(int width, int height,
                               QWidget *parent = nullptr);

  void set_display_reticle(bool on);
  void set_reticle_radius(double r);

public slots:
  void show_tensor(TensorView tens);

signals:
  void frame_displayed();

private:
  int width_;
  int height_;
  QLabel *image_;

  bool display_reticle_;
  double reticle_radius_;
};

} // namespace dh