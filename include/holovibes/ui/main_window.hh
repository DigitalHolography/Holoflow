#pragma once

#include <QMainWindow>

#include "holovibes/ui/tensor_display_widget.hh"

namespace holovibes::ui {

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow() override = default;

private:
  dh::TensorDisplayWidget *processed_display_widget_;
};

} // namespace holovibes::ui