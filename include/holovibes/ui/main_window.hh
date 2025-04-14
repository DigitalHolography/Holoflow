#pragma once

#include <QMainWindow>

namespace holovibes::ui {

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow() override = default;
};

} // namespace holovibes::ui