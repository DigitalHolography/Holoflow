#pragma once

#include <QByteArray>
#include <QImage>
#include <QPixmap>
#include <QWidget>

namespace holovibes::ui {

/// Widget that renders a 2D u8 tensor as a grayscale image.
class TensorDisplayWidget : public QWidget {
  Q_OBJECT

public:
  explicit TensorDisplayWidget(QWidget *parent = nullptr);

  [[nodiscard]] QSize sizeHint() const override;

public slots:
  /// Receives tensor bytes and renders them as grayscale.
  void presentTensor(const QByteArray &bytes, int width, int height);

  /// Renders an already constructed grayscale image.
  void presentImage(const QImage &image);

signals:
  /// Emitted after the widget swaps the currently displayed tensor.
  void tensorDisplayed();

protected:
  void paintEvent(QPaintEvent *event) override;

private:
  void updatePixmap(const QImage &image);

  QPixmap pixmap_;
};

} // namespace holovibes::ui
