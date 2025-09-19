// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
