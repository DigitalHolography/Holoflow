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
#include <QOpenGLExtraFunctions>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPixmap>
#include <QWidget>

namespace holovibes::ui {

/// Widget that renders a 2D u8 tensor as a grayscale image.
class TensorDisplayWidget : public QOpenGLWidget, protected QOpenGLExtraFunctions {
  Q_OBJECT

public:
  explicit TensorDisplayWidget(QWidget *parent = nullptr);

  // [[nodiscard]] QSize sizeHint() const override;

public slots:
  void presentTensor(const QByteArray &bytes, int width, int height);

signals:
  void tensorDisplayed();

protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;

private:
  void ensureTexture(int w, int h);
  void updateTexture(const quint8 *data, int w, int h);
  void updateLetterboxViewport();

  GLuint tex_   = 0;
  GLuint vao_   = 0;
  GLuint vbo_   = 0;
  GLuint prog_  = 0;
  int    img_w_ = 0, img_h_ = 0;
  bool   texture_dirty_ = false;
};

} // namespace holovibes::ui
