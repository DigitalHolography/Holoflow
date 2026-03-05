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
#include <QCloseEvent>
#include <QImage>
#include <QOpenGLExtraFunctions>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPixmap>
#include <QWidget>
#include <optional>

#include "holoflow/core/tensor.hh"

namespace holovibes::ui {

/// Available colormaps for tensor visualization
enum class Colormap {
  Grayscale,
  Twilight
};

/// Widget that renders a 2D tensor as an image (with optional colormaps and float support).
class TensorDisplayWidget : public QOpenGLWidget, protected QOpenGLExtraFunctions {
  Q_OBJECT

public:
  explicit TensorDisplayWidget(QWidget *parent = nullptr);

  void set_fixed_aspect(std::optional<QSize> size);
  void set_reticle_enabled(bool enabled);
  void set_reticle_radius(double radius);
  
  /// Set the active colormap
  void set_colormap(Colormap cmap);

  /// Set the display range for mapping float data (or adjusting contrast on integers)
  void set_value_range(float vmin, float vmax);

public slots:
  void presentTensor(const QByteArray &bytes, int width, int height, holoflow::core::DType dtype);

signals:
  void tensorDisplayed();

protected:
  void closeEvent(QCloseEvent *event) override;
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;

  bool hasHeightForWidth() const override;
  int  heightForWidth(int w) const override;

  void resizeEvent(QResizeEvent *event) override;

private:
  void  ensureTexture(int w, int h, holoflow::core::DType dtype);
  void  updateTexture(const void *data, int w, int h, holoflow::core::DType dtype);
  void  updateLetterboxViewport();
  QRect getLetterboxRect() const;
  void  initializeReticle();
  void  initializeColormaps();
  void  drawReticle();

  GLuint tex_               = 0;
  GLuint vao_               = 0;
  GLuint vbo_               = 0;
  GLuint prog_              = 0;
  
  GLuint reticle_vao_       = 0;
  GLuint reticle_vbo_       = 0;
  GLuint reticle_prog_      = 0;
  GLint  reticle_color_loc_ = -1;
  
  GLuint colormap_tex_      = 0;
  GLint  tex_loc_           = -1;
  GLint  cmap_tex_loc_      = -1;
  GLint  use_cmap_loc_      = -1;
  GLint  vmin_loc_          = -1;
  GLint  vmax_loc_          = -1;
  
  int    img_w_ = 0, img_h_ = 0;
  bool   texture_dirty_ = false;

  bool   reticle_enabled_ = false;
  double reticle_radius_  = 1.0;
  
  Colormap cmap_ = Colormap::Grayscale;
  float vmin_ = 0.0f;
  float vmax_ = 1.0f;
  
  // Track current dtype to reallocate texture format if it changes
  holoflow::core::DType current_dtype_ = holoflow::core::DType::U8;

  std::optional<QSize> fixed_aspect_size_{std::nullopt};
};

} // namespace holovibes::ui