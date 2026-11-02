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
#include <QString>
#include <QWidget>
#include <optional>
#include <utility>

#include "holoflow/core/tensor.hh"

class QLabel;
class QShowEvent;
class QTimer;

namespace holovibes::ui {

/// Available colormaps for tensor visualization
enum class Colormap { Grayscale, Twilight };

/// Widget that renders a 2D tensor as an image (with optional colormaps and float support).
class TensorDisplayWidget : public QOpenGLWidget, protected QOpenGLExtraFunctions {
  Q_OBJECT

public:
  explicit TensorDisplayWidget(QWidget *parent = nullptr);

  void set_fixed_aspect(std::optional<QSize> size);
  void set_reticle_enabled(bool enabled);
  void set_reticle_radius(double radius);
  /// Keep the last frame briefly while waiting for replacement data; nonpositive delays clear now.
  void show_waiting_placeholder(const QString &message = {}, int delay_ms = 300);

  /// Set the active colormap
  void set_colormap(Colormap cmap);

  /// Set the display range for mapping float data (or adjusting contrast on integers)
  void set_value_range(float vmin, float vmax);
  /// Derive a robust display range from each incoming F32 image without changing its data.
  void set_auto_value_range(bool enabled);
  /// Highlight nonpositive or nonfinite values, which have no valid positive measurement.
  void                                  set_nonpositive_invalid(bool enabled);
  [[nodiscard]] std::pair<float, float> display_range() const;
  /// Invert the normalized grayscale or colormap lookup value without changing tensor data.
  void set_display_inverted(bool inverted);

public slots:
  void presentTensor(const QByteArray &bytes, int width, int height, holoflow::core::DType dtype);

signals:
  void tensorDisplayed();

protected:
  void closeEvent(QCloseEvent *event) override;
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;
  void showEvent(QShowEvent *event) override;

  bool hasHeightForWidth() const override;
  int  heightForWidth(int w) const override;

private:
  void  ensureTexture(int w, int h, holoflow::core::DType dtype);
  void  updateTexture(const void *data, int w, int h, holoflow::core::DType dtype);
  void  updateLetterboxViewport();
  QRect getLetterboxRect() const;
  void  initializeReticle();
  void  initializeColormaps();
  void  drawReticle();
  void  presentPendingTensor();
  void  apply_waiting_placeholder();

  GLuint tex_  = 0;
  GLuint vao_  = 0;
  GLuint vbo_  = 0;
  GLuint prog_ = 0;

  GLuint reticle_vao_       = 0;
  GLuint reticle_vbo_       = 0;
  GLuint reticle_prog_      = 0;
  GLint  reticle_color_loc_ = -1;

  GLuint colormap_tex_ = 0;
  GLint  tex_loc_      = -1;
  GLint  cmap_tex_loc_ = -1;
  GLint  use_cmap_loc_ = -1;
  GLint  vmin_loc_     = -1;
  GLint  vmax_loc_     = -1;
  GLint  invert_display_loc_ = -1;
  GLint  nonpositive_invalid_loc_ = -1;

  int  img_w_ = 0, img_h_ = 0;
  bool texture_storage_allocated_ = false;

  bool   reticle_enabled_ = false;
  double reticle_radius_  = 1.0;

  Colormap cmap_ = Colormap::Grayscale;
  float    vmin_ = 0.0f;
  float    vmax_ = 1.0f;
  bool     display_inverted_ = false;
  bool     auto_value_range_    = false;
  bool     nonpositive_invalid_ = false;

  // Track current dtype to reallocate texture format if it changes
  holoflow::core::DType current_dtype_ = holoflow::core::DType::U8;

  std::optional<QSize> fixed_aspect_size_{std::nullopt};
  QLabel              *waiting_label_ = nullptr;
  QTimer              *waiting_timer_ = nullptr;
  QString              pending_waiting_message_;
  QByteArray            pending_tensor_bytes_;
  int                   pending_tensor_width_  = 0;
  int                   pending_tensor_height_ = 0;
  holoflow::core::DType pending_tensor_dtype_  = holoflow::core::DType::U8;
  bool                  has_pending_tensor_    = false;
};

} // namespace holovibes::ui
