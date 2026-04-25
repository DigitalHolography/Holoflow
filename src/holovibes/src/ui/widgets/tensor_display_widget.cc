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

#include "ui/widgets/tensor_display_widget.hh"

#include <QDebug>
#include <QMetaType>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QPaintEvent>
#include <QPainter>
#include <QStyleOption>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::ui {

static const char *kVS = R"(#version 330 core
layout(location=0) in vec2 inPos;
layout(location=1) in vec2 inUV;
out vec2 uv;
void main(){ uv=inUV; gl_Position=vec4(inPos,0.0,1.0); })";

static const char *kFS = R"(#version 330 core
in vec2 uv; 
out vec4 frag;
uniform sampler2D tex;
uniform sampler2D colormap_tex;
uniform int use_colormap;
uniform float vmin;
uniform float vmax;

void main(){
  float raw_val = texture(tex, uv).r;
  
  // Normalize the value to 0.0 - 1.0 based on bounds
  float g = clamp((raw_val - vmin) / (vmax - vmin), 0.0, 1.0);

  if (use_colormap == 1) {
    // Sample the colormap lookup texture
    frag = texture(colormap_tex, vec2(g, 0.5));
  } else {
    // Standard Grayscale
    frag = vec4(g, g, g, 1.0);
  }
})";

static const char *kReticleVS = R"(#version 330 core
layout(location=0) in vec2 inPos;
void main(){ gl_Position=vec4(inPos,0.0,1.0); })";

static const char *kReticleFS = R"(#version 330 core
out vec4 frag;
uniform vec4 color;
void main(){ frag = color; })";

TensorDisplayWidget::TensorDisplayWidget(QWidget *p) : QOpenGLWidget(p) {
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  setMinimumSize(0, 0);
}

void TensorDisplayWidget::set_fixed_aspect(std::optional<QSize> size) {
  HOLOVIBES_CHECK(!size || size->width() > 0);
  HOLOVIBES_CHECK(!size || size->height() > 0);
  fixed_aspect_size_ = size;
  updateGeometry();
}

bool TensorDisplayWidget::hasHeightForWidth() const { return fixed_aspect_size_.has_value(); }

int TensorDisplayWidget::heightForWidth(int w) const {
  HOLOVIBES_CHECK(fixed_aspect_size_);
  auto ar = float(fixed_aspect_size_->width()) / float(fixed_aspect_size_->height());
  HOLOVIBES_CHECK(ar > 0.f);
  return int(std::round(float(w) / ar));
}

void TensorDisplayWidget::closeEvent(QCloseEvent *event) {
  event->ignore();
  this->hide();
}

void TensorDisplayWidget::set_colormap(Colormap cmap) {
  if (cmap_ != cmap) {
    cmap_ = cmap;
    update();
  }
}

void TensorDisplayWidget::set_value_range(float vmin, float vmax) {
  vmin_ = vmin;
  vmax_ = vmax;
  update();
}

void TensorDisplayWidget::initializeGL() {
  initializeOpenGLFunctions();
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  // Quad: pos(x,y), uv
  const float quad[] = {
      -1.f, -1.f, 0.f, 0.f, 1.f, -1.f, 1.f, 0.f, -1.f, 1.f, 0.f, 1.f, 1.f, 1.f, 1.f, 1.f,
  };

  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);
  glGenBuffers(1, &vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));

  auto *sp = new QOpenGLShaderProgram(this);
  sp->addShaderFromSourceCode(QOpenGLShader::Vertex, kVS);
  sp->addShaderFromSourceCode(QOpenGLShader::Fragment, kFS);
  sp->link();
  prog_ = sp->programId();

  tex_loc_      = glGetUniformLocation(prog_, "tex");
  cmap_tex_loc_ = glGetUniformLocation(prog_, "colormap_tex");
  use_cmap_loc_ = glGetUniformLocation(prog_, "use_colormap");
  vmin_loc_     = glGetUniformLocation(prog_, "vmin");
  vmax_loc_     = glGetUniformLocation(prog_, "vmax");

  glGenTextures(1, &tex_);
  glBindTexture(GL_TEXTURE_2D, tex_);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  // Swizzle RED to RGB for single-channel base mapping
  GLint swz[4] = {GL_RED, GL_RED, GL_RED, GL_ONE};
  glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swz);

  initializeColormaps();
  initializeReticle();
}

void TensorDisplayWidget::initializeColormaps() {
  // Generate a 1D mapping texture for the Twilight Colormap (size 256x1)
  struct ColorStop {
    float pos, r, g, b;
  };
  const ColorStop stops[]   = {{0.00f, 0.886f, 0.851f, 0.886f}, {0.15f, 0.584f, 0.682f, 0.843f},
                               {0.30f, 0.341f, 0.443f, 0.690f}, {0.50f, 0.118f, 0.125f, 0.216f},
                               {0.70f, 0.549f, 0.251f, 0.349f}, {0.85f, 0.784f, 0.522f, 0.584f},
                               {1.00f, 0.886f, 0.851f, 0.886f}};
  const int       num_stops = sizeof(stops) / sizeof(ColorStop);

  unsigned char cmap_data[256 * 3];
  for (int i = 0; i < 256; ++i) {
    float t   = i / 255.0f;
    int   idx = 0;
    while (idx < num_stops - 2 && t > stops[idx + 1].pos)
      idx++;

    float t0 = stops[idx].pos;
    float t1 = stops[idx + 1].pos;
    float f  = (t - t0) / (t1 - t0);

    cmap_data[i * 3 + 0] =
        static_cast<unsigned char>(255.0f * (stops[idx].r + f * (stops[idx + 1].r - stops[idx].r)));
    cmap_data[i * 3 + 1] =
        static_cast<unsigned char>(255.0f * (stops[idx].g + f * (stops[idx + 1].g - stops[idx].g)));
    cmap_data[i * 3 + 2] =
        static_cast<unsigned char>(255.0f * (stops[idx].b + f * (stops[idx + 1].b - stops[idx].b)));
  }

  glGenTextures(1, &colormap_tex_);
  glBindTexture(GL_TEXTURE_2D, colormap_tex_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, cmap_data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void TensorDisplayWidget::initializeReticle() {
  reticle_prog_ = glCreateProgram();

  GLuint      vs        = glCreateShader(GL_VERTEX_SHADER);
  const char *vs_source = kReticleVS;
  glShaderSource(vs, 1, &vs_source, nullptr);
  glCompileShader(vs);
  glAttachShader(reticle_prog_, vs);

  GLuint      fs        = glCreateShader(GL_FRAGMENT_SHADER);
  const char *fs_source = kReticleFS;
  glShaderSource(fs, 1, &fs_source, nullptr);
  glCompileShader(fs);
  glAttachShader(reticle_prog_, fs);

  glLinkProgram(reticle_prog_);
  glDeleteShader(vs);
  glDeleteShader(fs);

  reticle_color_loc_ = glGetUniformLocation(reticle_prog_, "color");

  glGenVertexArrays(1, &reticle_vao_);
  glGenBuffers(1, &reticle_vbo_);
}

void TensorDisplayWidget::ensureTexture(int w, int h, holoflow::core::DType dtype) {
  if (w == img_w_ && h == img_h_ && dtype == current_dtype_)
    return;

  img_w_         = w;
  img_h_         = h;
  current_dtype_ = dtype;

  GLint  internal_format = GL_R8;
  GLenum type            = GL_UNSIGNED_BYTE;

  if (dtype == holoflow::core::DType::U16) {
    internal_format = GL_R16;
    type            = GL_UNSIGNED_SHORT;
  } else if (dtype == holoflow::core::DType::F32) { // Assuming F32 exists
    internal_format = GL_R32F;
    type            = GL_FLOAT;
  }

  glBindTexture(GL_TEXTURE_2D, tex_);
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, GL_RED, type, nullptr);

  updateLetterboxViewport();
}

void TensorDisplayWidget::updateTexture(const void *pixels, int w, int h,
                                        holoflow::core::DType dtype) {
  ensureTexture(w, h, dtype);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  if (dtype == holoflow::core::DType::U8) {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, pixels);
  } else if (dtype == holoflow::core::DType::U16) {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_SHORT, pixels);
  } else if (dtype == holoflow::core::DType::F32) {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, pixels);
  }

  texture_dirty_ = true;
}

void TensorDisplayWidget::presentTensor(const QByteArray &bytes, int w, int h,
                                        holoflow::core::DType dtype) {
  if (!isVisible() || !isValid())
    return;

  qsizetype need = qsizetype(w) * qsizetype(h);

  if (dtype == holoflow::core::DType::U16) {
    need *= 2;
  } else if (dtype == holoflow::core::DType::F32) {
    need *= 4;
  }

  if (w <= 0 || h <= 0 || bytes.size() < need)
    return;

  makeCurrent();
  updateTexture(reinterpret_cast<const void *>(bytes.constData()), w, h, dtype);
  doneCurrent();

  update();
  emit tensorDisplayed();
}

void TensorDisplayWidget::resizeGL(int, int) { updateLetterboxViewport(); }

void TensorDisplayWidget::updateLetterboxViewport() {
  const int widget_w = std::max(0, width());
  const int widget_h = std::max(0, height());
  if (img_w_ <= 0 || img_h_ <= 0 || widget_w <= 0 || widget_h <= 0) {
    glViewport(0, 0, widget_w, widget_h);
    return;
  }

  const int aspect_w = fixed_aspect_size_ ? fixed_aspect_size_->width() : img_w_;
  const int aspect_h = fixed_aspect_size_ ? fixed_aspect_size_->height() : img_h_;

  const float wnd = float(widget_w) / float(widget_h);
  const float img = float(aspect_w) / float(aspect_h);
  if (wnd > img) {
    int h = widget_h;
    int w = int(img * h);
    int x = (widget_w - w) / 2;
    glViewport(x, 0, w, h);
  } else {
    int w = widget_w;
    int h = int(w / img);
    int y = (widget_h - h) / 2;
    glViewport(0, y, w, h);
  }
}

void TensorDisplayWidget::set_reticle_enabled(bool enabled) {
  reticle_enabled_ = enabled;
  update();
}

void TensorDisplayWidget::set_reticle_radius(double radius) {
  reticle_radius_ = qBound(0.05, radius, 1.0);
  update();
}

void TensorDisplayWidget::drawReticle() {
  if (!reticle_enabled_ || img_w_ <= 0 || img_h_ <= 0)
    return;

  const int          segments = 64;
  std::vector<float> vertices;

  for (int i = 0; i <= segments; ++i) {
    float angle = 2.0f * 3.14159265359f * float(i) / float(segments);
    vertices.push_back(std::cos(angle) * reticle_radius_);
    vertices.push_back(std::sin(angle) * reticle_radius_);
  }

  glBindVertexArray(reticle_vao_);
  glBindBuffer(GL_ARRAY_BUFFER, reticle_vbo_);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);

  glUseProgram(reticle_prog_);
  glUniform4f(reticle_color_loc_, 1.0f, 0.0f, 0.0f, 1.0f); // Red color

  glLineWidth(2.0f);
  glDrawArrays(GL_LINE_STRIP, 0, segments + 1);
  glLineWidth(1.0f);
}

void TensorDisplayWidget::paintGL() {
  glClear(GL_COLOR_BUFFER_BIT);
  if (img_w_ <= 0 || img_h_ <= 0)
    return;

  updateLetterboxViewport();

  glUseProgram(prog_);

  // Pass bounds
  glUniform1f(vmin_loc_, vmin_);
  glUniform1f(vmax_loc_, vmax_);

  // Bind Main Image Texture
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glUniform1i(tex_loc_, 0);

  // Bind Colormap Texture
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, colormap_tex_);
  glUniform1i(cmap_tex_loc_, 1);

  // Tell the shader which map we are using
  glUniform1i(use_cmap_loc_, (cmap_ == Colormap::Twilight) ? 1 : 0);

  glBindVertexArray(vao_);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  drawReticle();
}

} // namespace holovibes::ui
