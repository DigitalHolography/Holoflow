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
#include <cmath>
#include <cstring>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::ui {

static const char *kVS = R"(#version 330 core
layout(location=0) in vec2 inPos;
layout(location=1) in vec2 inUV;
out vec2 uv;
void main(){ uv=inUV; gl_Position=vec4(inPos,0.0,1.0); })";

static const char *kFS = R"(#version 330 core
in vec2 uv; out vec4 frag;
uniform sampler2D tex;
void main(){
  float g = texture(tex, uv).r;
  frag = vec4(g,g,g,1.0);
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
  setMinimumSize(160, 120);
}

void TensorDisplayWidget::set_fixed_aspect(std::optional<QSize> size) {
  HOLOVIBES_CHECK(!size || size->width() > 0);
  HOLOVIBES_CHECK(!size || size->height() > 0);
  fixed_aspect_size_ = size;
  updateGeometry();

  if (fixed_aspect_size_) {
    float ar = float(fixed_aspect_size_->width()) / float(fixed_aspect_size_->height());
    HOLOVIBES_CHECK(ar > 0.f);
    int h = height();
    int w = int(std::round(h * ar));
    // if (w > h) {
    //   h = int(std::round(w / ar));
    // } else {
    //   w = int(std::round(h * ar));
    // }
    resize(w, h);
  }
}

void TensorDisplayWidget::resizeEvent(QResizeEvent *event) {
  if (fixed_aspect_size_) {
    float ar = float(fixed_aspect_size_->width()) / float(fixed_aspect_size_->height());
    HOLOVIBES_CHECK(ar > 0.f);
    int h = event->size().height();
    int w = event->size().width();
    if (w > h) {
      h = int(std::round(w / ar));
    } else {
      w = int(std::round(h * ar));
    }
    resize(w, h);
  }

  QOpenGLWidget::resizeEvent(event);
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

void TensorDisplayWidget::initializeGL() {
  initializeOpenGLFunctions();

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
  prog_ = sp->programId(); // Qt owns the object; we keep the raw id.

  glGenTextures(1, &tex_);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  // Swizzle RED to RGB for GL_R8 -> grayscale
  GLint swz[4] = {GL_RED, GL_RED, GL_RED, GL_ONE};
  glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swz);

  // Initialize reticle rendering
  initializeReticle();
}

void TensorDisplayWidget::initializeReticle() {
  // Create shader program for reticle
  reticle_prog_ = glCreateProgram();

  // Vertex shader
  GLuint      vs        = glCreateShader(GL_VERTEX_SHADER);
  const char *vs_source = kReticleVS;
  glShaderSource(vs, 1, &vs_source, nullptr);
  glCompileShader(vs);
  glAttachShader(reticle_prog_, vs);

  // Fragment shader
  GLuint      fs        = glCreateShader(GL_FRAGMENT_SHADER);
  const char *fs_source = kReticleFS;
  glShaderSource(fs, 1, &fs_source, nullptr);
  glCompileShader(fs);
  glAttachShader(reticle_prog_, fs);

  glLinkProgram(reticle_prog_);
  glDeleteShader(vs);
  glDeleteShader(fs);

  reticle_color_loc_ = glGetUniformLocation(reticle_prog_, "color");

  // Create VAO and VBO for reticle
  glGenVertexArrays(1, &reticle_vao_);
  glGenBuffers(1, &reticle_vbo_);
}

void TensorDisplayWidget::ensureTexture(int w, int h) {
  if (w == img_w_ && h == img_h_)
    return;
  img_w_ = w;
  img_h_ = h;
  glBindTexture(GL_TEXTURE_2D, tex_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
  updateLetterboxViewport();
}

void TensorDisplayWidget::updateTexture(const void *pixels, int w, int h,
                                        holoflow::core::DType dtype) {
  ensureTexture(w, h);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  switch (dtype) {
  case holoflow::core::DType::U8: {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, pixels);
  } break;

  case holoflow::core::DType::U16: {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_SHORT, pixels);
  } break;
  }

  texture_dirty_ = true;
}

void TensorDisplayWidget::presentTensor(const QByteArray &bytes, int w, int h,
                                        holoflow::core::DType dtype) {
  qsizetype need = qsizetype(w) * qsizetype(h);

  switch (dtype) {
  case holoflow::core::DType::U16: {
    need *= qsizetype(2);
    break;
  }
  }

  if (w <= 0 || h <= 0 || bytes.size() < need)
    return;
  makeCurrent(); // safe from GUI thread
  updateTexture(reinterpret_cast<const void *>(bytes.constData()), w, h, dtype);
  doneCurrent();
  update(); // triggers paintGL
  emit tensorDisplayed();
}

void TensorDisplayWidget::resizeGL(int, int) { updateLetterboxViewport(); }

void TensorDisplayWidget::updateLetterboxViewport() {
  if (img_w_ <= 0 || img_h_ <= 0) {
    glViewport(0, 0, width(), height());
    return;
  }
  const float wnd = float(width()) / float(height());
  const float img = float(img_w_) / float(img_h_);
  if (wnd > img) {
    int h = height();
    int w = int(img * h);
    int x = (width() - w) / 2;
    glViewport(x, 0, w, h);
  } else {
    int w = width();
    int h = int(w / img);
    int y = (height() - h) / 2;
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
  glUseProgram(prog_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glBindVertexArray(vao_);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  drawReticle();
}

} // namespace holovibes::ui
