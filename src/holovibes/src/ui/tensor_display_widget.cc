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

#include "tensor_display_widget.hh"

#include <QDebug>
#include <QMetaType>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QPaintEvent>
#include <QPainter>
#include <QStyleOption>
#include <cstring>

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

TensorDisplayWidget::TensorDisplayWidget(QWidget *p) : QOpenGLWidget(p) {
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  setMinimumSize(160, 120);
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

void TensorDisplayWidget::updateTexture(const quint8 *pixels, int w, int h) {
  ensureTexture(w, h);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, pixels);
  texture_dirty_ = true;
}

void TensorDisplayWidget::presentTensor(const QByteArray &bytes, int w, int h) {
  const qsizetype need = qsizetype(w) * qsizetype(h);
  if (w <= 0 || h <= 0 || bytes.size() < need)
    return;
  makeCurrent(); // safe from GUI thread
  updateTexture(reinterpret_cast<const quint8 *>(bytes.constData()), w, h);
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

void TensorDisplayWidget::paintGL() {
  glClear(GL_COLOR_BUFFER_BIT);
  if (img_w_ <= 0 || img_h_ <= 0)
    return;
  glUseProgram(prog_);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glBindVertexArray(vao_);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

} // namespace holovibes::ui
