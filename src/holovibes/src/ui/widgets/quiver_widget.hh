// Copyright 2026 Digital Holography Foundation
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

#include <QPointF>
#include <QWidget>
#include <vector>

class QuiverWidget final : public QWidget {
  Q_OBJECT
public:
  struct Arrow {
    QPointF pos; // logical coordinates (x,y)
    QPointF vec; // logical vector (u,v)
  };

  explicit QuiverWidget(QWidget *parent = nullptr);

  void set_arrows(std::vector<Arrow> arrows);
  void set_bounds(QRectF logical_bounds);          // e.g. (-0.5,-0.5)-(2.5,2.5)
  void set_grid_step(QPointF step);                // e.g. (1,1)
  void set_arrow_scale(double logical_to_logical); // multiply vec before drawing
  void set_arrow_head(double head_len_px, double head_angle_deg = 25.0);

protected:
  void paintEvent(QPaintEvent *) override;

private:
  QPointF map_to_px(QPointF p) const;

  std::vector<Arrow> arrows_;
  QRectF             bounds_{-0.5, -0.5, 3.0, 3.0};
  QPointF            grid_step_{1.0, 1.0};

  double arrow_scale_  = 1.0;  // scales vectors in logical units
  double head_len_px_  = 10.0; // arrowhead length in pixels
  double head_ang_deg_ = 25.0; // arrowhead angle

  int margin_l_ = 50, margin_r_ = 20, margin_t_ = 20, margin_b_ = 40;
};