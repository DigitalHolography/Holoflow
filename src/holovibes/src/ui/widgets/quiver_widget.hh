#pragma once

#include <QWidget>
#include <QPointF>
#include <vector>

class QuiverWidget final : public QWidget {
  Q_OBJECT
public:
  struct Arrow {
    QPointF pos; // logical coordinates (x,y)
    QPointF vec; // logical vector (u,v)
  };

  explicit QuiverWidget(QWidget* parent = nullptr);

  void set_arrows(std::vector<Arrow> arrows);
  void set_bounds(QRectF logical_bounds);      // e.g. (-0.5,-0.5)-(2.5,2.5)
  void set_grid_step(QPointF step);            // e.g. (1,1)
  void set_arrow_scale(double logical_to_logical); // multiply vec before drawing
  void set_arrow_head(double head_len_px, double head_angle_deg = 25.0);

protected:
  void paintEvent(QPaintEvent*) override;

private:
  QPointF map_to_px(QPointF p) const;

  std::vector<Arrow> arrows_;
  QRectF             bounds_{-0.5, -0.5, 3.0, 3.0};
  QPointF            grid_step_{1.0, 1.0};

  double arrow_scale_ = 1.0;     // scales vectors in logical units
  double head_len_px_ = 10.0;    // arrowhead length in pixels
  double head_ang_deg_= 25.0;    // arrowhead angle

  int margin_l_ = 50, margin_r_ = 20, margin_t_ = 20, margin_b_ = 40;
};