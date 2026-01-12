#include "quiver_widget.hh"

#include <QPainter>
#include <QtMath>

QuiverWidget::QuiverWidget(QWidget* parent) : QWidget(parent) {
  setMinimumSize(240, 240);
}

void QuiverWidget::set_arrows(std::vector<Arrow> arrows) {
  arrows_ = std::move(arrows);
  update();
}
void QuiverWidget::set_bounds(QRectF b) { bounds_ = b; update(); }
void QuiverWidget::set_grid_step(QPointF s) { grid_step_ = s; update(); }
void QuiverWidget::set_arrow_scale(double s) { arrow_scale_ = s; update(); }
void QuiverWidget::set_arrow_head(double len_px, double ang_deg) {
  head_len_px_ = len_px; head_ang_deg_ = ang_deg; update();
}

QPointF QuiverWidget::map_to_px(QPointF p) const {
  const QRect plot = rect().adjusted(margin_l_, margin_t_, -margin_r_, -margin_b_);

  const double x0 = bounds_.left();
  const double x1 = bounds_.right();
  const double y0 = bounds_.top();
  const double y1 = bounds_.bottom();

  const double nx = (p.x() - x0) / (x1 - x0);
  const double ny = (p.y() - y0) / (y1 - y0);

  // Qt Y axis is downward; invert to get math-like coords.
  const double px = plot.left() + nx * plot.width();
  const double py = plot.bottom() - ny * plot.height();
  return {px, py};
}

void QuiverWidget::paintEvent(QPaintEvent*) {
  QPainter qp(this);
  qp.setRenderHint(QPainter::Antialiasing, true);

  // Background
  qp.fillRect(rect(), Qt::white);

  const QRect plot = rect().adjusted(margin_l_, margin_t_, -margin_r_, -margin_b_);
  qp.setPen(QPen(Qt::black, 1.0));
  qp.drawRect(plot);

  // Grid
  qp.setPen(QPen(QColor(200, 200, 200), 1.0, Qt::DashLine));

  auto draw_vline = [&](double x) {
    const auto a = map_to_px({x, bounds_.top()});
    const auto b = map_to_px({x, bounds_.bottom()});
    qp.drawLine(a, b);
  };
  auto draw_hline = [&](double y) {
    const auto a = map_to_px({bounds_.left(), y});
    const auto b = map_to_px({bounds_.right(), y});
    qp.drawLine(a, b);
  };

  for (double x = std::ceil(bounds_.left() / grid_step_.x()) * grid_step_.x();
       x <= bounds_.right() + 1e-9; x += grid_step_.x()) {
    draw_vline(x);
  }
  for (double y = std::ceil(bounds_.top() / grid_step_.y()) * grid_step_.y();
       y <= bounds_.bottom() + 1e-9; y += grid_step_.y()) {
    draw_hline(y);
  }

  // Ticks + labels (simple)
  qp.setPen(QPen(Qt::black, 1.0));
  qp.setFont(QFont("Sans", 9));

  for (double x = std::ceil(bounds_.left() / grid_step_.x()) * grid_step_.x();
       x <= bounds_.right() + 1e-9; x += grid_step_.x()) {
    const auto p = map_to_px({x, bounds_.top()});
    qp.drawLine(QPointF(p.x(), plot.bottom()), QPointF(p.x(), plot.bottom() + 5));
    qp.drawText(QPointF(p.x() - 10, plot.bottom() + 18), QString::number(x));
  }
  for (double y = std::ceil(bounds_.top() / grid_step_.y()) * grid_step_.y();
       y <= bounds_.bottom() + 1e-9; y += grid_step_.y()) {
    const auto p = map_to_px({bounds_.left(), y});
    qp.drawLine(QPointF(plot.left() - 5, p.y()), QPointF(plot.left(), p.y()));
    qp.drawText(QPointF(5, p.y() + 4), QString::number(y));
  }

  // Arrows (quiver)
  qp.setPen(QPen(Qt::black, 2.0));
  qp.setBrush(Qt::NoBrush);

  const double head_ang = qDegreesToRadians(head_ang_deg_);

  for (const auto& a : arrows_) {
    const QPointF p0 = map_to_px(a.pos);
    const QPointF p1 = map_to_px(a.pos + arrow_scale_ * a.vec);

    qp.drawLine(p0, p1);

    // Arrowhead in pixel space
    const QPointF d = p0 - p1; // direction from tip backwards
    const double  L = std::hypot(d.x(), d.y());
    if (L < 1e-6) continue;

    const QPointF u = d / L; // unit vector backwards from tip
    const auto rot = [&](double ang) {
      const double c = std::cos(ang), s = std::sin(ang);
      return QPointF(c * u.x() - s * u.y(), s * u.x() + c * u.y());
    };

    const QPointF h1 = p1 + head_len_px_ * rot(+head_ang);
    const QPointF h2 = p1 + head_len_px_ * rot(-head_ang);

    qp.drawLine(p1, h1);
    qp.drawLine(p1, h2);
  }

  // Title (optional)
  qp.setFont(QFont("Sans", 11, QFont::Bold));
  qp.drawText(QRect(0, 0, width(), margin_t_), Qt::AlignCenter,
              "Wavefront Slopes from Sub-aperture Shifts");
}