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

#include "ui/graph_visualizer_widget.hh"

#include <QGraphicsScene>
#include <QGraphicsSvgItem>
#include <QGraphicsView>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QStandardPaths>
#include <QSvgRenderer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWheelEvent>

namespace holovibes::ui {

namespace {

class GraphView : public QGraphicsView {
public:
  explicit GraphView(QGraphicsScene *scene, QWidget *parent = nullptr)
      : QGraphicsView(scene, parent) {
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
  }

protected:
  void wheelEvent(QWheelEvent *event) override {
    const qreal factor = event->angleDelta().y() > 0 ? 1.15 : 1.0 / 1.15;
    scale(factor, factor);
    event->accept();
  }
};

} // namespace

GraphVisualizerWidget::GraphVisualizerWidget(QWidget *parent) : QWidget(parent) {
  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(8, 8, 8, 8);

  auto *toolbar    = new QHBoxLayout();
  auto *fit_button = new QToolButton(this);
  fit_button->setText(tr("Fit"));
  fit_button->setToolTip(tr("Fit the complete graph in the panel."));
  toolbar->addWidget(fit_button);
  toolbar->addStretch();
  status_ = new QLabel(tr("Select a pipeline graph or open a DOT file."), this);
  toolbar->addWidget(status_);
  layout->addLayout(toolbar);

  scene_ = new QGraphicsScene(this);
  view_  = new GraphView(scene_, this);
  view_->setRenderHint(QPainter::Antialiasing);
  layout->addWidget(view_, 1);

  process_ = new QProcess(this);
  connect(process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
          &GraphVisualizerWidget::render_finished);
  connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
    show_error(tr("Graphviz could not be started: %1").arg(process_->errorString()));
  });
  connect(fit_button, &QToolButton::clicked, this, &GraphVisualizerWidget::fit_graph);
}

void GraphVisualizerWidget::render_dot(const QString &dot) {
  const QString executable = QStandardPaths::findExecutable(QStringLiteral("dot"));
  if (executable.isEmpty()) {
    show_error(tr("Graphviz was not found. Install Graphviz and ensure its 'dot' executable is "
                  "available on PATH."));
    return;
  }

  status_->setText(tr("Rendering pipeline graph…"));
  process_->start(executable, {QStringLiteral("-Tsvg")});
  if (!process_->waitForStarted()) {
    show_error(tr("Graphviz could not be started: %1").arg(process_->errorString()));
    return;
  }

  process_->write(dot.toUtf8());
  process_->closeWriteChannel();
}

void GraphVisualizerWidget::show_error(const QString &message) {
  status_->setText(message);
  status_->setWordWrap(true);
}

void GraphVisualizerWidget::fit_graph() {
  if (!scene_->items().isEmpty()) {
    view_->fitInView(scene_->itemsBoundingRect(), Qt::KeepAspectRatio);
  }
}

void GraphVisualizerWidget::render_finished(int exit_code, QProcess::ExitStatus exit_status) {
  if (exit_status != QProcess::NormalExit || exit_code != 0) {
    const QString details = QString::fromUtf8(process_->readAllStandardError()).trimmed();
    show_error(details.isEmpty() ? tr("Graphviz failed to render the pipeline graph.") : details);
    return;
  }

  const QByteArray svg = process_->readAllStandardOutput();
  if (svg.isEmpty()) {
    show_error(tr("Graphviz returned an empty graph."));
    return;
  }

  auto *renderer = new QSvgRenderer(svg, this);
  if (!renderer->isValid()) {
    renderer->deleteLater();
    show_error(tr("Graphviz returned an invalid SVG graph."));
    return;
  }

  scene_->clear();
  if (renderer_ != nullptr) {
    renderer_->deleteLater();
  }
  renderer_ = renderer;

  auto *item = new QGraphicsSvgItem();
  item->setSharedRenderer(renderer_);
  scene_->addItem(item);
  scene_->setSceneRect(item->boundingRect());
  status_->setText(tr("Scroll to zoom; drag to pan."));
  status_->setWordWrap(false);
  fit_graph();
}

} // namespace holovibes::ui
