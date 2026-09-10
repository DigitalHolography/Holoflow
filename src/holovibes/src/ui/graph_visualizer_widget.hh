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

#include <QProcess>
#include <QWidget>

class QGraphicsScene;
class QGraphicsView;
class QLabel;
class QSvgRenderer;
class QToolButton;

namespace holovibes::ui {

class GraphVisualizerWidget : public QWidget {
  Q_OBJECT

public:
  explicit GraphVisualizerWidget(QWidget *parent = nullptr);

  void render_dot(const QString &dot);
  void set_reload_enabled(bool enabled);
  void show_error(const QString &message);

signals:
  void reload_requested();

private:
  void fit_graph();
  void render_finished(int exit_code, QProcess::ExitStatus exit_status);

  QGraphicsView  *view_     = nullptr;
  QGraphicsScene *scene_    = nullptr;
  QLabel         *status_   = nullptr;
  QProcess       *process_  = nullptr;
  QSvgRenderer   *renderer_ = nullptr;
  QToolButton    *reload_button_ = nullptr;
  bool            reload_enabled_ = false;
};

} // namespace holovibes::ui
