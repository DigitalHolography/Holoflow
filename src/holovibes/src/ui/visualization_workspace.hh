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

#include <QByteArray>
#include <QHash>
#include <QString>
#include <QWidget>
#include <memory>
#include <vector>

class QMenu;
class QSettings;

namespace ads {
class CDockManager;
class CDockWidget;
} // namespace ads

namespace holovibes::ui {

enum class DockPlacement { Root, Left, Right, Above, Below, Tab };

struct VisualizationDescriptor {
  QString       id;
  QString       title;
  QWidget      *widget = nullptr;
  QString       default_anchor_id;
  DockPlacement default_placement = DockPlacement::Tab;
};

class VisualizationWorkspace : public QWidget {
  Q_OBJECT

public:
  explicit VisualizationWorkspace(QWidget *parent = nullptr);
  ~VisualizationWorkspace() override;

  bool register_visualization(VisualizationDescriptor descriptor);
  void finalize_registration();
  void populate_view_menu(QMenu *view_menu);

  void set_pipeline_running(bool running);
  void set_visualization_availability(const QHash<QString, bool> &availability);
  void set_visualization_title(const QString &id, const QString &title);
  void select_visualization(const QString &id);

  void save_persistent_state(QSettings &settings);
  bool restore_persistent_state(QSettings &settings);

signals:
  void selected_visualization_changed(const QString &visualization_id);

private:
  struct Entry;

  Entry *entry_for_id(const QString &id) const;
  Entry *entry_for_dock(ads::CDockWidget *dock_widget) const;
  void   add_entry_at_default_location(Entry &entry);
  void   apply_state();
  void   update_central_page();
  void   cache_active_layout();
  void   show_all_visualizations();
  void   hide_all_visualizations();
  void   dock_all_visualizations();
  void   reset_visualization_layout();
  void   save_workspace_settings_now();

  ads::CDockManager                  *dock_manager_          = nullptr;
  QWidget                            *pipeline_stopped_page_ = nullptr;
  QWidget                            *no_displays_page_      = nullptr;
  QWidget                            *all_hidden_page_       = nullptr;
  std::vector<std::unique_ptr<Entry>> entries_;
  QByteArray                          default_layout_state_;
  QByteArray                          last_active_layout_state_;
  bool                                registration_finalized_ = false;
  bool                                pipeline_running_       = false;
  bool                                applying_state_         = false;
};

} // namespace holovibes::ui
