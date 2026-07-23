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

#include "ui/visualization_workspace.hh"

#include <DockAreaWidget.h>
#include <DockManager.h>
#include <DockWidget.h>

#include <QAction>
#include <QLabel>
#include <QMenu>
#include <QSettings>
#include <QSignalBlocker>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <utility>

#include "ui/widgets/tensor_display_widget.hh"
#include "ui/widgets/zernike_history_widget.hh"

namespace {

constexpr int kLayoutStateVersion = 1;

QWidget *create_placeholder(const QString &text, QWidget *parent) {
  auto *page   = new QWidget(parent);
  auto *layout = new QVBoxLayout(page);
  layout->setContentsMargins(24, 24, 24, 24);

  auto *label = new QLabel(text, page);
  label->setObjectName("visualizationWorkspacePlaceholder");
  label->setAlignment(Qt::AlignCenter);
  label->setWordWrap(true);
  layout->addWidget(label, 1);
  return page;
}

void show_waiting_placeholder(QWidget *widget) {
  if (auto *tensor_widget = qobject_cast<holovibes::ui::TensorDisplayWidget *>(widget);
      tensor_widget != nullptr) {
    tensor_widget->show_waiting_placeholder();
    return;
  }

  if (auto *history_widget = qobject_cast<holovibes::ui::ZernikeHistoryWidget *>(widget);
      history_widget != nullptr) {
    history_widget->show_waiting_placeholder();
  }
}

ads::DockWidgetArea dock_area_for(holovibes::ui::DockPlacement placement) {
  using holovibes::ui::DockPlacement;
  switch (placement) {
  case DockPlacement::Left:
    return ads::LeftDockWidgetArea;
  case DockPlacement::Right:
    return ads::RightDockWidgetArea;
  case DockPlacement::Above:
    return ads::TopDockWidgetArea;
  case DockPlacement::Below:
    return ads::BottomDockWidgetArea;
  case DockPlacement::Root:
  case DockPlacement::Tab:
    return ads::CenterDockWidgetArea;
  }

  return ads::CenterDockWidgetArea;
}

} // namespace

namespace holovibes::ui {

struct VisualizationWorkspace::Entry {
  VisualizationDescriptor descriptor;
  ads::CDockWidget       *dock_widget      = nullptr;
  QStackedWidget         *content_stack    = nullptr;
  QWidget                *placeholder      = nullptr;
  QLabel                 *placeholder_text = nullptr;
  QAction                *view_action      = nullptr;
  bool                    content_expected = false;
  bool                    content_received = false;
  QString                 unavailable_message;
  bool                    view_preference = true;
};

VisualizationWorkspace::VisualizationWorkspace(QWidget *parent) : QWidget(parent) {
  setObjectName("displayWorkspace");
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  setMinimumSize(0, 0);

  ads::CDockManager::setConfigFlag(ads::CDockManager::FocusHighlighting, true);
  ads::CDockManager::setConfigFlag(ads::CDockManager::EqualSplitOnInsertion, true);
  ads::CDockManager::setConfigFlag(ads::CDockManager::DockAreaCloseButtonClosesTab, true);
  ads::CDockManager::setConfigFlag(ads::CDockManager::TabCloseButtonIsToolButton, true);

  auto *stack = new QStackedLayout(this);
  stack->setContentsMargins(0, 0, 0, 0);
  stack->setSpacing(0);

  pipeline_stopped_page_ =
      create_placeholder(tr("Start the pipeline to display visualizations"), this);
  no_displays_page_ = create_placeholder(
      tr("No displays are enabled\n\nEnable a display in the processing settings"), this);
  all_hidden_page_ = create_placeholder(
      tr("No visualizations are currently shown\n\nUse View > Visualizations to show one"), this);

  dock_manager_ = new ads::CDockManager(this);
  dock_manager_->setObjectName("visualizationDockManager");
  dock_manager_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  dock_manager_->setMinimumSize(0, 0);

  stack->addWidget(pipeline_stopped_page_);
  stack->addWidget(no_displays_page_);
  stack->addWidget(all_hidden_page_);
  stack->addWidget(dock_manager_);
  stack->setCurrentWidget(pipeline_stopped_page_);

  connect(dock_manager_, &ads::CDockManager::focusedDockWidgetChanged, this,
          [this](ads::CDockWidget *, ads::CDockWidget *current) {
            const auto *entry = entry_for_dock(current);
            emit        selected_visualization_changed(entry == nullptr ? QString{}
                                                                        : entry->descriptor.id);
          });
}

VisualizationWorkspace::~VisualizationWorkspace() = default;

bool VisualizationWorkspace::register_visualization(VisualizationDescriptor descriptor) {
  if (registration_finalized_ || descriptor.id.isEmpty() || descriptor.title.isEmpty() ||
      descriptor.widget == nullptr || entry_for_id(descriptor.id) != nullptr) {
    return false;
  }

  auto entry             = std::make_unique<Entry>();
  entry->descriptor      = std::move(descriptor);
  entry->view_preference = entry->descriptor.default_enabled;
  entry->dock_widget     = new ads::CDockWidget(dock_manager_, entry->descriptor.title);
  entry->dock_widget->setObjectName(entry->descriptor.id);
  entry->dock_widget->setFeatures(
      ads::CDockWidget::DockWidgetClosable | ads::CDockWidget::DockWidgetMovable |
      ads::CDockWidget::DockWidgetFloatable | ads::CDockWidget::DockWidgetFocusable);
  entry->dock_widget->setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromDockWidget);

  entry->content_stack = new QStackedWidget(entry->dock_widget);
  entry->content_stack->setContentsMargins(0, 0, 0, 0);
  entry->content_stack->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  entry->content_stack->setMinimumSize(0, 0);

  entry->descriptor.widget->setMinimumSize(0, 0);
  entry->descriptor.widget->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  entry->content_stack->addWidget(entry->descriptor.widget);

  entry->placeholder = create_placeholder({}, entry->content_stack);
  entry->placeholder_text =
      entry->placeholder->findChild<QLabel *>(QStringLiteral("visualizationWorkspacePlaceholder"));
  entry->content_stack->addWidget(entry->placeholder);

  entry->dock_widget->setWidget(entry->content_stack, ads::CDockWidget::ForceNoScrollArea);

  entry->view_action = new QAction(entry->descriptor.title, this);
  entry->view_action->setCheckable(true);
  entry->view_action->setChecked(entry->view_preference);
  entry->view_action->setEnabled(false);

  auto *entry_pointer = entry.get();
  if (auto *tensor_widget = qobject_cast<TensorDisplayWidget *>(entry_pointer->descriptor.widget);
      tensor_widget != nullptr) {
    connect(tensor_widget, &TensorDisplayWidget::tensorDisplayed, this, [this, entry_pointer]() {
      entry_pointer->content_received = true;
      update_entry_content(*entry_pointer);
    });
  } else if (auto *history_widget =
                 qobject_cast<ZernikeHistoryWidget *>(entry_pointer->descriptor.widget);
             history_widget != nullptr) {
    connect(history_widget, &ZernikeHistoryWidget::samplesDisplayed, this, [this, entry_pointer]() {
      entry_pointer->content_received = true;
      update_entry_content(*entry_pointer);
    });
  }

  connect(entry->view_action, &QAction::toggled, this, [this, entry_pointer](bool checked) {
    if (applying_state_) {
      return;
    }
    entry_pointer->view_preference = checked;
    apply_state();
    emit visualization_preferences_changed();
  });
  connect(entry->dock_widget, &ads::CDockWidget::closed, this, [this, entry_pointer]() {
    if (applying_state_) {
      return;
    }
    entry_pointer->view_preference = false;
    const QSignalBlocker blocker(entry_pointer->view_action);
    entry_pointer->view_action->setChecked(false);
    cache_active_layout();
    update_central_page();
    emit visualization_preferences_changed();
  });

  entries_.push_back(std::move(entry));
  add_entry_at_default_location(*entries_.back());
  return true;
}

void VisualizationWorkspace::finalize_registration() {
  if (registration_finalized_) {
    return;
  }

  registration_finalized_   = true;
  default_layout_state_     = dock_manager_->saveState(kLayoutStateVersion);
  last_active_layout_state_ = default_layout_state_;
  apply_state();
}

void VisualizationWorkspace::populate_view_menu(QMenu *view_menu) {
  if (view_menu == nullptr) {
    return;
  }

  auto *visualizations_menu = view_menu->addMenu(tr("Visualizations"));
  for (const auto &entry : entries_) {
    visualizations_menu->addAction(entry->view_action);
  }

  view_menu->addSeparator();
  auto *show_all_action = view_menu->addAction(tr("Show All Visualizations"));
  auto *hide_all_action = view_menu->addAction(tr("Hide All Visualizations"));
  view_menu->addSeparator();
  auto *dock_all_action = view_menu->addAction(tr("Dock All Visualizations"));
  auto *reset_action    = view_menu->addAction(tr("Reset Visualization Layout"));

  connect(show_all_action, &QAction::triggered, this,
          &VisualizationWorkspace::show_all_visualizations);
  connect(hide_all_action, &QAction::triggered, this,
          &VisualizationWorkspace::hide_all_visualizations);
  connect(dock_all_action, &QAction::triggered, this,
          &VisualizationWorkspace::dock_all_visualizations);
  connect(reset_action, &QAction::triggered, this,
          &VisualizationWorkspace::reset_visualization_layout);
}

void VisualizationWorkspace::set_pipeline_running(bool running) {
  if (pipeline_running_ == running) {
    apply_state();
    return;
  }

  if (pipeline_running_) {
    cache_active_layout();
  }
  if (!pipeline_running_ && running) {
    reset_visualization_content();
  }
  pipeline_running_ = running;
  apply_state();
}

void VisualizationWorkspace::set_visualization_content_expectations(
    const QHash<QString, VisualizationContentExpectation> &expectations) {
  for (auto &entry : entries_) {
    const auto it = expectations.constFind(entry->descriptor.id);
    if (it != expectations.cend()) {
      const bool expected_changed = entry->content_expected != it->expected;
      entry->content_expected     = it->expected;
      entry->unavailable_message  = it->unavailable_message;
      if (expected_changed || !entry->content_expected) {
        entry->content_received = false;
      }
    }
  }
  apply_state();
}

void VisualizationWorkspace::set_visualization_title(const QString &id, const QString &title) {
  auto *entry = entry_for_id(id);
  if (entry == nullptr || title.isEmpty()) {
    return;
  }

  entry->descriptor.title = title;
  entry->dock_widget->setWindowTitle(title);
  entry->view_action->setText(title);
}

void VisualizationWorkspace::set_visualization_enabled(const QString &id, bool enabled) {
  auto *entry = entry_for_id(id);
  if (entry == nullptr || entry->view_preference == enabled) {
    return;
  }

  entry->view_preference = enabled;
  apply_state();
  emit visualization_preferences_changed();
}

bool VisualizationWorkspace::is_visualization_enabled(const QString &id) const {
  const auto *entry = entry_for_id(id);
  return entry != nullptr && entry->view_preference;
}

void VisualizationWorkspace::reset_visualization_content() {
  for (auto &entry : entries_) {
    entry->content_received = false;
  }
  apply_state();
}

void VisualizationWorkspace::select_visualization(const QString &id) {
  auto *entry = entry_for_id(id);
  if (entry == nullptr || entry->dock_widget->isClosed()) {
    emit selected_visualization_changed({});
    return;
  }

  entry->dock_widget->setAsCurrentTab();
  entry->dock_widget->raise();
  dock_manager_->setDockWidgetFocused(entry->dock_widget);
  emit selected_visualization_changed(id);
}

void VisualizationWorkspace::save_persistent_state(QSettings &settings) {
  if (!registration_finalized_ || default_layout_state_.isEmpty()) {
    return;
  }

  if (pipeline_running_) {
    cache_active_layout();
  }

  settings.beginGroup("visualization_workspace");
  settings.setValue("layout_state", last_active_layout_state_);
  for (const auto &entry : entries_) {
    settings.setValue(QString("view_preferences/%1").arg(entry->descriptor.id),
                      entry->view_preference);
  }
  settings.endGroup();
}

bool VisualizationWorkspace::restore_persistent_state(QSettings &settings) {
  if (!registration_finalized_) {
    return false;
  }

  settings.beginGroup("visualization_workspace");
  for (auto &entry : entries_) {
    entry->view_preference =
        settings
            .value(QString("view_preferences/%1").arg(entry->descriptor.id), entry->view_preference)
            .toBool();
  }

  const QByteArray saved_state = settings.value("layout_state").toByteArray();
  bool             restored    = false;
  if (!saved_state.isEmpty()) {
    applying_state_ = true;
    restored        = dock_manager_->restoreState(saved_state, kLayoutStateVersion);
    if (restored) {
      for (auto &entry : entries_) {
        if (entry->dock_widget->dockAreaWidget() == nullptr) {
          add_entry_at_default_location(*entry);
        }
      }
    }
    applying_state_ = false;
  }
  settings.endGroup();

  if (!restored) {
    applying_state_               = true;
    const bool has_default_layout = !default_layout_state_.isEmpty();
    if (has_default_layout) {
      restored = dock_manager_->restoreState(default_layout_state_, kLayoutStateVersion);
    }
    applying_state_ = false;
  }

  last_active_layout_state_ = dock_manager_->saveState(kLayoutStateVersion);
  apply_state();
  return restored;
}

VisualizationWorkspace::Entry *VisualizationWorkspace::entry_for_id(const QString &id) const {
  const auto it = std::find_if(entries_.cbegin(), entries_.cend(),
                               [&id](const auto &entry) { return entry->descriptor.id == id; });
  return it == entries_.cend() ? nullptr : it->get();
}

VisualizationWorkspace::Entry *
VisualizationWorkspace::entry_for_dock(ads::CDockWidget *dock_widget) const {
  const auto it =
      std::find_if(entries_.cbegin(), entries_.cend(),
                   [dock_widget](const auto &entry) { return entry->dock_widget == dock_widget; });
  return it == entries_.cend() ? nullptr : it->get();
}

void VisualizationWorkspace::add_entry_at_default_location(Entry &entry) {
  if (entry.descriptor.default_placement == DockPlacement::Root ||
      entry.descriptor.default_anchor_id.isEmpty()) {
    dock_manager_->addDockWidget(ads::CenterDockWidgetArea, entry.dock_widget);
    return;
  }

  auto *anchor = entry_for_id(entry.descriptor.default_anchor_id);
  if (anchor == nullptr || anchor->dock_widget->dockAreaWidget() == nullptr) {
    dock_manager_->addDockWidget(ads::CenterDockWidgetArea, entry.dock_widget);
    return;
  }

  dock_manager_->addDockWidget(dock_area_for(entry.descriptor.default_placement), entry.dock_widget,
                               anchor->dock_widget->dockAreaWidget());
}

void VisualizationWorkspace::apply_state() {
  if (!registration_finalized_ || applying_state_) {
    return;
  }

  applying_state_ = true;
  for (auto &entry : entries_) {
    {
      const QSignalBlocker blocker(entry->view_action);
      entry->view_action->setChecked(entry->view_preference);
      entry->view_action->setEnabled(true);
    }

    const bool should_be_open = entry->view_preference;
    if (should_be_open == entry->dock_widget->isClosed()) {
      entry->dock_widget->toggleView(should_be_open);
    }
    update_entry_content(*entry);
  }
  applying_state_ = false;
  update_central_page();
}

void VisualizationWorkspace::update_central_page() {
  auto *stack = qobject_cast<QStackedLayout *>(layout());
  if (stack == nullptr) {
    return;
  }

  const auto visible_count =
      std::count_if(entries_.cbegin(), entries_.cend(), [](const auto &entry) {
        return entry->view_preference && !entry->dock_widget->isClosed();
      });
  stack->setCurrentWidget(visible_count == 0 ? all_hidden_page_ : dock_manager_);
}

void VisualizationWorkspace::update_entry_content(Entry &entry) {
  if (entry.content_stack == nullptr || entry.placeholder == nullptr) {
    return;
  }

  if (!entry.content_expected) {
    const QString message =
        entry.unavailable_message.isEmpty()
            ? tr("This visualization is unavailable in the current configuration.")
            : entry.unavailable_message;
    if (entry.placeholder_text != nullptr) {
      entry.placeholder_text->setText(message);
    }
    entry.content_stack->setCurrentWidget(entry.placeholder);
    return;
  }

  entry.content_stack->setCurrentWidget(entry.descriptor.widget);
  if (!entry.content_received) {
    show_waiting_placeholder(entry.descriptor.widget);
  }
}

void VisualizationWorkspace::cache_active_layout() {
  if (!registration_finalized_ || dock_manager_ == nullptr) {
    return;
  }

  const QByteArray state = dock_manager_->saveState(kLayoutStateVersion);
  if (!state.isEmpty()) {
    last_active_layout_state_ = state;
  }
}

void VisualizationWorkspace::show_all_visualizations() {
  for (auto &entry : entries_) {
    entry->view_preference = true;
  }
  apply_state();
  emit visualization_preferences_changed();
}

void VisualizationWorkspace::hide_all_visualizations() {
  for (auto &entry : entries_) {
    entry->view_preference = false;
  }
  apply_state();
  emit visualization_preferences_changed();
}

void VisualizationWorkspace::dock_all_visualizations() {
  ads::CDockAreaWidget *target_area = nullptr;
  for (const auto &entry : entries_) {
    if (!entry->dock_widget->isClosed() && !entry->dock_widget->isInFloatingContainer()) {
      target_area = entry->dock_widget->dockAreaWidget();
      break;
    }
  }

  for (const auto &entry : entries_) {
    if (entry->dock_widget->isClosed() || !entry->dock_widget->isInFloatingContainer()) {
      continue;
    }

    target_area =
        dock_manager_->addDockWidget(ads::CenterDockWidgetArea, entry->dock_widget, target_area);
  }
  cache_active_layout();
}

void VisualizationWorkspace::reset_visualization_layout() {
  if (default_layout_state_.isEmpty()) {
    return;
  }

  applying_state_     = true;
  const bool restored = dock_manager_->restoreState(default_layout_state_, kLayoutStateVersion);
  applying_state_     = false;
  if (!restored) {
    return;
  }

  last_active_layout_state_ = default_layout_state_;
  apply_state();
  save_workspace_settings_now();
}

void VisualizationWorkspace::save_workspace_settings_now() {
  QSettings settings;
  save_persistent_state(settings);
  settings.sync();
}

} // namespace holovibes::ui
