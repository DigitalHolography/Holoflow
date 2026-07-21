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

#include <QGroupBox>
#include <QPointer>

class QStackedWidget;
class QLabel;

namespace holovibes::ui {

class ZernikeHistorySettingsWidget;
class ZernikeHistoryWidget;

class SelectedWidgetSettingsPanel : public QGroupBox {
  Q_OBJECT

public:
  explicit SelectedWidgetSettingsPanel(QWidget *parent = nullptr);

  void set_selected_widget(ZernikeHistoryWidget *widget);
  void show_no_configurable_settings();
  void clear_selection();

private:
  QStackedWidget                *pages_                = nullptr;
  QWidget                       *empty_page_           = nullptr;
  QLabel                        *empty_placeholder_    = nullptr;
  ZernikeHistorySettingsWidget  *zernike_history_page_ = nullptr;
  QPointer<ZernikeHistoryWidget> selected_widget_;
  QMetaObject::Connection        destroyed_connection_;
};

} // namespace holovibes::ui
