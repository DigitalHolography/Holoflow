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

#include <QStyle>
#include <QVariant>
#include <QWidget>

namespace holovibes::ui {

inline void refresh_validation_style(QWidget *widget) {
  if (widget == nullptr) {
    return;
  }

  widget->style()->unpolish(widget);
  widget->style()->polish(widget);
  widget->update();
}

inline void mark_validation_error(QWidget *widget) {
  if (widget == nullptr) {
    return;
  }

  widget->setProperty("validationState", "error");
  refresh_validation_style(widget);
}

inline void clear_validation_error(QWidget *widget) {
  if (widget == nullptr) {
    return;
  }

  widget->setProperty("validationState", QVariant());
  refresh_validation_style(widget);
}

} // namespace holovibes::ui
