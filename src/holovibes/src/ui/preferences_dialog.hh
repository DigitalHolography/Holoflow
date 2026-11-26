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

#pragma once

#include "holoflow/runtime/graph_display.hh"
#include <functional>

class QSettings;
class QWidget;

namespace holovibes::pipeline {
class Manager;
}

namespace holovibes::ui {

void show_preferences_dialog(QWidget *parent, pipeline::Manager &manager,
                             std::function<void()> on_auto_count_changed);
void restore_graph_preferences(QSettings &settings,
                               holoflow::core::GraphSpecDumpPreferences &spec,
                               holoflow::runtime::GraphCompiledDumpPreferences &compiled);

} // namespace holovibes::ui
