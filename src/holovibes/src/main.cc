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

#include <QApplication>
#include <cstdlib>

#include "bug.hh"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/graph_exec.hh"
#include "logger.hh"
#include "spdlog/common.h"
#include "ui/main_window.hh"
#include "ui/tensor_display_widget.hh"
#include "app_utils.hh"

int main(int argc, char **argv) {
  spdlog::set_level(spdlog::level::debug);

  QCoreApplication::setApplicationName("Holovibes");
  QCoreApplication::setApplicationVersion(HOLOVIBES_VERSION_SEMVER2);
  holovibes::utils::setupAppData();
  QApplication              app(argc, argv);
  holovibes::ui::MainWindow main_window;
  app.setWindowIcon(QIcon(":/resources/holovibes/assets/holovibes_logo.png"));
  main_window.show();
  return QApplication::exec();
}
