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

#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>

#define HOLOVIBES_VERSION_SEMVER2 "0.0.0-dev-0"

void copy_resources_recursively(const QString &resourceRoot, const QString &targetRoot) {
  QDirIterator it(resourceRoot, QDir::NoFilter, QDirIterator::Subdirectories);
  const int    prefixLength = resourceRoot.length() + 1;

  while (it.hasNext()) {
    QString resourcePath = it.next();

    QFileInfo info(resourcePath);
    QString   relativePath = resourcePath.mid(prefixLength);
    QString   targetPath   = targetRoot + "/" + relativePath;

    if (info.isDir()) {
      QDir().mkpath(targetPath);
    } else {
      if (!QFile::exists(targetPath)) {
        QDir().mkpath(QFileInfo(targetPath).absolutePath());
        if (QFile::copy(resourcePath, targetPath)) {
          QFile::setPermissions(targetPath,
                                QFile::permissions(targetPath) | QFileDevice::WriteOwner);
          qDebug() << "Copied" << resourcePath << "->" << targetPath;
        } else {
          qWarning() << "Failed to copy" << resourcePath << "->" << targetPath;
        }
      }
    }
  }
}

void setup_appdata() {
  QString appDataBase = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
  QString appDataPath = appDataBase + "/" + QCoreApplication::applicationVersion();

  QDir dir(appDataPath);
  if (!dir.exists()) {
    dir.mkpath(".");
  }

  copy_resources_recursively(":/resources/holovibes", appDataPath);
}

int main(int argc, char **argv) {
  spdlog::set_level(spdlog::level::debug);

  QCoreApplication::setApplicationName("Holovibes");
  QCoreApplication::setApplicationVersion(HOLOVIBES_VERSION_SEMVER2);
  QApplication              app(argc, argv);
  holovibes::ui::MainWindow main_window;
  main_window.show();
  setup_appdata();

  return QApplication::exec();
}
