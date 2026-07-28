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
#include <QObject>
#include <QString>
#include <QUrl>
#include <optional>

class QNetworkAccessManager;

namespace holovibes::ui {

struct AvailableRelease {
  QString version;
  QUrl    page_url;
};

std::optional<AvailableRelease> find_available_release(const QByteArray &response,
                                                       const QString    &current_version);

class UpdateChecker : public QObject {
  Q_OBJECT

public:
  explicit UpdateChecker(QString current_version, QObject *parent = nullptr);

  void start();

signals:
  void update_available(const QString &version, const QUrl &page_url);

private:
  QString                current_version_;
  QNetworkAccessManager *network_manager_;
  bool                   started_ = false;
};

} // namespace holovibes::ui
