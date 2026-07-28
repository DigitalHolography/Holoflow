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

#include "ui/update_checker.hh"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRegularExpression>
#include <QStringList>
#include <algorithm>
#include <utility>

namespace holovibes::ui {

namespace {

constexpr auto kReleasesApiUrl =
    "https://api.github.com/repos/DigitalHolography/Holoflow/releases?per_page=100";

struct SemanticVersion {
  quint64     major;
  quint64     minor;
  quint64     patch;
  QStringList prerelease;
};

std::optional<SemanticVersion> parse_version(const QString &text) {
  static const QRegularExpression expression(
      QStringLiteral(R"(^v?([0-9]+)\.([0-9]+)\.([0-9]+)(?:-([0-9A-Za-z.-]+))?$)"));

  const auto match = expression.match(text);
  if (!match.hasMatch()) {
    return std::nullopt;
  }

  bool          major_ok = false;
  bool          minor_ok = false;
  bool          patch_ok = false;
  const quint64 major    = match.captured(1).toULongLong(&major_ok);
  const quint64 minor    = match.captured(2).toULongLong(&minor_ok);
  const quint64 patch    = match.captured(3).toULongLong(&patch_ok);
  if (!major_ok || !minor_ok || !patch_ok) {
    return std::nullopt;
  }

  QStringList prerelease;
  if (!match.captured(4).isEmpty()) {
    prerelease = match.captured(4).split('.');
    if (prerelease.contains(QString{})) {
      return std::nullopt;
    }
  }

  return SemanticVersion{major, minor, patch, std::move(prerelease)};
}

int compare_identifier(const QString &left, const QString &right) {
  bool          left_numeric  = false;
  bool          right_numeric = false;
  const quint64 left_value    = left.toULongLong(&left_numeric);
  const quint64 right_value   = right.toULongLong(&right_numeric);

  if (left_numeric && right_numeric) {
    return left_value < right_value ? -1 : left_value > right_value ? 1 : 0;
  }
  if (left_numeric != right_numeric) {
    return left_numeric ? -1 : 1;
  }
  return QString::compare(left, right, Qt::CaseSensitive);
}

int compare_versions(const SemanticVersion &left, const SemanticVersion &right) {
  if (left.major != right.major) {
    return left.major < right.major ? -1 : 1;
  }
  if (left.minor != right.minor) {
    return left.minor < right.minor ? -1 : 1;
  }
  if (left.patch != right.patch) {
    return left.patch < right.patch ? -1 : 1;
  }

  if (left.prerelease.isEmpty() != right.prerelease.isEmpty()) {
    return left.prerelease.isEmpty() ? 1 : -1;
  }

  const qsizetype common_size = std::min(left.prerelease.size(), right.prerelease.size());
  for (qsizetype index = 0; index < common_size; ++index) {
    const int comparison = compare_identifier(left.prerelease[index], right.prerelease[index]);
    if (comparison != 0) {
      return comparison;
    }
  }
  if (left.prerelease.size() == right.prerelease.size()) {
    return 0;
  }
  return left.prerelease.size() < right.prerelease.size() ? -1 : 1;
}

bool is_trusted_release_url(const QUrl &url) {
  return url.scheme() == QStringLiteral("https") &&
         url.host().compare(QStringLiteral("github.com"), Qt::CaseInsensitive) == 0 &&
         url.path().startsWith(QStringLiteral("/DigitalHolography/Holoflow/releases/"));
}

} // namespace

std::optional<AvailableRelease> find_available_release(const QByteArray &response,
                                                       const QString    &current_version) {
  const auto current = parse_version(current_version);
  if (!current.has_value()) {
    return std::nullopt;
  }

  QJsonParseError error;
  const auto      document = QJsonDocument::fromJson(response, &error);
  if (error.error != QJsonParseError::NoError || !document.isArray()) {
    return std::nullopt;
  }

  std::optional<SemanticVersion>  best_version;
  std::optional<AvailableRelease> best_release;
  for (const auto &value : document.array()) {
    const auto release = value.toObject();
    if (release.isEmpty() || release.value(QStringLiteral("draft")).toBool()) {
      continue;
    }

    const QString tag_name = release.value(QStringLiteral("tag_name")).toString();
    const auto    version  = parse_version(tag_name);
    const QUrl    page_url(release.value(QStringLiteral("html_url")).toString());
    if (!version.has_value() || compare_versions(*version, *current) <= 0 ||
        !is_trusted_release_url(page_url)) {
      continue;
    }

    if (!best_version.has_value() || compare_versions(*version, *best_version) > 0) {
      best_version = version;
      best_release = AvailableRelease{
          tag_name.startsWith('v') ? tag_name.sliced(1) : tag_name,
          page_url,
      };
    }
  }

  return best_release;
}

UpdateChecker::UpdateChecker(QString current_version, QObject *parent)
    : QObject(parent), current_version_(std::move(current_version)),
      network_manager_(new QNetworkAccessManager(this)) {}

void UpdateChecker::start() {
  if (started_) {
    return;
  }
  started_ = true;

  QNetworkRequest request{QUrl(QString::fromLatin1(kReleasesApiUrl))};
  request.setHeader(QNetworkRequest::UserAgentHeader,
                    QStringLiteral("Holovibes/%1").arg(current_version_));
  request.setRawHeader("Accept", "application/vnd.github+json");
  request.setRawHeader("X-GitHub-Api-Version", "2022-11-28");
  request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                       QNetworkRequest::NoLessSafeRedirectPolicy);
  request.setTransferTimeout(5000);

  auto *reply = network_manager_->get(request);
  connect(reply, &QNetworkReply::finished, this, [this, reply]() {
    const auto status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    if (reply->error() == QNetworkReply::NoError && status == 200) {
      const auto release = find_available_release(reply->readAll(), current_version_);
      if (release.has_value()) {
        emit update_available(release->version, release->page_url);
      }
    }
    reply->deleteLater();
  });
}

} // namespace holovibes::ui
