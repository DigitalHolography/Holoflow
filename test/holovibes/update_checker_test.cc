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

#include <gtest/gtest.h>

#include "ui/update_checker.hh"

namespace holovibes::ui {

namespace {

QByteArray releases(std::initializer_list<QString> tags) {
  QByteArray result = "[";
  bool       first  = true;
  for (const auto &tag : tags) {
    if (!first) {
      result += ",";
    }
    first = false;
    result +=
        QStringLiteral(
            R"({"tag_name":"%1","html_url":"https://github.com/DigitalHolography/Holoflow/releases/tag/%1","draft":false})")
            .arg(tag)
            .toUtf8();
  }
  result += "]";
  return result;
}

} // namespace

TEST(UpdateCheckerTest, FindsNewerDevelopmentRelease) {
  const auto release =
      find_available_release(releases({"v0.2.0-dev.2", "v0.2.0-dev.10"}), "0.2.0-dev.1");

  ASSERT_TRUE(release.has_value());
  EXPECT_EQ(release->version, "0.2.0-dev.10");
}

TEST(UpdateCheckerTest, SelectsHighestSemanticVersion) {
  const auto release =
      find_available_release(releases({"v1.8.4", "v2.0.0-dev.1", "v1.10.0"}), "1.0.0");

  ASSERT_TRUE(release.has_value());
  EXPECT_EQ(release->version, "2.0.0-dev.1");
}

TEST(UpdateCheckerTest, StableReleaseOutranksPrerelease) {
  const auto release =
      find_available_release(releases({"v1.0.0-dev.20", "v1.0.0"}), "1.0.0-dev.21");

  ASSERT_TRUE(release.has_value());
  EXPECT_EQ(release->version, "1.0.0");
}

TEST(UpdateCheckerTest, EqualOrOlderReleasesDoNotProduceUpdate) {
  EXPECT_FALSE(
      find_available_release(releases({"v0.2.0-dev.1", "v0.1.9"}), "v0.2.0-dev.1").has_value());
}

TEST(UpdateCheckerTest, IgnoresDraftMalformedAndUntrustedReleases) {
  const QByteArray response = R"([
    {"tag_name":"v9.0.0","html_url":"https://github.com/DigitalHolography/Holoflow/releases/tag/v9.0.0","draft":true},
    {"tag_name":"nightly","html_url":"https://github.com/DigitalHolography/Holoflow/releases/tag/nightly","draft":false},
    {"tag_name":"v8.0.0","html_url":"https://example.com/releases/v8.0.0","draft":false}
  ])";

  EXPECT_FALSE(find_available_release(response, "1.0.0").has_value());
}

TEST(UpdateCheckerTest, InvalidResponsesDoNotProduceUpdate) {
  EXPECT_FALSE(find_available_release("not json", "1.0.0").has_value());
  EXPECT_FALSE(find_available_release("[]", "manual").has_value());
}

} // namespace holovibes::ui
