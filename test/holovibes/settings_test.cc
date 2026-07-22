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

#include <algorithm>
#include <limits>
#include <string_view>

#include "pipeline/validation.hh"
#include "settings_loader.hh"

namespace holovibes::pipeline {

namespace {

bool has_issue(const ValidationResult &result, std::string_view code) {
  return std::any_of(result.issues.begin(), result.issues.end(),
                     [code](const ValidationIssue &issue) { return issue.code == code; });
}

} // namespace

TEST(SignalPlotSettingsTest, DefaultsArePositive) {
  const Settings settings{};
  EXPECT_DOUBLE_EQ(settings.signal_plot_time_window_seconds, 8.0);
  EXPECT_DOUBLE_EQ(settings.signal_plot_sample_time_seconds, 1.0 / 15.0);
  EXPECT_TRUE(settings.autofocus_skip_subapertures_outside_pupil);
  EXPECT_FALSE(settings.autofocus_use_graph_laplacian);
}

TEST(SignalPlotSettingsTest, LegacyJsonRoundTripPreservesValues) {
  Settings settings{};
  settings.signal_plot_time_window_seconds           = 12.5;
  settings.signal_plot_sample_time_seconds           = 0.025;
  settings.autofocus_skip_subapertures_outside_pupil = false;
  settings.autofocus_use_graph_laplacian             = true;

  const auto json     = settings_to_old_json(settings);
  const auto restored = old_json_to_settings(json, Settings{});

  EXPECT_EQ(json.at("compute_settings").at("image_rendering").at("autofocus").at("slope_mode"),
            "full_pairwise");
  EXPECT_DOUBLE_EQ(restored.signal_plot_time_window_seconds, 12.5);
  EXPECT_DOUBLE_EQ(restored.signal_plot_sample_time_seconds, 0.025);
  EXPECT_FALSE(restored.autofocus_skip_subapertures_outside_pupil);
  EXPECT_TRUE(restored.autofocus_use_graph_laplacian);
}

TEST(SignalPlotSettingsTest, UnknownAutofocusSlopeModeKeepsDefault) {
  Settings defaults{};
  defaults.autofocus_use_graph_laplacian = true;
  const nlohmann::json json              = {
      {"compute_settings", {{"image_rendering", {{"autofocus", {{"slope_mode", "unknown"}}}}}}},
  };

  const auto restored = old_json_to_settings(json, defaults);

  EXPECT_TRUE(restored.autofocus_use_graph_laplacian);
}

TEST(SignalPlotSettingsTest, MissingLegacyFieldsKeepDefaults) {
  Settings defaults{};
  defaults.signal_plot_time_window_seconds           = 9.0;
  defaults.signal_plot_sample_time_seconds           = 0.2;
  defaults.autofocus_skip_subapertures_outside_pupil = false;
  defaults.autofocus_use_graph_laplacian             = true;

  const auto restored = old_json_to_settings(nlohmann::json::object(), defaults);

  EXPECT_DOUBLE_EQ(restored.signal_plot_time_window_seconds, 9.0);
  EXPECT_DOUBLE_EQ(restored.signal_plot_sample_time_seconds, 0.2);
  EXPECT_FALSE(restored.autofocus_skip_subapertures_outside_pupil);
  EXPECT_TRUE(restored.autofocus_use_graph_laplacian);
}

TEST(SignalPlotSettingsTest, ValidationRejectsInvalidValues) {
  Settings settings{};
  settings.signal_plot_time_window_seconds = 0.0;
  settings.signal_plot_sample_time_seconds = std::numeric_limits<double>::infinity();

  const auto result = validate_settings(settings, {});

  EXPECT_TRUE(has_issue(result, "signal_plot_time_window_non_positive"));
  EXPECT_TRUE(has_issue(result, "signal_plot_sample_time_non_positive"));
}

} // namespace holovibes::pipeline
