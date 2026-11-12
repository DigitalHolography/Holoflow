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
  EXPECT_DOUBLE_EQ(settings.input_sampling_frequency_hz, 1.0e6 / 27.0);
  EXPECT_TRUE(settings.autofocus_skip_subapertures_outside_pupil);
  EXPECT_FALSE(settings.autofocus_use_graph_laplacian);
}

TEST(SignalPlotSettingsTest, LegacyJsonRoundTripPreservesValues) {
  Settings settings{};
  settings.signal_plot_time_window_seconds           = 12.5;
  settings.input_sampling_frequency_hz               = 25'000.0;
  settings.autofocus_skip_subapertures_outside_pupil = false;
  settings.autofocus_use_graph_laplacian             = true;

  const auto json     = settings_to_old_json(settings);
  const auto restored = old_json_to_settings(json, Settings{});

  EXPECT_EQ(json.at("compute_settings").at("image_rendering").at("autofocus").at("slope_mode"),
            "full_pairwise");
  EXPECT_DOUBLE_EQ(restored.signal_plot_time_window_seconds, 12.5);
  EXPECT_DOUBLE_EQ(restored.input_sampling_frequency_hz, 25'000.0);
  EXPECT_FALSE(restored.autofocus_skip_subapertures_outside_pupil);
  EXPECT_TRUE(restored.autofocus_use_graph_laplacian);
}

TEST(AngularSpectrumPaddingSettingsTest, LegacyJsonRoundTripPreservesValues) {
  Settings settings{};
  settings.asp_padding_enabled = true;
  settings.asp_padded_width    = 1280;
  settings.asp_padded_height   = 1024;

  const auto json     = settings_to_old_json(settings);
  const auto restored = old_json_to_settings(json, Settings{});

  const auto &padding =
      json.at("compute_settings").at("image_rendering").at("angular_spectrum").at("padding");
  EXPECT_TRUE(padding.at("enabled").get<bool>());
  EXPECT_EQ(padding.at("width"), 1280);
  EXPECT_EQ(padding.at("height"), 1024);
  EXPECT_TRUE(restored.asp_padding_enabled);
  EXPECT_EQ(restored.asp_padded_width, 1280);
  EXPECT_EQ(restored.asp_padded_height, 1024);
}

TEST(AngularSpectrumPaddingSettingsTest, MissingLegacyFieldsKeepDefaults) {
  Settings defaults{};
  defaults.asp_padding_enabled = true;
  defaults.asp_padded_width    = 2048;
  defaults.asp_padded_height   = 1536;

  const auto restored = old_json_to_settings(nlohmann::json::object(), defaults);

  EXPECT_TRUE(restored.asp_padding_enabled);
  EXPECT_EQ(restored.asp_padded_width, 2048);
  EXPECT_EQ(restored.asp_padded_height, 1536);
}

TEST(AngularSpectrumPaddingSettingsTest, ValidationRejectsInvalidResolution) {
  Settings settings{};
  settings.view_type           = ViewType::PROCESSED;
  settings.spacial_method      = SpacialMethod::ANGULAR_SPECTRUM;
  settings.asp_padding_enabled = true;
  ValidationContext context{
      .source_width  = 640,
      .source_height = 480,
  };

  settings.asp_padded_width  = 638;
  settings.asp_padded_height = 480;
  EXPECT_TRUE(has_issue(validate_settings(settings, context), "asp_padding_smaller_than_source"));

  settings.asp_padded_width  = 641;
  settings.asp_padded_height = 480;
  EXPECT_TRUE(has_issue(validate_settings(settings, context), "asp_padding_not_even"));

  settings.asp_padded_width  = 1024;
  settings.asp_padded_height = 1024;
  const auto valid_padding   = validate_settings(settings, context);
  EXPECT_FALSE(has_issue(valid_padding, "asp_padding_smaller_than_source"));
  EXPECT_FALSE(has_issue(valid_padding, "asp_padding_not_even"));
}

TEST(SignalPlotSettingsTest, DerivesSampleTimeFromFrequencyStride) {
  Settings settings{};
  settings.input_sampling_frequency_hz = 40'000.0;
  settings.time_stride                 = 32;
  settings.pp_accumulation             = 8;

  EXPECT_DOUBLE_EQ(settings.signal_plot_sample_time_seconds(), 32.0 / 40'000.0);

  settings.time_stride     = 64;
  settings.pp_accumulation = 4;
  EXPECT_DOUBLE_EQ(settings.signal_plot_sample_time_seconds(), 64.0 / 40'000.0);
}

TEST(SlidingAverageSettingsTest, RejectsMultipleAutofocusIterationsForNonUnitWindow) {
  Settings settings{};
  settings.view_type           = ViewType::PROCESSED;
  settings.time_method         = TimeMethod::RFFT;
  settings.spacial_method      = SpacialMethod::FRESNEL_DIFFRACTION;
  settings.time_window         = 8;
  settings.time_stride         = 8;
  settings.time_z_begin        = 0;
  settings.time_z_end          = 5;
  settings.pp_accumulation     = 4;
  settings.autofocus_enabled   = true;
  settings.autofocus_nb_subaps = 3;
  settings.autofocus_nb_iter   = 2;

  EXPECT_TRUE(
      has_issue(validate_settings(settings, {}), "autofocus_sliding_average_multiple_iterations"));

  settings.pp_accumulation = 1;
  EXPECT_FALSE(
      has_issue(validate_settings(settings, {}), "autofocus_sliding_average_multiple_iterations"));
}

TEST(LocalPcaSettingsTest, ValidatesComponentRangeAndStride) {
  Settings settings{};
  settings.view_type = ViewType::PROCESSED;
  settings.pp_spatial_contrast = true;
  settings.pp_spatial_contrast_dimensions = 3;
  settings.pp_local_pca = true;
  settings.time_window = 16;
  settings.time_z_begin = 0;
  settings.time_z_end = 15;
  settings.pp_spatial_contrast_window_size = 5;
  settings.pp_local_pca_stride = 1;
  ValidationContext context{.source_width = 8, .source_height = 8,
                            .source_frame_count = 16};
  EXPECT_FALSE(has_issue(validate_settings(settings, context), "local_pca_settings_invalid"));
  EXPECT_FALSE(has_issue(validate_settings(settings, context), "local_pca_components_invalid"));

  settings.time_z_end = 17;
  EXPECT_TRUE(has_issue(validate_settings(settings, context), "local_pca_components_invalid"));
  settings.pp_local_pca_automatic = true;
  EXPECT_FALSE(has_issue(validate_settings(settings, context), "local_pca_components_invalid"));
  settings.pp_local_pca_automatic = false;
  settings.time_z_end = 15;
  settings.pp_local_pca_stride = 9;
  EXPECT_TRUE(has_issue(validate_settings(settings, context), "local_pca_settings_invalid"));
}

TEST(LocalPcaSettingsTest, LegacyJsonPreservesAutomaticSelection) {
  Settings settings{};
  settings.pp_spatial_contrast = true;
  settings.pp_local_pca = true;
  settings.pp_local_pca_stride = 2;
  settings.pp_local_pca_automatic = true;
  settings.pp_local_pca_lsci = true;
  settings.pp_local_pca_map = 1;
  settings.pp_local_pca_dynamic_auto_scale = false;
  const auto restored = old_json_to_settings(settings_to_old_json(settings), Settings{});
  EXPECT_TRUE(restored.pp_local_pca);
  EXPECT_EQ(restored.pp_local_pca_stride, 1);
  EXPECT_TRUE(restored.pp_local_pca_automatic);
  EXPECT_TRUE(restored.pp_local_pca_lsci);
  EXPECT_EQ(restored.pp_local_pca_map, 1);
  EXPECT_FALSE(restored.pp_local_pca_dynamic_auto_scale);
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
  defaults.input_sampling_frequency_hz               = 12'345.0;
  defaults.autofocus_skip_subapertures_outside_pupil = false;
  defaults.autofocus_use_graph_laplacian             = true;

  const auto restored = old_json_to_settings(nlohmann::json::object(), defaults);

  EXPECT_DOUBLE_EQ(restored.signal_plot_time_window_seconds, 9.0);
  EXPECT_DOUBLE_EQ(restored.input_sampling_frequency_hz, 12'345.0);
  EXPECT_FALSE(restored.autofocus_skip_subapertures_outside_pupil);
  EXPECT_TRUE(restored.autofocus_use_graph_laplacian);
}

TEST(SignalPlotSettingsTest, IgnoresLegacySampleTimeField) {
  Settings defaults{};
  defaults.input_sampling_frequency_hz = 22'000.0;
  const nlohmann::json json            = {
      {"compute_settings",
       {{"image_rendering", {{"autofocus", {{"a4_history", {{"sample_time_seconds", 123.0}}}}}}}}},
  };

  const auto restored = old_json_to_settings(json, defaults);

  EXPECT_DOUBLE_EQ(restored.input_sampling_frequency_hz, 22'000.0);
}

TEST(SignalPlotSettingsTest, ValidationRejectsInvalidValues) {
  for (const double invalid_frequency : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
    Settings settings{};
    settings.input_sampling_frequency_hz = invalid_frequency;

    const auto result = validate_settings(settings, {});

    EXPECT_TRUE(has_issue(result, "input_sampling_frequency_non_positive"));
  }
}

} // namespace holovibes::pipeline
