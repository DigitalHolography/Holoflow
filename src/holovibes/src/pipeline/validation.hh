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

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "pipeline/settings.hh"

namespace holovibes::pipeline {

enum class ValidationSeverity {
  Warning,
  Error,
};

enum class SettingsField {
  LoadPath,
  CameraConfigPath,
  LoadBegin,
  LoadEnd,
  LoadBatch,
  Filter2D,
  Filter2DInnerRadius,
  Filter2DOuterRadius,
  SpacialMethod,
  TimeMethod,
  TimeWindow,
  TimeStride,
  TimeZBegin,
  TimeZEnd,
  View3DCuts,
  ViewRawSpectrum,
  ViewProcessedSpectrum,
  PpConvolution,
  PpRegistration,
  RecordingPath,
  RecordingCount,
  AutofocusNbSubaps,
};

struct ValidationIssue {
  ValidationSeverity         severity;
  std::string                code;
  std::string                message;
  std::vector<SettingsField> fields;
};

struct ValidationResult {
  std::vector<ValidationIssue> issues;

  [[nodiscard]] bool                                 ok() const;
  [[nodiscard]] bool                                 has_errors() const;
  [[nodiscard]] std::vector<const ValidationIssue *> issues_for(SettingsField field) const;
};

struct ValidationContext {
  bool                       load_path_exists     = false;
  bool                       camera_config_exists = false;
  bool                       camera_config_valid  = false;
  std::optional<std::string> recording_path_error;
  std::optional<int>         source_width;
  std::optional<int>         source_height;
  std::optional<int>         source_frame_count;
};

std::optional<std::string> validate_recording_path(const std::filesystem::path &record_path);

ValidationResult validate_settings(const Settings &settings, const ValidationContext &context);

} // namespace holovibes::pipeline
