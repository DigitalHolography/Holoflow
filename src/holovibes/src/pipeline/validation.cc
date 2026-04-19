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

#include "pipeline/validation.hh"

#include <format>

namespace holovibes::pipeline {

namespace {

void add_issue(ValidationResult &result, ValidationSeverity severity, std::string code,
               std::string message, std::vector<SettingsField> fields) {
  result.issues.push_back(ValidationIssue{
      .severity = severity,
      .code     = std::move(code),
      .message  = std::move(message),
      .fields   = std::move(fields),
  });
}

bool requires_processed_pipeline(const Settings &settings) {
  return settings.view_type != ViewType::RAW;
}

int recording_batch_size(const Settings &settings) {
  return settings.recording_method == RecordingMethod::RAW ? settings.load_batch
                                                           : settings.cpu_out_size;
}

int max_time_z_end(const Settings &settings) {
  if (settings.time_method == TimeMethod::RFFT) {
    return settings.time_window / 2 + 1;
  }

  return settings.time_window;
}

} // namespace

bool ValidationResult::ok() const { return !has_errors(); }

bool ValidationResult::has_errors() const {
  for (const auto &issue : issues) {
    if (issue.severity == ValidationSeverity::Error) {
      return true;
    }
  }

  return false;
}

std::vector<const ValidationIssue *> ValidationResult::issues_for(SettingsField field) const {
  std::vector<const ValidationIssue *> matches;

  for (const auto &issue : issues) {
    for (const auto issue_field : issue.fields) {
      if (issue_field == field) {
        matches.push_back(&issue);
        break;
      }
    }
  }

  return matches;
}

ValidationResult validate_settings(const Settings &settings, const ValidationContext &context) {
  ValidationResult result;

  if (settings.import_source == ImportSource::HOLOFILE) {
    if (settings.load_path.empty()) {
      add_issue(result, ValidationSeverity::Error, "load_path_empty",
                "An input holofile must be selected.", {SettingsField::LoadPath});
    } else if (!context.load_path_exists) {
      add_issue(result, ValidationSeverity::Error, "load_path_missing",
                std::format("Input holofile does not exist: {}", settings.load_path.string()),
                {SettingsField::LoadPath});
    }

    if (settings.load_end <= settings.load_begin) {
      add_issue(result, ValidationSeverity::Error, "load_range_invalid",
                "End frame must be strictly greater than start frame.",
                {SettingsField::LoadBegin, SettingsField::LoadEnd});
    }

    if (context.source_frame_count.has_value() && settings.load_end > *context.source_frame_count) {
      add_issue(result, ValidationSeverity::Error, "load_end_out_of_bounds",
                std::format("End frame ({}) exceeds the number of frames in the holofile ({}).",
                            settings.load_end, *context.source_frame_count),
                {SettingsField::LoadEnd});
    }
  } else {
    if (settings.camera_config_path.empty()) {
      add_issue(result, ValidationSeverity::Error, "camera_config_empty",
                "A camera configuration file must be selected.", {SettingsField::CameraConfigPath});
    } else if (!context.camera_config_exists) {
      add_issue(result, ValidationSeverity::Error, "camera_config_missing",
                std::format("Camera configuration file does not exist: {}",
                            settings.camera_config_path.string()),
                {SettingsField::CameraConfigPath});
    } else if (!context.camera_config_valid) {
      add_issue(result, ValidationSeverity::Error, "camera_config_invalid",
                "The selected camera configuration file is invalid or missing required fields.",
                {SettingsField::CameraConfigPath});
    }
  }

  if (settings.load_batch <= 0) {
    add_issue(result, ValidationSeverity::Error, "load_batch_non_positive",
              "Input batch size must be strictly positive.", {SettingsField::LoadBatch});
  }

  if (settings.time_window <= 0) {
    add_issue(result, ValidationSeverity::Error, "time_window_non_positive",
              "Time window must be strictly positive.", {SettingsField::TimeWindow});
  }

  if (settings.time_stride <= 0) {
    add_issue(result, ValidationSeverity::Error, "time_stride_non_positive",
              "Time stride must be strictly positive.", {SettingsField::TimeStride});
  }

  if (settings.time_window > 0 && settings.time_stride > 0 &&
      (settings.time_stride % settings.time_window != 0)) {
    add_issue(result, ValidationSeverity::Error, "time_stride_not_multiple",
              "Time stride must be a multiple of the time window.",
              {SettingsField::TimeWindow, SettingsField::TimeStride});
  }

  if (settings.import_source == ImportSource::HOLOFILE && settings.load_end > settings.load_begin &&
      settings.time_stride > (settings.load_end - settings.load_begin)) {
    add_issue(result, ValidationSeverity::Error, "time_stride_exceeds_range",
              "Time stride cannot exceed the selected input frame range.",
              {SettingsField::LoadBegin, SettingsField::LoadEnd, SettingsField::TimeStride});
  }

  if (requires_processed_pipeline(settings)) {
    if (settings.time_method == TimeMethod::NONE) {
      add_issue(result, ValidationSeverity::Error, "time_method_unsupported",
                "A time transform must be selected for processed view mode.",
                {SettingsField::TimeMethod});
    }

    if (settings.spacial_method != SpacialMethod::FRESNEL_DIFFRACTION) {
      add_issue(result, ValidationSeverity::Error, "spacial_method_unsupported",
                "Only Fresnel Diffraction is currently supported for processed view mode.",
                {SettingsField::SpacialMethod});
    }

    if (settings.view_3d_cuts) {
      add_issue(result, ValidationSeverity::Error, "view_3d_cuts_unsupported",
                "3D cuts are not currently supported by the pipeline.",
                {SettingsField::View3DCuts});
    }

    if (settings.pp_registration) {
      add_issue(result, ValidationSeverity::Error, "pp_registration_unsupported",
                "Registration is not currently supported by the pipeline.",
                {SettingsField::PpRegistration});
    }

    if (settings.pp_convolution) {
      add_issue(result, ValidationSeverity::Error, "pp_convolution_unsupported",
                "Convolution is not currently supported by the pipeline.",
                {SettingsField::PpConvolution});
    }
  }

  if (requires_processed_pipeline(settings) && settings.time_method != TimeMethod::NONE) {
    if (settings.time_z_end <= settings.time_z_begin) {
      add_issue(result, ValidationSeverity::Error, "time_z_range_invalid",
                "Z range width must be strictly positive.",
                {SettingsField::TimeZBegin, SettingsField::TimeZEnd});
    } else if (settings.time_z_end > max_time_z_end(settings)) {
      add_issue(result, ValidationSeverity::Error, "time_z_range_out_of_bounds",
                std::format("Z range end ({}) exceeds the maximum supported value ({}).",
                            settings.time_z_end, max_time_z_end(settings)),
                {SettingsField::TimeWindow, SettingsField::TimeZBegin, SettingsField::TimeZEnd});
    }
  }

  if (settings.recording_method != RecordingMethod::NONE) {
    if (settings.recording_path.empty()) {
      add_issue(result, ValidationSeverity::Error, "recording_path_empty",
                "A recording output path must be selected.", {SettingsField::RecordingPath});
    }

    if (settings.recording_count <= 0) {
      add_issue(result, ValidationSeverity::Error, "recording_count_non_positive",
                "Recording frame count must be strictly positive.",
                {SettingsField::RecordingCount});
    } else {
      const int batch_size = recording_batch_size(settings);
      if (batch_size > 0 && (settings.recording_count % batch_size != 0)) {
        add_issue(
            result, ValidationSeverity::Error, "recording_count_batch_mismatch",
            std::format(
                "Recording frame count ({}) must be divisible by the pipeline batch size ({}).",
                settings.recording_count, batch_size),
            {SettingsField::RecordingCount});
      }
    }
  }

  if (settings.autofocus_enabled) {
    if (settings.autofocus_nb_subaps <= 0) {
      add_issue(result, ValidationSeverity::Error, "autofocus_nb_subaps_non_positive",
                "Auto-focus sub-aperture count must be strictly positive.",
                {SettingsField::AutofocusNbSubaps});
    } else if ((settings.autofocus_nb_subaps % 2) == 0) {
      add_issue(result, ValidationSeverity::Error, "autofocus_nb_subaps_even",
                "Auto-focus sub-aperture count must be odd.", {SettingsField::AutofocusNbSubaps});
    }

    if (context.source_width.has_value() && context.source_height.has_value() &&
        (settings.autofocus_nb_subaps > *context.source_width ||
         settings.autofocus_nb_subaps > *context.source_height)) {
      add_issue(
          result, ValidationSeverity::Error, "autofocus_nb_subaps_too_large",
          std::format("Auto-focus sub-aperture count ({}) exceeds the source dimensions ({}x{}).",
                      settings.autofocus_nb_subaps, *context.source_width, *context.source_height),
          {SettingsField::AutofocusNbSubaps});
    }
  }

  return result;
}

} // namespace holovibes::pipeline
