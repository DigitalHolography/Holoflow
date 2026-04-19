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

#include "pipeline/field_help.hh"

#include <array>

#include "bug.hh"

namespace holovibes::pipeline {

namespace {

constexpr auto kLoadPathConstraints          = std::to_array<const char *>({
    "Select an existing holofile when import source is file-based.",
});
constexpr auto kCameraConfigConstraints      = std::to_array<const char *>({
    "Select a readable JSON camera configuration file.",
});
constexpr auto kLoadBeginConstraints         = std::to_array<const char *>({
    "Must be strictly smaller than End Index.",
});
constexpr auto kLoadEndConstraints           = std::to_array<const char *>({
    "Must be strictly greater than Start Index.",
    "Must not exceed the number of frames in the holofile.",
});
constexpr auto kLoadBatchConstraints         = std::to_array<const char *>({
    "Must be strictly positive.",
    "Used as the recording batch size for raw recording.",
});
constexpr auto kSpacialMethodConstraints     = std::to_array<const char *>({
    "Processed mode currently supports Fresnel Diffraction only.",
});
constexpr auto kTimeMethodConstraints        = std::to_array<const char *>({
    "Processed mode requires a time transform.",
});
constexpr auto kTimeWindowConstraints        = std::to_array<const char *>({
    "Must be strictly positive.",
    "Time Stride must be a multiple of Time Window.",
});
constexpr auto kTimeStrideConstraints        = std::to_array<const char *>({
    "Must be strictly positive.",
    "Must be a multiple of Time Window.",
    "For holofile import, must not exceed the selected frame range.",
});
constexpr auto kTimeZBeginConstraints        = std::to_array<const char *>({
    "Defines the start of the selected temporal-frequency range.",
    "Must stay within the current time transform output.",
});
constexpr auto kTimeZEndConstraints          = std::to_array<const char *>({
    "Defines the end of the selected temporal-frequency range.",
    "Must be strictly greater than Z start and within the current time transform output.",
});
constexpr auto kView3DCutsConstraints        = std::to_array<const char *>({
    "Not currently supported by the pipeline.",
});
constexpr auto kSpectrumConstraints          = std::to_array<const char *>({
    "Display-only toggle. No validator issue is emitted yet for this field.",
});
constexpr auto kConvolutionConstraints       = std::to_array<const char *>({
    "Not currently supported by the pipeline.",
});
constexpr auto kRegistrationConstraints      = std::to_array<const char *>({
    "Not currently supported by the pipeline.",
});
constexpr auto kRecordingPathConstraints     = std::to_array<const char *>({
    "Required when recording is enabled.",
});
constexpr auto kRecordingCountConstraints    = std::to_array<const char *>({
    "Must be strictly positive.",
    "Must be divisible by the effective pipeline batch size.",
});
constexpr auto kAutofocusNbSubapsConstraints = std::to_array<const char *>({
    "Must be strictly positive.",
    "Must be odd.",
    "Must fit within the source dimensions.",
});

const FieldHelp kFieldHelp[] = {
    {SettingsField::LoadPath, "Input File",
     "Holofile used as the acquisition source when camera mode is disabled.", kLoadPathConstraints},
    {SettingsField::CameraConfigPath, "Camera Config",
     "JSON configuration file used to initialize the selected camera source.",
     kCameraConfigConstraints},
    {SettingsField::LoadBegin, "Start Index",
     "First input frame included in the imported holofile range.", kLoadBeginConstraints},
    {SettingsField::LoadEnd, "End Index", "Frame index just after the imported holofile range.",
     kLoadEndConstraints},
    {SettingsField::LoadBatch, "Batch Size",
     "Number of frames acquired or loaded together by the source stage.", kLoadBatchConstraints},
    {SettingsField::SpacialMethod, "Space Transform",
     "Spatial propagation method applied after temporal analysis.", kSpacialMethodConstraints},
    {SettingsField::TimeMethod, "Time Transform",
     "Temporal transform used to convert frame windows into the analysis domain.",
     kTimeMethodConstraints},
    {SettingsField::TimeWindow, "Time Window", "Number of frames in each temporal analysis window.",
     kTimeWindowConstraints},
    {SettingsField::TimeStride, "Time Stride",
     "Number of frames between successive temporal analysis windows.", kTimeStrideConstraints},
    {SettingsField::TimeZBegin, "Z Start",
     "Start of the selected temporal-frequency range displayed or processed.",
     kTimeZBeginConstraints},
    {SettingsField::TimeZEnd, "Z Width",
     "Extent of the selected temporal-frequency range displayed or processed.",
     kTimeZEndConstraints},
    {SettingsField::View3DCuts, "3D Cuts",
     "Enables XZ and YZ cut views derived from the processed volume.", kView3DCutsConstraints},
    {SettingsField::ViewRawSpectrum, "Raw Spectrum View",
     "Displays the raw temporal spectrum view.", kSpectrumConstraints},
    {SettingsField::ViewProcessedSpectrum, "Processed Spectrum View",
     "Displays the processed temporal spectrum view.", kSpectrumConstraints},
    {SettingsField::PpConvolution, "Convolution",
     "Applies a convolution kernel during post-processing.", kConvolutionConstraints},
    {SettingsField::PpRegistration, "Registration",
     "Aligns successive output frames during post-processing.", kRegistrationConstraints},
    {SettingsField::RecordingPath, "Recording Path",
     "Output path used when recording raw or processed data.", kRecordingPathConstraints},
    {SettingsField::RecordingCount, "Recording Count",
     "Number of frames to record before the writer stops automatically.",
     kRecordingCountConstraints},
    {SettingsField::AutofocusNbSubaps, "Auto-focus Sub-apertures",
     "Number of Shack-Hartmann sub-apertures per image dimension.", kAutofocusNbSubapsConstraints},
};

} // namespace

const FieldHelp &get_field_help(SettingsField field) {
  for (const auto &help : kFieldHelp) {
    if (help.field == field) {
      return help;
    }
  }

  HOLOVIBES_BUG("Missing field help for settings field {}", static_cast<int>(field));
  HOLOVIBES_UNREACHABLE();
}

} // namespace holovibes::pipeline
