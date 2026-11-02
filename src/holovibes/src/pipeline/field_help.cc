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

constexpr auto kLoadPathConstraints               = std::to_array<const char *>({
    "Select an existing holofile when import source is file-based.",
});
constexpr auto kCameraConfigConstraints           = std::to_array<const char *>({
    "Select a readable JSON camera configuration file.",
});
constexpr auto kInputSamplingFrequencyConstraints = std::to_array<const char *>({
    "Must be strictly positive.",
    "Represents physical acquisition timing, not processing or playback throughput.",
});
constexpr auto kLoadBeginConstraints              = std::to_array<const char *>({
    "Must be strictly smaller than End Index.",
});
constexpr auto kLoadEndConstraints                = std::to_array<const char *>({
    "Must be strictly greater than Start Index.",
    "Must not exceed the number of frames in the holofile.",
});
constexpr auto kLoadBatchConstraints              = std::to_array<const char *>({
    "Must be strictly positive.",
    "Used as the recording batch size for raw recording.",
});
constexpr auto kFilter2DConstraints               = std::to_array<const char *>({
    "Angular Spectrum folds this filter into its propagation transfer function.",
    "Other propagation methods use a separate frequency-domain filtering stage.",
    "Uses the inner and outer radii as pixel distances in the spatial-frequency plane.",
});
constexpr auto kFilter2DInnerConstraints          = std::to_array<const char *>({
    "Must be non-negative.",
    "Must not exceed the outer radius when Filter 2D is enabled.",
});
constexpr auto kFilter2DOuterConstraints          = std::to_array<const char *>({
    "Must be non-negative.",
    "Must be greater than or equal to the inner radius when Filter 2D is enabled.",
});
constexpr auto kSpacialMethodConstraints             = std::to_array<const char *>({
    "Processed reconstruction supports Fresnel Diffraction and Angular Spectrum; LSCI (2D) "
    "calculates spatial contrast per frame; MESI estimates decorrelation time from raw frames.",
    "Angular Spectrum is disabled when Auto Focus is enabled.",
});
constexpr auto kAspPaddingConstraints             = std::to_array<const char *>({
    "Available only for Angular Spectrum propagation.",
    "Width and height must be at least the source dimensions.",
    "The number of pixels added on each axis must be even.",
});
constexpr auto kTimeMethodConstraints                = std::to_array<const char *>({
    "Processed reconstruction requires a time transform; LSCI (3D) calculates spatial-temporal "
    "contrast, while MESI analyzes raw frames directly.",
});
constexpr auto kTimeWindowConstraints                = std::to_array<const char *>({
    "Must be strictly positive.",
    "Outside processed 3D LSCI and MESI, Time Stride must be a multiple of Time Window.",
    "MESI requires a power-of-two window between 4 and 512 frames.",
});
constexpr auto kTimeStrideConstraints                = std::to_array<const char *>({
    "Must be strictly positive.",
    "Outside processed 3D LSCI and MESI, must be a multiple of Time Window.",
    "For 3D LSCI, a stride smaller than Time Window produces overlapping windows.",
    "MESI also supports overlapping analysis windows.",
    "For holofile replay, windows stay within one pass through the selected frames.",
    "For holofile import, must not exceed the selected frame range.",
});
constexpr auto kTimeZBeginConstraints             = std::to_array<const char *>({
    "Defines the start of the selected temporal-frequency range.",
    "Must stay within the current time transform output.",
});
constexpr auto kTimeZEndConstraints               = std::to_array<const char *>({
    "Defines the end of the selected temporal-frequency range.",
    "Must be strictly greater than Z start and within the current time transform output.",
});
constexpr auto kView3DCutsConstraints             = std::to_array<const char *>({
    "Not currently supported by the pipeline.",
});
constexpr auto kAccumulationConstraints           = std::to_array<const char *>({
    "Must be strictly positive.",
    "Controls both reconstruction and Shack-Hartmann sliding windows.",
    "Values above one require a single auto-focus iteration.",
});
constexpr auto kFlatfieldCutoffConstraints        = std::to_array<const char *>({
    "Must be strictly positive.",
    "Physical period of the 50% amplitude transition convention, not a hard cutoff.",
    "Converted to anisotropic Gaussian sigmas using the current image pitch.",
});
constexpr auto kSpatialContrastWindowConstraints = std::to_array<const char *>({
    "Must be odd and between 3 and 255 pixels.",
});
constexpr auto kSpatialContrastDimensionsConstraints = std::to_array<const char *>({
    "Choose LSCI (2D) in Space Transform or LSCI (3D) in Time Transform.",
});
constexpr auto kSpatialContrastMethodConstraints = std::to_array<const char *>({
    "Kernel evaluates the sliding window directly; FFT performs linear convolution with a box kernel.",
});
constexpr auto kMesiExposureConstraints              = std::to_array<const char *>({
    "Must be greater than 0 and at most one frame period.",
    "REMI/sREMI require exactly one; use the fixed LUT when frames have dead time.",
});
constexpr auto kMesiEstimatorConstraints             = std::to_array<const char *>({
    "Fixed LUT accepts 4 or more frames and estimates τ with user-supplied β and ρ.",
    "REMI/sREMI require at least 16 frames and continuous exposure; sREMI smooths β spatially.",
    "Validate synthetic grouped exposures against independently captured exposure curves.",
});
constexpr auto kMesiBetaSmoothingConstraints         = std::to_array<const char *>({
    "Odd Gaussian width between 1 and 31 pixels; 1 disables smoothing.",
});
constexpr auto kMesiBetaConstraints                  = std::to_array<const char *>({
    "Used only by the fixed LUT; must be in (0, 1].",
});
constexpr auto kMesiRhoConstraints                   = std::to_array<const char *>({
    "Used only by the fixed LUT; must be in (0, 1].",
});
constexpr auto kMesiTauMinimumConstraints            = std::to_array<const char *>({
    "Must be positive and finite, expressed in camera frame periods.",
});
constexpr auto kMesiTauMaximumConstraints            = std::to_array<const char *>({
    "Must be greater than the minimum τ, expressed in camera frame periods.",
});
constexpr auto kMesiTauSamplesConstraints            = std::to_array<const char *>({
    "Between 16 and 512 logarithmically spaced lookup-table candidates.",
});
constexpr auto kConvolutionConstraints            = std::to_array<const char *>({
    "Not currently supported by the pipeline.",
});
constexpr auto kRegistrationConstraints           = std::to_array<const char *>({
    "Not currently supported by the pipeline.",
});
constexpr auto kRecordingPathConstraints          = std::to_array<const char *>({
    "Required when recording is enabled.",
    "The parent directory must exist and be writable.",
});
constexpr auto kRecordingCountConstraints         = std::to_array<const char *>({
    "Must be strictly positive.",
    "Must be divisible by the effective pipeline batch size.",
});
constexpr auto kAutofocusNbSubapsConstraints      = std::to_array<const char *>({
    "Must be strictly positive.",
    "Must be odd.",
    "Must fit within the source dimensions.",
});
constexpr auto kAutofocusNbIterConstraints        = std::to_array<const char *>({
    "Must be strictly positive.",
    "Each pass applies one Shack-Hartmann phase correction.",
});

const FieldHelp kFieldHelp[] = {
    {SettingsField::LoadPath, "Input File",
     "Holofile used as the acquisition source when camera mode is disabled.", kLoadPathConstraints},
    {SettingsField::CameraConfigPath, "Camera Config",
     "JSON configuration file used to initialize the selected camera source.",
     kCameraConfigConstraints},
    {SettingsField::InputSamplingFrequency, "Input Sampling Frequency",
     "Physical camera frame sampling frequency used to derive Zernike history timing.",
     kInputSamplingFrequencyConstraints},
    {SettingsField::LoadBegin, "Start Index",
     "First input frame included in the imported holofile range.", kLoadBeginConstraints},
    {SettingsField::LoadEnd, "End Index", "Frame index just after the imported holofile range.",
     kLoadEndConstraints},
    {SettingsField::LoadBatch, "Batch Size",
     "Number of frames acquired or loaded together by the source stage.", kLoadBatchConstraints},
    {SettingsField::Filter2D, "Filter 2D",
     "Enables radial frequency-domain filtering of the propagated complex field.",
     kFilter2DConstraints},
    {SettingsField::Filter2DInnerRadius, "Filter 2D Inner Radius",
     "Inner cutoff radius in pixels. Frequencies inside this radius are attenuated.",
     kFilter2DInnerConstraints},
    {SettingsField::Filter2DOuterRadius, "Filter 2D Outer Radius",
     "Outer cutoff radius in pixels. Frequencies outside this radius are attenuated.",
     kFilter2DOuterConstraints},
    {SettingsField::SpacialMethod, "Space Transform",
     "Spatial propagation method applied after temporal analysis, or LSCI (2D) for local spatial "
     "contrast in each frame, or MESI for a raw-frame decorrelation-time estimate.",
     kSpacialMethodConstraints},
    {SettingsField::AspPadding, "ASP Padding",
     "Centered zero-padding resolution used for Angular Spectrum propagation.",
     kAspPaddingConstraints},
    {SettingsField::TimeMethod, "Time Transform",
     "Temporal transform used to convert frame windows into the analysis domain, or LSCI (3D) "
     "for local contrast across space and time.",
     kTimeMethodConstraints},
    {SettingsField::TimeWindow, "Time Window",
     "Number of frames in each temporal-analysis or local-metric analysis window.",
     kTimeWindowConstraints},
    {SettingsField::TimeStride, "Time Stride",
     "For temporal LSCI (3D), frames to advance between windows. 2D LSCI processes each input "
     "frame. The regular processing pipeline does not currently apply this setting.",
     kTimeStrideConstraints},
    {SettingsField::TimeZBegin, "Z Start",
     "Start of the selected temporal-frequency range displayed or processed.",
     kTimeZBeginConstraints},
    {SettingsField::TimeZEnd, "Z Width",
     "Extent of the selected temporal-frequency range displayed or processed.",
     kTimeZEndConstraints},
    {SettingsField::View3DCuts, "3D Cuts",
     "Enables XZ and YZ cut views derived from the processed volume.", kView3DCutsConstraints},
    {SettingsField::PpAccumulation, "Accumulation",
     "Number of temporal samples in the reconstruction and Shack-Hartmann sliding averages.",
     kAccumulationConstraints},
    {SettingsField::PpFlatfieldCutoffPeriod, "Flatfield Cutoff",
     "Physical cutoff period used to derive the Gaussian background subtraction scale.",
     kFlatfieldCutoffConstraints},
    {SettingsField::PpSpatialContrastWindowSize, "Spatial LSCI Window",
     "Odd square sliding-window side length in pixels for local speckle statistics.",
     kSpatialContrastWindowConstraints},
    {SettingsField::PpSpatialContrastDimensions, "Spatial LSCI Dimensions",
     "2D calculates contrast within each frame; 3D combines a sliding temporal window with "
     "a spatial neighborhood. Choose LSCI (2D) in Space Transform or LSCI (3D) in Time Transform; "
     "only one LSCI mode can be active.",
     kSpatialContrastDimensionsConstraints},
    {SettingsField::PpSpatialContrastMethod, "Spatial LSCI Method",
     "Calculation method used for the local sliding-window statistics.",
     kSpatialContrastMethodConstraints},
    {SettingsField::MesiExposureFraction, "MESI Exposure / Frame Period",
     "Camera exposure duration as a fraction of the frame period.", kMesiExposureConstraints},
    {SettingsField::MesiEstimator, "MESI Estimator",
     "Choose the fixed gated-exposure LUT, REMI, or sREMI.", kMesiEstimatorConstraints},
    {SettingsField::MesiBetaSmoothingSize, "sREMI β Smoothing Size",
     "Gaussian width applied to the preliminary β map.", kMesiBetaSmoothingConstraints},
    {SettingsField::MesiBeta, "MESI Coherence Factor β",
     "Fixed coherence factor used by the LUT model.", kMesiBetaConstraints},
    {SettingsField::MesiRho, "MESI Dynamic Fraction ρ",
     "Fixed fraction of the scattered field assigned to dynamic scatterers.", kMesiRhoConstraints},
    {SettingsField::MesiTauMinFrames, "MESI Minimum τ",
     "Lower bound of the logarithmic decorrelation-time search.", kMesiTauMinimumConstraints},
    {SettingsField::MesiTauMaxFrames, "MESI Maximum τ",
     "Upper bound of the logarithmic decorrelation-time search.", kMesiTauMaximumConstraints},
    {SettingsField::MesiTauSamples, "MESI τ LUT Samples",
     "Number of precomputed model values tested for each pixel.", kMesiTauSamplesConstraints},
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
    {SettingsField::AutofocusNbIter, "Auto-focus Iterations",
     "Number of Shack-Hartmann correction passes.", kAutofocusNbIterConstraints},
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
