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

#include "graph_builder_tasks.hh"

#include "bug.hh"

namespace holovibes::pipeline {

#define DEFINE_SOURCE_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                    \
  GraphBuilderTasks::TDesc GraphBuilderTasks::fn_name(SettingsType s) {                            \
    return std::move(make_source_sync_node(node_name_str, kind_str, kind_str, s).at(0));           \
  }

#define DEFINE_UNARY_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                     \
  GraphBuilderTasks::TDesc GraphBuilderTasks::fn_name(const TDesc &X, SettingsType s) {            \
    return std::move(make_unary_sync_node(node_name_str, kind_str, kind_str, X, s).at(0));         \
  }

#define DEFINE_SINK_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                      \
  void GraphBuilderTasks::fn_name(const TDesc &X, SettingsType s) {                                \
    make_unary_sync_node(node_name_str, kind_str, kind_str, X, s);                                 \
  }

#define DEFINE_NARY_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                      \
  GraphBuilderTasks::TDesc GraphBuilderTasks::fn_name(std::span<const TDesc> Xs, SettingsType s) { \
    return std::move(make_nary_sync_node(node_name_str, kind_str, kind_str, Xs, s).at(0));         \
  }

#define DEFINE_UNARY_ASYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                    \
  GraphBuilderTasks::TDesc GraphBuilderTasks::fn_name(const TDesc &X, SettingsType s) {            \
    return std::move(make_unary_async_node(node_name_str, kind_str, kind_str, X, s).at(0));        \
  }

// clang-format off
DEFINE_SOURCE_SYNC_NODE(holofile_read,                          "source",                              "Holofile",                        holotask::sources::HolofileSettings)
DEFINE_SOURCE_SYNC_NODE(empty,                                  "empty",                               "Empty",                           holonp::EmptySettings)
DEFINE_SOURCE_SYNC_NODE(zeros,                                  "zeros",                               "Zeros",                           holonp::ZerosSettings)
DEFINE_SOURCE_SYNC_NODE(asarray,                                "asarray",                             "AsArray",                         holonp::AsArraySettings)
DEFINE_UNARY_SYNC_NODE (ascontiguousarray,                      "ascontiguousarray",                   "AsContiguousArray",               holonp::AsContiguousArraySettings)
DEFINE_UNARY_SYNC_NODE (copy,                                   "copy",                                "Copy",                            holonp::CopySettings)
DEFINE_SOURCE_SYNC_NODE(ametek_s710_euresys_coaxlink_octo,      "ametek_s710_euresys_coaxlink_octo",   "AmetekS710EuresysCoaxlinkOcto",   holotask::sources::AmetekS710EuresysCoaxlinkOctoSettings)
DEFINE_SOURCE_SYNC_NODE(ametek_s711_euresys_coaxlink_qsfp_plus, "ametek_s711_euresys_coaxlink_qsfp_+", "AmetekS711EuresysCoaxlinkQSFP+",  holotask::sources::AmetekS711EuresysCoaxlinkQSFPSettings)
DEFINE_UNARY_SYNC_NODE (fresnel_qin,                            "fresnel_qin",                         "FresnelQin",                      holotask::sources::FresnelQinSettings)
DEFINE_UNARY_SYNC_NODE (fresnel_qout,                           "fresnel_qout",                        "FresnelQout",                     holotask::sources::FresnelQoutSettings)
DEFINE_SOURCE_SYNC_NODE(shack_hartmann_phase_ramps,             "shack_hartmann_phase_ramps",          "ShackHartmannPhaseRamps",         holotask::syncs::ShackHartmannPhaseRampsSettings)
DEFINE_UNARY_SYNC_NODE (memcpy,                                 "memcpy",                              "Memcpy",                          holotask::syncs::MemcpySettings)
DEFINE_UNARY_SYNC_NODE (convert,                                "conversion",                          "Conversion",                      holotask::syncs::ConversionSettings)
DEFINE_UNARY_SYNC_NODE (pca,                                    "pca",                                 "Pca",                             holotask::syncs::PcaSettings)
DEFINE_UNARY_SYNC_NODE (stft,                                   "stft",                                "Stft",                            holotask::syncs::StftSettings)
DEFINE_UNARY_SYNC_NODE (filter_2d,                              "filter_2d",                           "Filter2D",                        holotask::syncs::Filter2DSettings)
DEFINE_UNARY_SYNC_NODE (fresnel_diffraction,                    "fresnel_diffraction",                 "FresnelDiffraction",              holotask::syncs::FresnelDiffractionSettings)
DEFINE_UNARY_SYNC_NODE (angular_spectrum,                       "angular_spectrum",                    "AngularSpectrum",                 holotask::syncs::AngularSpectrumSettings)
DEFINE_UNARY_SYNC_NODE (reshape,                                "reshape",                             "Reshape",                         holonp::ReshapeSettings)
DEFINE_UNARY_SYNC_NODE (average,                                "average",                             "Average",                         holotask::syncs::AverageSettings)
DEFINE_UNARY_SYNC_NODE (convolution,                            "convolution",                         "Convolution",                     holotask::syncs::ConvolutionSettings)
DEFINE_UNARY_SYNC_NODE (crop,                                   "crop",                                "Crop",                            holotask::syncs::CropSettings)
DEFINE_UNARY_SYNC_NODE (fft_shift,                              "fft_shift",                           "FFTShift",                        holotask::syncs::FFTShiftSettings)
DEFINE_UNARY_SYNC_NODE (pct_clip,                               "pct_clip",                            "PctClip",                         holotask::syncs::PctClipSettings)
DEFINE_UNARY_SYNC_NODE (registration,                           "registration",                        "Registration",                    holotask::syncs::RegistrationSettings)
DEFINE_UNARY_SYNC_NODE (rotation,                               "rotation",                            "Rotation",                        holotask::syncs::RotationSettings)
DEFINE_UNARY_SYNC_NODE (wrap2pi,                                "wrap2pi",                             "Wrap2Pi",                         holotask::syncs::Wrap2PiSettings)
DEFINE_UNARY_SYNC_NODE (zernike,                                "zernike",                             "Zernike",                         holotask::syncs::ZernikeSettings)
DEFINE_UNARY_SYNC_NODE (zernike_phase,                          "zernike_phase",                       "ZernikePhase",                    holotask::syncs::ZernikePhaseSettings)

DEFINE_SINK_SYNC_NODE  (xy_raw_display,               "xy_raw_display",               "DisplayTensorXYRaw",              tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (xy_processed_display,         "xy_processed_display",         "DisplayTensorXY",                 tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (xz_processed_display,         "xz_processed_display",         "DisplayTensorXZ",                 tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (yz_processed_display,         "yz_processed_display",         "DisplayTensorYZ",                 tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (shack_hartmann_display,       "shack_hartmann_display",       "DisplayTensorShackHartmann",      tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (shack_hartmann_xcorr_display, "shack_hartmann_xcorr_display", "DisplayTensorShackHartmannXcorr", tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (zernike_phase_display,        "zernike_phase_display",        "DisplayTensorZernikePhase",       tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (zernike_coefficients_display, "zernike_coefficients_display", "DisplayZernikeCoefficients",      tasks::sinks::DisplayZernikeCoefficientsSettings)

DEFINE_NARY_SYNC_NODE  (concatenate,                  "concatenate",                  "Concatenate",                     holonp::ConcatenateSettings)
DEFINE_UNARY_SYNC_NODE (transpose,                    "transpose",                    "Transpose",                       holonp::TransposeSettings)
DEFINE_UNARY_SYNC_NODE (conj,                         "conj",                         "Conj",                            holonp::ConjSettings)
DEFINE_UNARY_SYNC_NODE (rfft,                         "rfft",                         "RFFT",                            holonp::RFFTSettings)
DEFINE_UNARY_SYNC_NODE (rfft2,                        "rfft2",                        "RFFT2",                           holonp::RFFT2Settings)
DEFINE_UNARY_SYNC_NODE (irfft2,                       "irfft2",                       "IRFFT2",                          holonp::IRFFT2Settings)
DEFINE_UNARY_SYNC_NODE (slice,                        "slice",                        "Slice",                           holonp::SliceSettings)
DEFINE_UNARY_SYNC_NODE (fft,                          "fft",                          "FFT",                             holonp::FFTSettings)
DEFINE_UNARY_SYNC_NODE (fft2,                         "fft2",                         "FFT2",                            holonp::FFT2Settings)
DEFINE_UNARY_SYNC_NODE (fftshift,                     "fftshift",                     "FFTShiftNp",                      holonp::FFTShiftSettings)
DEFINE_UNARY_SYNC_NODE (abs,                          "abs",                          "Abs",                             holonp::AbsSettings)
DEFINE_UNARY_SYNC_NODE (mean,                         "mean",                         "Mean",                            holonp::MeanSettings)
DEFINE_UNARY_SYNC_NODE (mean_abs,                     "mean_abs",                     "MeanAbs",                         holonp::MeanAbsSettings)
DEFINE_UNARY_SYNC_NODE (min,                          "min",                          "Min",                             holonp::MinSettings)
DEFINE_UNARY_SYNC_NODE (max,                          "max",                          "Max",                             holonp::MaxSettings)
DEFINE_UNARY_SYNC_NODE (normalize,                    "normalize",                    "Normalize",                       holonp::NormalizeSettings)
DEFINE_UNARY_ASYNC_NODE(batched_queue,                "batch_queue",                  "BatchQueue",                      holotask::asyncs::BatchQueueSettings)
DEFINE_UNARY_ASYNC_NODE(slide_avg,                    "slide_avg",                    "SlidingAverage",                  holotask::asyncs::SlidingAverageSettings)
// clang-format on

GraphBuilderTasks::TDesc GraphBuilderTasks::mul(const TDesc &A, const TDesc &B,
                                                holonp::MulSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return std::move(
      make_nary_sync_node("mul", "Mul", "Mul", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilderTasks::TDesc
GraphBuilderTasks::cross_correlation2(const TDesc &Moving, const TDesc &Reference,
                                      holonp::CrossCorrelation2Settings s) {
  std::array<TDesc, 2> inputs{Moving, Reference};
  return std::move(make_nary_sync_node("cross_correlation2", "CrossCorrelation2",
                                       "CrossCorrelation2", std::span<const TDesc>{inputs}, s)
                       .at(0));
}

GraphBuilderTasks::TDesc GraphBuilderTasks::add(const TDesc &A, const TDesc &B,
                                                holonp::AddSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return std::move(
      make_nary_sync_node("add", "Add", "Add", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilderTasks::TDesc GraphBuilderTasks::div(const TDesc &A, const TDesc &B,
                                                holonp::DivSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return std::move(
      make_nary_sync_node("div", "Div", "Div", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilderTasks::TDesc GraphBuilderTasks::sub(const TDesc &A, const TDesc &B,
                                                holonp::SubSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return std::move(
      make_nary_sync_node("sub", "Sub", "Sub", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilderTasks::TDesc GraphBuilderTasks::correct_phase(const TDesc &X, const TDesc &PhaseMask,
                                                          holotask::syncs::CorrectPhaseSettings s) {
  std::array<TDesc, 2> inputs{X, PhaseMask};
  return std::move(make_nary_sync_node("correct_phase", "CorrectPhase", "CorrectPhase",
                                       std::span<const TDesc>{inputs}, s)
                       .at(0));
}

GraphBuilderTasks::TDesc GraphBuilderTasks::equal(const TDesc &A, const TDesc &B,
                                                  holonp::EqualSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return std::move(
      make_nary_sync_node("equal", "Equal", "Equal", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilderTasks::TDesc GraphBuilderTasks::where(const TDesc &Cond, const TDesc &X, const TDesc &Y,
                                                  holonp::WhereSettings s) {
  std::array<TDesc, 3> inputs{Cond, X, Y};
  return std::move(
      make_nary_sync_node("where", "Where", "Where", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilderTasks::TDesc GraphBuilderTasks::assign(const TDesc &X, const TDesc &Y,
                                                   holonp::AssignSettings s) {
  std::array<TDesc, 2> inputs{X, Y};
  return std::move(
      make_nary_sync_node("assign", "Assign", "Assign", std::span<const TDesc>{inputs}, s).at(0));
}

void GraphBuilderTasks::holofile_write(const TDesc &X, holotask::sinks::HolofileSettings s) {
  auto node_name = "record";
  auto kind      = "HolofileWriter";
  auto reg_key   = "HolofileWriter";
  auto debug     = false;

  HOLOVIBES_CHECK(X.producer.has_value());
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name},
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, 0}, g_);

  auto      &factory     = reg_.get_sync(std::string{reg_key});
  const auto core_inputs = to_core_descs(std::span{&X, 1});
  (void)factory.infer(core_inputs, nlohmann::json(s));
}

} // namespace holovibes::pipeline
