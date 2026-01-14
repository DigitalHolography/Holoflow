// Copyright 2025 Digital Holography Foundation
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

#include "graph_builder_v2.hh"

#include "bug.hh"
#include "logger.hh"
#include "settings_loader.hh"

namespace holovibes::pipeline {

// -------------------------------------------------------------------------------------------------
// GraphBuilder Implementation
// -------------------------------------------------------------------------------------------------

GraphBuilder_v2::GraphBuilder_v2(const Settings &settings, holoflow::core::Registry &registry)
    : s_(settings), reg_(registry) {}

holoflow::core::GraphSpec GraphBuilder_v2::build() {
  auto cam_path = s_.camera_config_path.string();

  auto lam    = s_.spacial_lambda;
  auto dx     = s_.spacial_pixel_size;
  auto dy     = s_.spacial_pixel_size;
  auto z_prop = s_.spacial_z;

  auto ri = s_.filter_r_inner;
  auto ro = s_.filter_r_outer;
  auto si = s_.filter_smooth_inner;
  auto so = s_.filter_smooth_outer;

  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Host    = holotask::syncs::MemcpySettings::Target::Host;
  auto Device  = holotask::syncs::MemcpySettings::Target::Device;

  // -------------------------------------------------------------------------------------------------
  // Acquisition (H - Hologram)
  // -------------------------------------------------------------------------------------------------

  TDesc H;

  if (s_.import_source == ImportSource::HOLOFILE) {
    std::tie(H) =
        unpack<1>(holofile_read({s_.load_path.string(), load_method_map_.at(s_.load_method),
                                 s_.load_begin, s_.load_end, s_.load_batch}));
  } else if (s_.import_source == ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO) {
    std::tie(H) = unpack<1>(ametek_s710_euresys_coaxlink_octo({cam_path}));
  } else if (s_.import_source == ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP) {
    std::tie(H) = unpack<1>(ametek_s711_euresys_coaxlink_qsfp_plus({cam_path}));
  } else {
    HOLOVIBES_UNREACHABLE();
  }

  // Raw record
  if (s_.recording_method == RecordingMethod::RAW) {
    TDesc H_rec;
    auto  count    = s_.recording_count;
    auto  batch    = s_.time_window;
    auto  path     = s_.recording_path.string();
    auto  settings = settings_to_old_json(s_);

    std::tie(H_rec) = unpack<1>(memcpy(H, {Host}));
    std::tie(H_rec) = unpack<1>(batched_queue(H_rec, {count, batch, batch}));
    holofile_write(H_rec, {path, count, settings});
  }

  // Raw view
  if (s_.raw_view || s_.view_type == ViewType::RAW) {
    auto shape  = H.shape;
    shape.at(0) = 1;

    auto [H_disp]     = unpack<1>(memcpy(H, {Host}));
    auto [H_view]     = unpack<1>(batched_queue(H_disp, {s_.cpu_out_size, 1, 1}));
    auto [H_reshaped] = unpack<1>(reshape(H_view, {shape}));

    if (s_.raw_view) {
      xy_raw_display(H_reshaped, {});
    }
    if (s_.view_type == ViewType::RAW) {
      xy_processed_display(H_reshaped, {});
      return g_;
    }
  }

  // GPU transfer and conversion to complex32
  if (s_.load_method != LoadMethod::LOAD_IN_GPU) {
    std::tie(H) = unpack<1>(memcpy(H, {Device}));
    std::tie(H) = unpack<1>(batched_queue(H, {s_.gpu_in_size, s_.time_window, s_.time_window}));
  }
  std::tie(H) = unpack<1>(convert(H, {Target::CF32, Strat::Real}));

  // -------------------------------------------------------------------------------------------------
  // Spacial Propagation (H -> H_z - Propagated Hologram)
  // -------------------------------------------------------------------------------------------------

  TDesc H_z;

  if (s_.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION) {
    std::tie(H_z) = unpack<1>(fresnel_diffraction(H, {lam, dx, dy, z_prop}));
  }

  else if (s_.spacial_method == SpacialMethod::ANGULAR_SPECTRUM) {
    std::tie(H_z) = unpack<1>(angular_spectrum(H, {lam, dx, dy, z_prop, std::nullopt}));
  }

  if (s_.filter_2d) {
    std::tie(H_z) = unpack<1>(filter_2d(H_z, {ri, ro, si, so}));
  }

  // -------------------------------------------------------------------------------------------------
  // Time-Frequency Analysis (H_z -> FH_z - Frequency Hologram)
  // -------------------------------------------------------------------------------------------------

  TDesc FH_z;

  if (s_.time_method != TimeMethod::NONE) {
    std::tie(H_z) = unpack<1>(batched_queue(H_z, {s_.gpu_in_size, s_.time_window, s_.time_window}));
  }

  if (s_.time_method == TimeMethod::SHORT_TIME_FOURIER) {
    // TODO: Enquire about Zoom FFT
    std::tie(FH_z) = unpack<1>(stft(H_z, {}));
    std::tie(FH_z) = unpack<1>(memcpy(FH_z, {Device}));
  }

  else if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS) {
    // PCA range depends on whether we need the full 3D volume for cuts
    int pca_min = s_.view_3d_cuts ? 0 : s_.time_z_begin;
    int pca_max = s_.view_3d_cuts ? s_.time_window : s_.time_z_end;

    std::tie(FH_z) = unpack<1>(pca(H_z, {pca_min, pca_max}));
  }

  // TODO: This costs a lot of memory when time_window is large
  // if (s_.time_method != TimeMethod::NONE) {
  //   std::tie(FH_z) =
  //       unpack<1>(batched_queue(FH_z, {s_.time_window, s_.time_window, s_.time_window}));
  // }

  // -------------------------------------------------------------------------------------------------
  // Spectral Density (FH_z -> S - Spectral Density)
  // -------------------------------------------------------------------------------------------------
  auto [S] = unpack<1>(convert(FH_z, {Target::F32, Strat::Modulus}));

  // -------------------------------------------------------------------------------------------------
  // XY View Processing (S -> M0 - Processed XY View)
  // -------------------------------------------------------------------------------------------------
  {
    // Define accumulation range (act as Time-domain frequency filter)
    int z0 = (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS && !s_.view_3d_cuts)
                 ? 0
                 : s_.time_z_begin;
    int z1 = (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS && !s_.view_3d_cuts)
                 ? (int)S.shape.at(0)
                 : s_.time_z_end;

    auto [M0] = unpack<1>(average(S, {0, z0, z1}));

    if (s_.pp_fft_shift) {
      std::tie(M0) = unpack<1>(fft_shift(M0, {.axes = {1, 2}}));
    }

    if (s_.pp_registration) {
      std::tie(M0) = unpack<1>(registration(M0, {s_.pp_registration_radius}));
    }

    auto [M0_avg] = unpack<1>(slide_avg(M0, {128, (size_t)s_.pp_accumulation}));

    if (s_.pp_convolution) {
      std::tie(M0_avg) =
          unpack<1>(convolution(M0_avg, {s_.pp_convolution_path, s_.pp_convolution_divide}));
    }

    std::tie(M0_avg) = unpack<1>(convert(M0_avg, {Target::U8, Strat::Scaled}));
    std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.gpu_out_size, 1, 1}));
    std::tie(M0_avg) = unpack<1>(memcpy(M0_avg, {Host}));
    std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.cpu_out_size, 1, 1}));
    xy_processed_display(M0_avg, {});

    // XY processed recording
    if (s_.recording_method == RecordingMethod::PROCESSED) {
      auto path     = s_.recording_path;
      auto count    = s_.recording_count;
      auto settings = settings_to_old_json(s_);

      auto [M0_rec]    = unpack<1>(memcpy(M0_avg, {Host}));
      std::tie(M0_rec) = unpack<1>(batched_queue(M0_rec, {s_.cpu_out_size, 1, 1}));
      holofile_write(M0_rec, {path.string(), count, settings});
    }
  }

  if (s_.view_3d_cuts) {
    // -------------------------------------------------------------------------------------------------
    // XZ View Processing (S -> M0 - Processed XZ View)
    // -------------------------------------------------------------------------------------------------
    {
      std::vector<size_t> crop_origin = {0, 10, 0};
      std::vector<size_t> crop_shape  = {1, S.shape.at(1) - 20, S.shape.at(0)};

      auto [M0]        = unpack<1>(average(S, {1, s_.time_y_begin, s_.time_y_end}));
      std::tie(M0)     = unpack<1>(reshape(M0, {{1, M0.shape.at(1), M0.shape.at(0)}}));
      auto [M0_avg]    = unpack<1>(slide_avg(M0, {128, (size_t)s_.pp_accumulation}));
      std::tie(M0_avg) = unpack<1>(crop(M0_avg, {crop_origin, crop_shape}));
      std::tie(M0_avg) = unpack<1>(convert(M0_avg, {Target::U8, Strat::Scaled}));
      std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.gpu_out_size, 1, 1}));
      std::tie(M0_avg) = unpack<1>(memcpy(M0_avg, {Host}));
      std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.cpu_out_size, 1, 1}));
      xz_processed_display(M0_avg, {});
    }

    // -------------------------------------------------------------------------------------------------
    // YZ View Processing (S -> M0 - Processed YZ View)
    // -------------------------------------------------------------------------------------------------
    {
      std::vector<size_t> crop_origin = {0, 10, 0};
      std::vector<size_t> crop_shape  = {1, S.shape.at(1) - 20, S.shape.at(0)};

      auto [M0]        = unpack<1>(average(S, {2, s_.time_x_begin, s_.time_x_end}));
      std::tie(M0)     = unpack<1>(reshape(M0, {{1, M0.shape.at(1), M0.shape.at(0)}}));
      auto [M0_avg]    = unpack<1>(slide_avg(M0, {128, (size_t)s_.pp_accumulation}));
      std::tie(M0_avg) = unpack<1>(crop(M0_avg, {crop_origin, crop_shape}));
      std::tie(M0_avg) = unpack<1>(convert(M0_avg, {Target::U8, Strat::Scaled}));
      std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.gpu_out_size, 1, 1}));
      std::tie(M0_avg) = unpack<1>(memcpy(M0_avg, {Host}));
      std::tie(M0_avg) = unpack<1>(batched_queue(M0_avg, {s_.cpu_out_size, 1, 1}));
      yz_processed_display(M0_avg, {});
    }
  }

  // -------------------------------------------------------------------------------------------------
  // Shack-Hartmann View Processing (S -> SH - Shack-Hartmann View)
  // -------------------------------------------------------------------------------------------------

  if (true) {
    // TODO: start from u8 SH
    auto nb_subap = 3;
    auto subap_w  = S.shape.at(2) / nb_subap;
    auto subap_h  = S.shape.at(1) / nb_subap;
    subap_w       = S.shape.at(2);
    subap_h       = S.shape.at(1);
    auto tmp      = convert(H, {Target::F32, Strat::Modulus});
    auto [H]      = unpack<1>(tmp);

    // -------------------------------------------------------------------------------------------------
    // Time-Frequency Analysis (SH -> FH - Frequency Hologram -> FH_filt - Filtered Frequency
    // Hologram)
    // -------------------------------------------------------------------------------------------------
    auto [FH]      = unpack<1>(rfft(H, {0}));
    auto [FH_filt] = unpack<1>(slice_copy(FH, {{{111, 145}, {}, {}}}));

    // -------------------------------------------------------------------------------------------------
    // Shack-Hartmann processing (FH_filt -> FH_Qin (fresnel lens applied) -> H_sub (subaperture) ->
    // H_sub_prop -> S -> M0 (final display)
    // -------------------------------------------------------------------------------------------------

    auto nx          = S.shape.at(2);
    auto ny          = S.shape.at(1);
    auto [FH_Qin]    = unpack<1>(fresnel_qin({lam, dx, dy, z_prop, nx, ny}));
    std::tie(FH_Qin) = unpack<1>(mul(FH_filt, FH_Qin, {}));
    auto [FH_sub]    = unpack<1>(slice_copy(FH_Qin, {{{}, {0, subap_h, 1}, {0, subap_w, 1}}}));

    auto [S]  = unpack<1>(abs(FH_sub, {}));
    auto [M0] = unpack<1>(mean(S, {{0}, true}));

    std::tie(M0) = unpack<1>(convert(M0, {Target::U8, Strat::Scaled}));
    std::tie(M0) = unpack<1>(batched_queue(M0, {s_.gpu_out_size, 1, 1}));
    std::tie(M0) = unpack<1>(memcpy(M0, {Host}));
    std::tie(M0) = unpack<1>(batched_queue(M0, {s_.cpu_out_size, 1, 1}));
    shack_hartmann_display(M0, {});
  }

  return g_;
}

// -------------------------------------------------------------------------------------------------
// Task Wrappers Implementations
// -------------------------------------------------------------------------------------------------

#define DEFINE_SOURCE_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                    \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(SettingsType s) {                   \
    return make_source_sync_node(node_name_str, kind_str, kind_str, s);                            \
  }

#define DEFINE_UNARY_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                     \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(const TDesc &X, SettingsType s) {   \
    return make_unary_sync_node(node_name_str, kind_str, kind_str, X, s);                          \
  }

#define DEFINE_NARY_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                      \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(std::span<const TDesc> Xs,          \
                                                               SettingsType           s) {                   \
    return make_nary_sync_node(node_name_str, kind_str, kind_str, Xs, s);                          \
  }

#define DEFINE_UNARY_ASYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                    \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(const TDesc &X, SettingsType s) {   \
    return make_unary_async_node(node_name_str, kind_str, kind_str, X, s);                         \
  }

// clang-format off
DEFINE_SOURCE_SYNC_NODE(holofile_read,                          "source",                              "Holofile",                       holotask::sources::HolofileSettings)
DEFINE_SOURCE_SYNC_NODE(ametek_s710_euresys_coaxlink_octo,      "ametek_s710_euresys_coaxlink_octo",   "AmetekS710EuresysCoaxlinkOcto",  holotask::sources::AmetekS710EuresysCoaxlinkOctoSettings)
DEFINE_SOURCE_SYNC_NODE(ametek_s711_euresys_coaxlink_qsfp_plus, "ametek_s711_euresys_coaxlink_qsfp_+", "AmetekS711EuresysCoaxlinkQSFP+", holotask::sources::AmetekS711EuresysCoaxlinkQSFPSettings)
DEFINE_SOURCE_SYNC_NODE(fresnel_qin,                            "fresnel_qin",                         "FresnelQin",                     holotask::sources::FresnelQinSettings)
DEFINE_UNARY_SYNC_NODE (memcpy,                                 "memcpy",                              "Memcpy",                         holotask::syncs::MemcpySettings)
DEFINE_UNARY_SYNC_NODE (convert,                                "conversion",                          "Conversion",                     holotask::syncs::ConversionSettings)
DEFINE_UNARY_SYNC_NODE (pca,                                    "pca",                                 "Pca",                            holotask::syncs::PcaSettings)
DEFINE_UNARY_SYNC_NODE (stft,                                   "stft",                                "Stft",                           holotask::syncs::StftSettings)
DEFINE_UNARY_SYNC_NODE (filter_2d,                              "filter_2d",                           "Filter2D",                       holotask::syncs::Filter2DSettings)
DEFINE_UNARY_SYNC_NODE (fresnel_diffraction,                    "fresnel_diffraction",                 "FresnelDiffraction",             holotask::syncs::FresnelDiffractionSettings)
DEFINE_UNARY_SYNC_NODE (angular_spectrum,                       "angular_spectrum",                    "AngularSpectrum",                holotask::syncs::AngularSpectrumSettings)
DEFINE_UNARY_SYNC_NODE (reshape,                                "reshape",                             "Reshape",                        holotask::syncs::ReshapeSettings)
DEFINE_UNARY_SYNC_NODE (average,                                "average",                             "Average",                        holotask::syncs::AverageSettings)
DEFINE_UNARY_SYNC_NODE (convolution,                            "convolution",                         "Convolution",                    holotask::syncs::ConvolutionSettings)
DEFINE_UNARY_SYNC_NODE (crop,                                   "crop",                                "Crop",                           holotask::syncs::CropSettings)
DEFINE_UNARY_SYNC_NODE (fft_shift,                              "fft_shift",                           "FFTShift",                       holotask::syncs::FFTShiftSettings)
DEFINE_UNARY_SYNC_NODE (pct_clip,                               "pct_clip",                            "PctClip",                        holotask::syncs::PctClipSettings)
DEFINE_UNARY_SYNC_NODE (registration,                           "registration",                        "Registration",                   holotask::syncs::RegistrationSettings)
DEFINE_UNARY_SYNC_NODE (rotation,                               "rotation",                            "Rotation",                       holotask::syncs::RotationSettings)
DEFINE_UNARY_SYNC_NODE (xy_raw_display,                         "xy_raw_display",                      "DisplayTensorXYRaw",             tasks::sinks::DisplayTensorSettings)
DEFINE_UNARY_SYNC_NODE (xy_processed_display,                   "xy_processed_display",                "DisplayTensorXY",                tasks::sinks::DisplayTensorSettings)
DEFINE_UNARY_SYNC_NODE (xz_processed_display,                   "xz_processed_display",                "DisplayTensorXZ",                tasks::sinks::DisplayTensorSettings)
DEFINE_UNARY_SYNC_NODE (yz_processed_display,                   "yz_processed_display",                "DisplayTensorYZ",                tasks::sinks::DisplayTensorSettings)
DEFINE_UNARY_SYNC_NODE (shack_hartmann_display,                 "shack_hartmann_display",              "DisplayTensorShackHartmann",     tasks::sinks::DisplayTensorSettings)
DEFINE_NARY_SYNC_NODE  (concatenate,                            "concatenate",                         "Concatenate",                    holonp::ConcatenateSettings)
DEFINE_UNARY_SYNC_NODE (transpose,                              "transpose",                           "Transpose",                      holonp::TransposeSettings)
DEFINE_UNARY_SYNC_NODE (rfft,                                   "rfft",                                "RFFT",                           holonp::RFFTSettings)
DEFINE_UNARY_SYNC_NODE (slice_copy,                             "slice_copy",                          "SliceCopy",                      holonp::SliceCopySettings)
DEFINE_UNARY_SYNC_NODE (fft,                                    "fft",                                 "FFT",                            holonp::FFTSettings)
DEFINE_UNARY_SYNC_NODE (fft2,                                   "fft2",                                "FFT2",                           holonp::FFT2Settings)
DEFINE_UNARY_SYNC_NODE (fftshift,                               "fftshift",                            "FFTShiftNp",                     holonp::FFTShiftSettings)
DEFINE_UNARY_SYNC_NODE (abs,                                    "abs",                                 "Abs",                            holonp::AbsSettings)
DEFINE_UNARY_SYNC_NODE (mean,                                   "mean",                                "Mean",                           holonp::MeanSettings)
DEFINE_UNARY_ASYNC_NODE(batched_queue,                          "batch_queue",                         "BatchQueue",                     holotask::asyncs::BatchQueueSettings)
DEFINE_UNARY_ASYNC_NODE(slide_avg,                              "slide_avg",                           "SlidingAverage",                 holotask::asyncs::SlidingAverageSettings)
// clang-format on

std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::mul(const TDesc &A, const TDesc &B,
                                                         holonp::MulSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return make_nary_sync_node("mul", "Mul", "Mul", std::span<const TDesc>{inputs}, s);
}

std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::holofile_write(const TDesc &X, holotask::sinks::HolofileSettings s) {
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
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

// -------------------------------------------------------------------------------------------------
// Various Helpers
// -------------------------------------------------------------------------------------------------

holoflow::core::TDesc GraphBuilder_v2::TDesc::as_core() const {
  holoflow::core::TDesc t{};
  t.shape   = shape;
  t.dtype   = dtype;
  t.mem_loc = mem_loc;
  return t;
}

GraphBuilder_v2::TDesc GraphBuilder_v2::TDesc::from_core(const holoflow::core::TDesc &base) {
  TDesc t;
  static_cast<holoflow::core::TDesc &>(t) = base;
  return t;
}

std::vector<holoflow::core::TDesc> GraphBuilder_v2::to_core_descs(std::span<const TDesc> src) {
  std::vector<holoflow::core::TDesc> out;
  out.reserve(src.size());
  for (const auto &t : src) {
    out.push_back(t.as_core());
  }
  return out;
}

template <class InferResult>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::wrap_infer_outputs(std::string_view node_id, V vertex, const InferResult &infer) {
  std::vector<TDesc> ys;
  ys.reserve(infer.output_descs.size());
  int out_idx = 0;
  for (const auto &base : infer.output_descs) {
    auto y     = TDesc::from_core(base);
    y.producer = TDesc::Producer{
        .node_id = std::string{node_id},
        .out_idx = out_idx,
        .vertex  = vertex,
    };
    ys.push_back(std::move(y));
    ++out_idx;
  }
  return ys;
}

template <typename SettingsT>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::make_source_sync_node(std::string_view node_name, std::string_view kind,
                                       std::string_view reg_key, const SettingsT &s, bool debug) {
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);

  auto      &factory = reg_.get_sync(std::string{reg_key});
  const auto infer   = factory.infer(std::span<const holoflow::core::TDesc>{}, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

template <typename SettingsT>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::make_unary_sync_node(std::string_view node_name, std::string_view kind,
                                      std::string_view reg_key, const TDesc &X, const SettingsT &s,
                                      bool debug) {
  HOLOVIBES_CHECK(X.producer.has_value());
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, 0}, g_);

  auto      &factory     = reg_.get_sync(std::string{reg_key});
  const auto core_inputs = to_core_descs(std::span{&X, 1});
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

template <typename SettingsT>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::make_nary_sync_node(std::string_view node_name, std::string_view kind,
                                     std::string_view reg_key, std::span<const TDesc> inputs,
                                     const SettingsT &s, bool debug) {
  HOLOVIBES_CHECK(!inputs.empty());
  for (const auto &X : inputs) {
    HOLOVIBES_CHECK(X.producer.has_value());
  }
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &X = inputs[i];
    boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, static_cast<int>(i)}, g_);
  }

  auto      &factory     = reg_.get_sync(std::string{reg_key});
  const auto core_inputs = to_core_descs(inputs);
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

template <typename SettingsT>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::make_unary_async_node(std::string_view node_name, std::string_view kind,
                                       std::string_view reg_key, const TDesc &X, const SettingsT &s,
                                       bool debug) {
  HOLOVIBES_CHECK(X.producer.has_value());
  HOLOVIBES_CHECK(reg_.is_async_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, 0}, g_);

  auto      &factory     = reg_.get_async(std::string{reg_key});
  const auto core_inputs = to_core_descs(std::span{&X, 1});
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

} // namespace holovibes::pipeline
