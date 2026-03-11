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

#include "graph_builder_v2.hh"

#include <spdlog/fmt/ranges.h>

#include "bug.hh"
#include "logger.hh"
#include "settings_loader.hh"

namespace holovibes::pipeline {

// -------------------------------------------------------------------------------------------------
// GraphBuilder Initialization & Main Pipeline Logic
// -------------------------------------------------------------------------------------------------

GraphBuilder_v2::GraphBuilder_v2(const Settings &settings, holoflow::core::Registry &registry)
    : s_(settings), reg_(registry) {}

holoflow::core::GraphSpec GraphBuilder_v2::build() {
  TDesc H = build_acquisition();

  if (s_.recording_method == RecordingMethod::RAW) {
    build_raw_record(H);
  }

  if (s_.raw_view || s_.view_type == ViewType::RAW) {
    bool should_exit = build_raw_view(H);
    if (should_exit) {
      return g_;
    }
  }

  H = build_preprocessing(H);

  TDesc FH = build_time_frequency_analysis(H);

  if (s_.autofocus_enabled) {
    FH = build_shack_hartmann(FH);
  }

  TDesc FH_z = build_spatial_propagation(FH);

  build_xy_view(FH_z);

  if (s_.view_3d_cuts) {
    build_3d_cuts(FH_z);
  }

  return g_;
}

// -------------------------------------------------------------------------------------------------
// Pipeline Stage Subfunctions
// -------------------------------------------------------------------------------------------------

GraphBuilder_v2::TDesc GraphBuilder_v2::build_acquisition() {
  auto cam_path = s_.camera_config_path.string();

  if (s_.import_source == ImportSource::HOLOFILE) {
    return holofile_read({
        s_.load_path.string(),
        load_method_map_.at(s_.load_method),
        s_.load_begin,
        s_.load_end,
        s_.load_batch,
    });
  } else if (s_.import_source == ImportSource::AMETEK_S710_EURESYS_COAXLINK_OCTO) {
    return ametek_s710_euresys_coaxlink_octo({cam_path});
  } else if (s_.import_source == ImportSource::AMETEK_S711_EURESYS_COAXLINK_QSFP) {
    return ametek_s711_euresys_coaxlink_qsfp_plus({cam_path});
  }

  HOLOVIBES_UNREACHABLE();
}

void GraphBuilder_v2::build_raw_record(const TDesc &H) {
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  auto H_rec = memcpy(H, {Host});
  H_rec      = batched_queue(H_rec, {s_.recording_count, s_.time_window, s_.time_window});

  holofile_write(H_rec, {s_.recording_path.string(), s_.recording_count, settings_to_old_json(s_)});
}

bool GraphBuilder_v2::build_raw_view(const TDesc &H) {
  auto Host = holotask::syncs::MemcpySettings::Target::Host;

  int64_t new_y = static_cast<int64_t>(H.shape.at(1));
  int64_t new_x = static_cast<int64_t>(H.shape.at(2));

  auto H_disp     = memcpy(H, {Host});
  auto H_view     = batched_queue(H_disp, {s_.cpu_out_size, 1, 1});
  auto H_reshaped = reshape(H_view, {{1, new_y, new_x}, true});

  if (s_.raw_view) {
    xy_raw_display(H_reshaped, {});
  }

  if (s_.view_type == ViewType::RAW) {
    xy_processed_display(H_reshaped, {});
    return true; // Signal to caller to exit pipeline early
  }

  return false;
}

GraphBuilder_v2::TDesc GraphBuilder_v2::build_preprocessing(TDesc H) {
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Device  = holotask::syncs::MemcpySettings::Target::Device;

  if (s_.load_method != LoadMethod::LOAD_IN_GPU) {
    H = memcpy(H, {Device});
    H = batched_queue(H, {s_.gpu_in_size, s_.time_window, s_.time_window});
  }

  return convert(H, {Target::F32, Strat::Real});
}

GraphBuilder_v2::TDesc GraphBuilder_v2::build_time_frequency_analysis(TDesc H) {
  TDesc FH;
  if (s_.time_method == TimeMethod::SHORT_TIME_FOURIER) {
    FH = rfft(H, {0});

    // Optimization: slice relevant components early
    if (!s_.view_3d_cuts) {
      FH = slice(FH, {{holonp::SliceRange{s_.time_z_begin, s_.time_z_end}, {}, {}}});
      FH = copy(FH, {});
    }
  }

  else if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS) {
    // Optimization: slice relevant components early
    int z0 = s_.view_3d_cuts ? 0 : s_.time_z_begin;
    int z1 = s_.view_3d_cuts ? static_cast<int>(H.shape.at(0)) : s_.time_z_end;
    FH     = pca(H, {z0, z1, 1});
  }

  else {
    throw std::logic_error{"Time method is currently not supported in GraphBuilder_v2"};
  }

  int64_t Nz = static_cast<int64_t>(FH.shape.at(0));
  int64_t Ny = static_cast<int64_t>(FH.shape.at(1));
  int64_t Nx = static_cast<int64_t>(FH.shape.at(2));

  FH = reshape(FH, {{1, Nz, Ny, Nx}, false});
  return batched_queue(FH, {s_.pp_accumulation * 2, s_.pp_accumulation, s_.pp_accumulation});
}

GraphBuilder_v2::TDesc GraphBuilder_v2::build_shack_hartmann(TDesc FH) {
  auto nb_subap = 5ULL;
  auto lam      = s_.spacial_lambda;
  auto dx       = s_.spacial_pixel_size;
  auto dy       = s_.spacial_pixel_size;
  auto z_prop   = s_.spacial_z;

  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Host    = holotask::syncs::MemcpySettings::Target::Host;
  auto Device  = holotask::syncs::MemcpySettings::Target::Device;

  // Spatial Cropping & Fresnel Lens Application
  auto subap_w = FH.shape.at(3) / nb_subap;
  auto subap_h = FH.shape.at(2) / nb_subap;
  auto valid_w = subap_w * nb_subap;
  auto valid_h = subap_h * nb_subap;

  holonp::SliceRange x_crop{0, valid_w};
  holonp::SliceRange y_crop{0, valid_h};

  auto FH_cropped = slice(FH, {{{}, {}, y_crop, x_crop}});
  auto n_freq     = FH_cropped.shape.at(1);
  auto Qin        = fresnel_qin(asarray({z_prop}), {lam, dx, dy, valid_w, valid_h});
  auto FH_Qin     = mul(FH_cropped, Qin, {});

  // Organize into sub-aperture groups:
  // (accumulation, freq, subap_y, subap_x, subap_h, subap_w)
  auto FH_6d      = reshape(FH_Qin, {{
                                   (int64_t)s_.pp_accumulation,
                                   (int64_t)n_freq,
                                   (int64_t)nb_subap,
                                   (int64_t)subap_h,
                                   (int64_t)nb_subap,
                                   (int64_t)subap_w,
                               }});
  auto FH_grouped = transpose(FH_6d, {{0, 1, 2, 4, 3, 5}});

  // Sub-aperture Processing
  auto FH_prop = fft2(FH_grouped, {{-2, -1}});
  auto M0      = mean_abs(FH_prop, {{1}, false});
  M0           = mean(M0, {{0}, true});
  M0           = fftshift(M0, {{-2, -1}});

  // Cross Correlation with Reference
  int64_t sy_ref = nb_subap / 2;
  int64_t sx_ref = nb_subap / 2;
  auto    M0_ref = slice(M0, {{{}, sy_ref, sx_ref, {}, {}}});

  auto F_mov      = rfft2(M0, {{-2, -1}});
  auto F_ref      = rfft2(M0_ref, {{-2, -1}});
  auto F_ref_conj = conj(F_ref, {});

  auto F_xcorr     = mul(F_mov, F_ref_conj, {});
  auto F_xcorr_abs = abs(F_xcorr, {});
  F_xcorr          = div(F_xcorr, F_xcorr_abs, {});

  auto xcorr = irfft2(F_xcorr, {{-2, -1}});

  // Shack-Hartmann Output Processing
  int64_t h = static_cast<int64_t>(valid_h);
  int64_t w = static_cast<int64_t>(valid_w);

  auto M0_sh_disp = normalize(M0, {{-2, -1}, 0.0f, 255.0f});
  M0_sh_disp      = transpose(M0_sh_disp, {{0, 1, 3, 2, 4}});
  M0_sh_disp      = reshape(M0_sh_disp, {{1, h, w}});
  M0_sh_disp      = convert(M0_sh_disp, {Target::U8, Strat::Scaled});
  M0_sh_disp      = memcpy(M0_sh_disp, {Host});
  M0_sh_disp      = batched_queue(M0_sh_disp, {s_.cpu_out_size, 1, 1});
  shack_hartmann_display(M0_sh_disp, {});

  auto xcorr_flattened = fftshift(xcorr, {{-2, -1}});
  xcorr_flattened      = normalize(xcorr_flattened, {{-2, -1}, 0.0f, 255.0f});
  xcorr_flattened      = convert(xcorr_flattened, {Target::U8, Strat::Scaled});
  xcorr_flattened      = transpose(xcorr_flattened, {{0, 1, 3, 2, 4}});
  xcorr_flattened      = reshape(xcorr_flattened, {{1, h, w}});
  xcorr_flattened      = batched_queue(xcorr_flattened, {s_.cpu_out_size, 1, 1});
  shack_hartmann_xcorr_display(xcorr_flattened, {});

  // Zernike & Phase Correction
  int ny = static_cast<int>(FH.shape.at(2));
  int nx = static_cast<int>(FH.shape.at(3));

  if (!s_.autofocus_zernike_orders.empty()) {
    auto xcorr_zernike = fftshift(xcorr, {{-2, -1}});
    xcorr_zernike      = normalize(xcorr_zernike, {{-2, -1}, 0.0f, 255.0f});
    xcorr_zernike      = memcpy(xcorr_zernike, {Host});

    auto zernike_coeffs = zernike(xcorr_zernike, {
                                                     s_.autofocus_zernike_orders,
                                                     lam,
                                                     dx,
                                                     dy,
                                                     z_prop,
                                                 });
    zernike_coefficients_display(zernike_coeffs, {s_.autofocus_zernike_orders});

    auto phase     = zernike_phase(zernike_coeffs, {s_.autofocus_zernike_orders, ny, nx});
    auto phase_gpu = memcpy(phase, {Device});
    FH             = correct_phase(FH, phase_gpu, {});

    auto phase_disp = wrap2pi(phase_gpu, {});
    phase_disp      = reshape(phase_disp, {{1, ny, nx}});
    phase_disp      = batched_queue(phase_disp, {s_.cpu_out_size, 1, 1});
    zernike_phase_display(phase_disp, {});
  }

  // When no Zernike orders are specified, still display an empty phase map for consistency
  else {
    auto empty_phase = zeros({
        {1, static_cast<size_t>(ny), static_cast<size_t>(nx)},
        holoflow::core::DType::F32,
    });
    FH               = correct_phase(FH, empty_phase, {});
    zernike_phase_display(empty_phase, {});
  }

  return FH;
}

GraphBuilder_v2::TDesc GraphBuilder_v2::build_spatial_propagation(const TDesc &FH) {
  if (s_.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION) {
    return fresnel_diffraction(FH, {
                                       s_.spacial_lambda,
                                       s_.spacial_pixel_size,
                                       s_.spacial_pixel_size,
                                       s_.spacial_z,
                                       {-2, -1},
                                   });
  }

  else if (s_.spacial_method == SpacialMethod::ANGULAR_SPECTRUM) {
    throw std::logic_error{"Angular Spectrum is currently not supported in GraphBuilder_v2"};
  }

  throw std::logic_error{"Spacial method is currently not supported in GraphBuilder_v2"};
}

void GraphBuilder_v2::build_xy_view(const TDesc &FH_z) {
  using Target = holotask::syncs::ConversionSettings::Target;
  using Strat  = holotask::syncs::ConversionSettings::Strategy;
  auto Host    = holotask::syncs::MemcpySettings::Target::Host;

  auto M0 = mean_abs(FH_z, {{-3}, false});

  if (s_.pp_fft_shift) {
    M0 = fftshift(M0, {{-2, -1}});
  }

  if (s_.pp_registration) {
    throw std::logic_error{"Registration is currently not supported"};
  }

  auto M0_avg = mean(M0, {{0}, true});

  if (s_.pp_convolution) {
    throw std::logic_error{"Convolution is currently not supported"};
  }

  if (s_.pp_pctclip) {
    M0_avg = pct_clip(M0_avg, {s_.pp_pctclip_lower,
                               s_.pp_pctclip_upper,
                               {0.5f, 0.5f, s_.pp_pctclip_radius, s_.pp_pctclip_radius, 0.0f}});
  }

  M0_avg = convert(M0_avg, {Target::U8, Strat::Scaled});
  M0_avg = batched_queue(M0_avg, {s_.gpu_out_size, 1, 1});
  M0_avg = memcpy(M0_avg, {Host});
  M0_avg = batched_queue(M0_avg, {s_.cpu_out_size, 1, 1});
  xy_processed_display(M0_avg, {});

  if (s_.recording_method == RecordingMethod::PROCESSED) {
    auto M0_rec = memcpy(M0_avg, {Host});
    M0_rec      = batched_queue(M0_rec, {s_.cpu_out_size, 1, 1});
    holofile_write(M0_rec,
                   {s_.recording_path.string(), s_.recording_count, settings_to_old_json(s_)});
  }
}

void GraphBuilder_v2::build_3d_cuts(const TDesc &FH_z) {
  (void)FH_z;
  throw std::logic_error{"3D cuts are currently not supported in GraphBuilder_v2"};
}

// -------------------------------------------------------------------------------------------------
// Internal Core Templates and Helpers
// -------------------------------------------------------------------------------------------------

holoflow::core::TDesc GraphBuilder_v2::TDesc::as_core() const {
  holoflow::core::TDesc t{};
  t.shape   = shape;
  t.dtype   = dtype;
  t.mem_loc = mem_loc;
  t.strides = strides;
  t.offset  = offset;
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

  auto       v       = boost::add_vertex(node_spec, g_);
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

// -------------------------------------------------------------------------------------------------
// Task Wrappers Implementations
// -------------------------------------------------------------------------------------------------

#define DEFINE_SOURCE_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                    \
  GraphBuilder_v2::TDesc GraphBuilder_v2::fn_name(SettingsType s) {                                \
    return std::move(make_source_sync_node(node_name_str, kind_str, kind_str, s).at(0));           \
  }

#define DEFINE_UNARY_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                     \
  GraphBuilder_v2::TDesc GraphBuilder_v2::fn_name(const TDesc &X, SettingsType s) {                \
    return std::move(make_unary_sync_node(node_name_str, kind_str, kind_str, X, s).at(0));         \
  }

#define DEFINE_SINK_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                      \
  void GraphBuilder_v2::fn_name(const TDesc &X, SettingsType s) {                                  \
    make_unary_sync_node(node_name_str, kind_str, kind_str, X, s);                                 \
  }

#define DEFINE_NARY_SYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                      \
  GraphBuilder_v2::TDesc GraphBuilder_v2::fn_name(std::span<const TDesc> Xs, SettingsType s) {     \
    return std::move(make_nary_sync_node(node_name_str, kind_str, kind_str, Xs, s).at(0));         \
  }

#define DEFINE_UNARY_ASYNC_NODE(fn_name, node_name_str, kind_str, SettingsType)                    \
  GraphBuilder_v2::TDesc GraphBuilder_v2::fn_name(const TDesc &X, SettingsType s) {                \
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

DEFINE_SINK_SYNC_NODE  (xy_raw_display,                 "xy_raw_display",                      "DisplayTensorXYRaw",              tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (xy_processed_display,           "xy_processed_display",                "DisplayTensorXY",                 tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (xz_processed_display,           "xz_processed_display",                "DisplayTensorXZ",                 tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (yz_processed_display,           "yz_processed_display",                "DisplayTensorYZ",                 tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (shack_hartmann_display,         "shack_hartmann_display",              "DisplayTensorShackHartmann",      tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (shack_hartmann_xcorr_display,   "shack_hartmann_xcorr_display",        "DisplayTensorShackHartmannXcorr", tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (zernike_phase_display,          "zernike_phase_display",               "DisplayTensorZernikePhase",       tasks::sinks::DisplayTensorSettings)
DEFINE_SINK_SYNC_NODE  (zernike_coefficients_display,   "zernike_coefficients_display",        "DisplayZernikeCoefficients",      tasks::sinks::DisplayZernikeCoefficientsSettings)

DEFINE_NARY_SYNC_NODE  (concatenate,                    "concatenate",                         "Concatenate",                     holonp::ConcatenateSettings)
DEFINE_UNARY_SYNC_NODE (transpose,                      "transpose",                           "Transpose",                       holonp::TransposeSettings)
DEFINE_UNARY_SYNC_NODE (conj,                           "conj",                                "Conj",                            holonp::ConjSettings)
DEFINE_UNARY_SYNC_NODE (rfft,                           "rfft",                                "RFFT",                            holonp::RFFTSettings)
DEFINE_UNARY_SYNC_NODE (rfft2,                          "rfft2",                               "RFFT2",                           holonp::RFFT2Settings)
DEFINE_UNARY_SYNC_NODE (irfft2,                         "irfft2",                              "IRFFT2",                          holonp::IRFFT2Settings)
DEFINE_UNARY_SYNC_NODE (slice,                          "slice",                               "Slice",                           holonp::SliceSettings)
DEFINE_UNARY_SYNC_NODE (fft,                            "fft",                                 "FFT",                             holonp::FFTSettings)
DEFINE_UNARY_SYNC_NODE (fft2,                           "fft2",                                "FFT2",                            holonp::FFT2Settings)
DEFINE_UNARY_SYNC_NODE (fftshift,                       "fftshift",                            "FFTShiftNp",                      holonp::FFTShiftSettings)
DEFINE_UNARY_SYNC_NODE (abs,                            "abs",                                 "Abs",                             holonp::AbsSettings)
DEFINE_UNARY_SYNC_NODE (mean,                           "mean",                                "Mean",                            holonp::MeanSettings)
DEFINE_UNARY_SYNC_NODE (mean_abs,                       "mean_abs",                            "MeanAbs",                         holonp::MeanAbsSettings)
DEFINE_UNARY_SYNC_NODE (min,                            "min",                                 "Min",                             holonp::MinSettings)
DEFINE_UNARY_SYNC_NODE (max,                            "max",                                 "Max",                             holonp::MaxSettings)
DEFINE_UNARY_SYNC_NODE (normalize,                      "normalize",                           "Normalize",                       holonp::NormalizeSettings)
DEFINE_UNARY_ASYNC_NODE(batched_queue,                  "batch_queue",                         "BatchQueue",                      holotask::asyncs::BatchQueueSettings)
DEFINE_UNARY_ASYNC_NODE(slide_avg,                      "slide_avg",                           "SlidingAverage",                  holotask::asyncs::SlidingAverageSettings)
// clang-format on

GraphBuilder_v2::TDesc GraphBuilder_v2::mul(const TDesc &A, const TDesc &B, holonp::MulSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return std::move(
      make_nary_sync_node("mul", "Mul", "Mul", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilder_v2::TDesc GraphBuilder_v2::add(const TDesc &A, const TDesc &B, holonp::AddSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return std::move(
      make_nary_sync_node("add", "Add", "Add", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilder_v2::TDesc GraphBuilder_v2::div(const TDesc &A, const TDesc &B, holonp::DivSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return std::move(
      make_nary_sync_node("div", "Div", "Div", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilder_v2::TDesc GraphBuilder_v2::sub(const TDesc &A, const TDesc &B, holonp::SubSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return std::move(
      make_nary_sync_node("sub", "Sub", "Sub", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilder_v2::TDesc GraphBuilder_v2::correct_phase(const TDesc &X, const TDesc &PhaseMask,
                                                      holotask::syncs::CorrectPhaseSettings s) {
  std::array<TDesc, 2> inputs{X, PhaseMask};
  return std::move(make_nary_sync_node("correct_phase", "CorrectPhase", "CorrectPhase",
                                       std::span<const TDesc>{inputs}, s)
                       .at(0));
}

GraphBuilder_v2::TDesc GraphBuilder_v2::equal(const TDesc &A, const TDesc &B,
                                              holonp::EqualSettings s) {
  std::array<TDesc, 2> inputs{A, B};
  return std::move(
      make_nary_sync_node("equal", "Equal", "Equal", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilder_v2::TDesc GraphBuilder_v2::where(const TDesc &Cond, const TDesc &X, const TDesc &Y,
                                              holonp::WhereSettings s) {
  std::array<TDesc, 3> inputs{Cond, X, Y};
  return std::move(
      make_nary_sync_node("where", "Where", "Where", std::span<const TDesc>{inputs}, s).at(0));
}

GraphBuilder_v2::TDesc GraphBuilder_v2::assign(const TDesc &X, const TDesc &Y,
                                               holonp::AssignSettings s) {
  std::array<TDesc, 2> inputs{X, Y};
  return std::move(
      make_nary_sync_node("assign", "Assign", "Assign", std::span<const TDesc>{inputs}, s).at(0));
}

void GraphBuilder_v2::holofile_write(const TDesc &X, holotask::sinks::HolofileSettings s) {
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