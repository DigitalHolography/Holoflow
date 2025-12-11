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

namespace holovibes::pipeline {

GraphBuilder_v2::GraphBuilder_v2(const Settings &settings, holoflow::core::Registry &registry) :
    s_(settings),
    reg_(registry) {}

holoflow::core::GraphSpec GraphBuilder_v2::build() {
  logger()->info("[GraphBuilder_v2::build] Building graph spec...");

  auto lam    = s_.spacial_lambda;
  auto dx     = s_.spacial_pixel_size;
  auto dy     = s_.spacial_pixel_size;
  auto z_prop = s_.spacial_z;

  TDesc I0;
  TDesc FH;
  TDesc FH_prop;
  TDesc S;

  if (s_.import_source == ImportSource::HOLOFILE) {
    std::tie(I0) = unpack<1>(holofile_read({s_.load_path.string(),
                                            load_method_map_.at(s_.load_method),
                                            s_.load_begin,
                                            s_.load_end,
                                            s_.load_batch}));
  }

  else {
    HOLOVIBES_UNREACHABLE();
  }

  if (s_.load_method != LoadMethod::LOAD_IN_GPU) {
    std::tie(I0) = unpack<1>(memcpy(I0, {tasks::syncs::MemcpySettings::Target::Device}));
    std::tie(I0) = unpack<1>(batched_queue(I0, {s_.gpu_in_size, s_.time_window, s_.time_window}));
  }

  using Target = tasks::syncs::ConversionSettings::Target;
  using Strat  = tasks::syncs::ConversionSettings::Strategy;
  std::tie(I0) = unpack<1>(convert(I0, {Target::F32, Strat::Real}));

  if (s_.time_method == TimeMethod::SHORT_TIME_FOURIER) {
    std::tie(FH) = unpack<1>(stft(I0, {}));
  }

  else if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS && !s_.view_3d_cuts) {
    std::tie(FH) = unpack<1>(pca(I0, {s_.time_z_begin, s_.time_z_end}));
    std::tie(FH) = unpack<1>(convert(FH, {Target::CF32, Strat::Real}));
  }

  else if (s_.time_method == TimeMethod::PRINCIPAL_COMPONENT_ANALYSIS && s_.view_3d_cuts) {
    std::tie(FH) = unpack<1>(pca(I0, {0, s_.time_window}));
    std::tie(FH) = unpack<1>(convert(FH, {Target::CF32, Strat::Real}));
  }

  if (s_.spacial_method == SpacialMethod::FRESNEL_DIFFRACTION) {
    std::tie(FH_prop) = unpack<1>(fresnel_diffraction(FH, {lam, dx, dy, z_prop}));
  }

  else if (s_.spacial_method == SpacialMethod::ANGULAR_SPECTRUM) {
    std::tie(FH_prop) = unpack<1>(angular_spectrum(FH, {lam, dx, dy, z_prop, std::nullopt}));
  }

  if (s_.filter_2d) {
    std::tie(FH_prop) = unpack<1>(filter_2d(
        FH_prop,
        {s_.filter_r_inner, s_.filter_r_outer, s_.filter_smooth_inner, s_.filter_smooth_outer}));
  }

  std::tie(S) = unpack<1>(convert(FH_prop, {Target::F32, Strat::Modulus}));

  // XY branch
  {
    TDesc M0;

    std::tie(M0) = unpack<1>(average(S, {0, 0, (int)S.shape.at(0)}));
  }

  logger()->debug("[GraphBuilder_v2::build] Graph spec built successfully");
  return g_;
}

} // namespace holovibes::pipeline

#define DEFINE_SOURCE_SYNC_NODE(fn_name, node_name_str, kind_str, reg_key_str, SettingsType)       \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(SettingsType s) {                   \
    return make_source_sync_node(node_name_str, kind_str, reg_key_str, s);                         \
  }

#define DEFINE_UNARY_SYNC_NODE(fn_name, node_name_str, kind_str, reg_key_str, SettingsType)        \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(const TDesc &X, SettingsType s) {   \
    return make_unary_sync_node(node_name_str, kind_str, reg_key_str, X, s);                       \
  }

#define DEFINE_UNARY_ASYNC_NODE(fn_name, node_name_str, kind_str, reg_key_str, SettingsType)       \
  std::vector<GraphBuilder_v2::TDesc> GraphBuilder_v2::fn_name(const TDesc &X, SettingsType s) {   \
    return make_unary_async_node(node_name_str, kind_str, reg_key_str, X, s);                      \
  }

namespace holovibes::pipeline {

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
GraphBuilder_v2::make_source_sync_node(std::string_view node_name,
                                       std::string_view kind,
                                       std::string_view reg_key,
                                       const SettingsT &s) {
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name},
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = true,
  };

  auto v = boost::add_vertex(node_spec, g_);

  auto      &factory = reg_.get_sync(std::string{reg_key});
  const auto infer   = factory.infer(std::span<const holoflow::core::TDesc>{}, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

template <typename SettingsT>
std::vector<GraphBuilder_v2::TDesc>
GraphBuilder_v2::make_unary_sync_node(std::string_view node_name,
                                      std::string_view kind,
                                      std::string_view reg_key,
                                      const TDesc     &X,
                                      const SettingsT &s) {
  HOLOVIBES_CHECK(X.producer.has_value());
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name},
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = true,
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
GraphBuilder_v2::make_unary_async_node(std::string_view node_name,
                                       std::string_view kind,
                                       std::string_view reg_key,
                                       const TDesc     &X,
                                       const SettingsT &s) {
  HOLOVIBES_CHECK(X.producer.has_value());
  HOLOVIBES_CHECK(reg_.is_async_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name},
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = true,
  };

  auto v = boost::add_vertex(node_spec, g_);
  boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, 0}, g_);

  auto      &factory     = reg_.get_async(std::string{reg_key});
  const auto core_inputs = to_core_descs(std::span{&X, 1});
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

DEFINE_SOURCE_SYNC_NODE(holofile_read,
                        "source",
                        "Holofile",
                        "Holofile",
                        tasks::sources::HolofileSettings)

DEFINE_SOURCE_SYNC_NODE(ametek_s710_euresys_coaxlink_octo,
                        "ametek_s710_euresys_coaxlink_octo",
                        "AmetekS710EuresysCoaxlinkOcto",
                        "AmetekS710EuresysCoaxlinkOcto",
                        tasks::sources::AmetekS710EuresysCoaxlinkOctoSettings)

DEFINE_SOURCE_SYNC_NODE(ametek_s711_euresys_coaxlink_qsfp_plus,
                        "ametek_s711_euresys_coaxlink_qsfp_plus",
                        "AmetekS711EuresysCoaxlinkQSFPPlus",
                        "AmetekS711EuresysCoaxlinkQSFPPlus",
                        tasks::sources::AmetekS711EuresysCoaxlinkQSFPSettings)

DEFINE_UNARY_SYNC_NODE(memcpy, "memcpy", "Memcpy", "Memcpy", tasks::syncs::MemcpySettings)

DEFINE_UNARY_SYNC_NODE(convert,
                       "conversion",
                       "Conversion",
                       "Conversion",
                       tasks::syncs::ConversionSettings)

DEFINE_UNARY_SYNC_NODE(pca, "pca", "Pca", "Pca", tasks::syncs::PcaSettings)

DEFINE_UNARY_SYNC_NODE(stft, "stft", "Stft", "Stft", tasks::syncs::StftSettings)

DEFINE_UNARY_SYNC_NODE(filter_2d,
                       "filter_2d",
                       "Filter2D",
                       "Filter2D",
                       tasks::syncs::Filter2DSettings)

DEFINE_UNARY_SYNC_NODE(fresnel_diffraction,
                       "fresnel_diffraction",
                       "FresnelDiffraction",
                       "FresnelDiffraction",
                       tasks::syncs::FresnelDiffractionSettings)

DEFINE_UNARY_SYNC_NODE(angular_spectrum,
                       "angular_spectrum",
                       "AngularSpectrum",
                       "AngularSpectrum",
                       tasks::syncs::AngularSpectrumSettings)

DEFINE_UNARY_SYNC_NODE(reshape, "reshape", "Reshape", "Reshape", tasks::syncs::ReshapeSettings)

DEFINE_UNARY_SYNC_NODE(average, "average", "Average", "Average", tasks::syncs::AverageSettings)

DEFINE_UNARY_SYNC_NODE(convolution,
                       "convolution",
                       "Convolution",
                       "Convolution",
                       tasks::syncs::ConvolutionSettings)

DEFINE_UNARY_SYNC_NODE(crop, "crop", "Crop", "Crop", tasks::syncs::CropSettings)

DEFINE_UNARY_SYNC_NODE(fft_shift,
                       "fft_shift",
                       "FftShift",
                       "FftShift",
                       tasks::syncs::FFTShiftSettings)

DEFINE_UNARY_SYNC_NODE(pct_clip, "pct_clip", "PctClip", "PctClip", tasks::syncs::PctClipSettings)

DEFINE_UNARY_SYNC_NODE(registration,
                       "registration",
                       "Registration",
                       "Registration",
                       tasks::syncs::RegistrationSettings)

DEFINE_UNARY_SYNC_NODE(rotation, "rotation", "Rotation", "Rotation", tasks::syncs::RotationSettings)

DEFINE_UNARY_SYNC_NODE(xy_raw_display,
                       "xy_raw_display",
                       "DisplayTensorXYRaw",
                       "DisplayTensorXYRaw",
                       tasks::sinks::DisplayTensorSettings)

DEFINE_UNARY_SYNC_NODE(xy_processed_display,
                       "xy_processed_display",
                       "DisplayTensorXY",
                       "DisplayTensorXY",
                       tasks::sinks::DisplayTensorSettings)

DEFINE_UNARY_SYNC_NODE(xz_processed_display,
                       "xz_processed_display",
                       "DisplayTensorXZ",
                       "DisplayTensorXZ",
                       tasks::sinks::DisplayTensorSettings)

DEFINE_UNARY_SYNC_NODE(yz_processed_display,
                       "yz_processed_display",
                       "DisplayTensorYZ",
                       "DisplayTensorYZ",
                       tasks::sinks::DisplayTensorSettings)

DEFINE_UNARY_SYNC_NODE(holofile_write,
                       "holofile_write",
                       "HolofileWrite",
                       "HolofileWrite",
                       tasks::sinks::HolofileSettings)

DEFINE_UNARY_ASYNC_NODE(batched_queue,
                        "batch_queue",
                        "BatchQueue",
                        "BatchQueue",
                        tasks::asyncs::BatchQueueSettings)

DEFINE_UNARY_ASYNC_NODE(slide_avg,
                        "slide_avg",
                        "SlidingAverage",
                        "SlidingAverage",
                        tasks::asyncs::SlidingAverageSettings)

} // namespace holovibes::pipeline
