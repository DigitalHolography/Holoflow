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

#include <array>
#include <map>
#include <optional>
#include <ranges>
#include <span>
#include <stack>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/core/tasks.hh"
#include "holotask/asyncs/batch_queue.hh"
#include "holotask/asyncs/slide_avg.hh"
#include "holotask/sinks/holofile.hh"
#include "holotask/sources/ametek_s710_euresys_coaxlink_octo.hh"
#include "holotask/sources/ametek_s711_euresys_coaxlink_qsfp+.hh"
#include "holotask/sources/holofile.hh"
#include "holotask/syncs/angular_spectrum.hh"
#include "holotask/syncs/average.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/convolution.hh"
#include "holotask/syncs/crop.hh"
#include "holotask/syncs/fft_shift.hh"
#include "holotask/syncs/filter2d.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/memcpy.hh"
#include "holotask/syncs/pca.hh"
#include "holotask/syncs/pct_clip.hh"
#include "holotask/syncs/registration.hh"
#include "holotask/syncs/reshape.hh"
#include "holotask/syncs/rotation.hh"
#include "holotask/syncs/stft.hh"
#include "pipeline/settings.hh"
#include "tasks/sinks/display_tensor.hh"

namespace holovibes::pipeline {

class GraphBuilder_v2 {
public:
  GraphBuilder_v2(const Settings &settings, holoflow::core::Registry &registry);

  holoflow::core::GraphSpec build();

private:
  using V      = holoflow::core::GraphSpec::vertex_descriptor;
  using NodeId = std::string;

  class TDesc : public holoflow::core::TDesc {
  public:
    struct Producer {
      NodeId node_id;
      int    out_idx;
      V      vertex;
    };

    struct Consumer {
      NodeId node_id;
      int    in_idx;
      V      vertex;
    };

    std::optional<Producer> producer;
    std::vector<Consumer>   consumers;

    [[nodiscard]] holoflow::core::TDesc as_core() const;

    [[nodiscard]] static TDesc from_core(const holoflow::core::TDesc &base);
  };

  [[nodiscard]] static std::vector<holoflow::core::TDesc> to_core_descs(std::span<const TDesc> src);

  template <class InferResult>
  [[nodiscard]] static std::vector<TDesc> wrap_infer_outputs(std::string_view node_id, V vertex,
                                                             const InferResult &infer);

  template <std::size_t N, class R>
    requires std::ranges::contiguous_range<R> &&
             std::same_as<std::remove_cvref_t<std::ranges::range_value_t<R>>, TDesc>
  [[nodiscard]] auto unpack(R &&r) {
    auto s = std::span{r};
    if (s.size() != N) {
      throw std::logic_error{"GraphBuilder_v2::unpack size mismatch"};
    }
    using T = TDesc;
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return std::make_tuple(s[I]...);
    }(std::make_index_sequence<N>{});
  }

  template <typename SettingsT>
  std::vector<TDesc> make_source_sync_node(std::string_view node_name, std::string_view kind,
                                           std::string_view reg_key, const SettingsT &s,
                                           bool debug = true);

  template <typename SettingsT>
  std::vector<TDesc> make_unary_sync_node(std::string_view node_name, std::string_view kind,
                                          std::string_view reg_key, const TDesc &X,
                                          const SettingsT &s, bool debug = true);

  template <typename SettingsT>
  std::vector<TDesc> make_unary_async_node(std::string_view node_name, std::string_view kind,
                                           std::string_view reg_key, const TDesc &X,
                                           const SettingsT &s, bool debug = true);

  Settings                  s_;
  holoflow::core::Registry &reg_;

  holoflow::core::GraphSpec g_;
  std::stack<std::string>   scope_;
  size_t                    unique_id_counter_ = 0;

private:
  std::map<LoadMethod, holotask::sources::HolofileSettings::LoadKind> load_method_map_{
      {LoadMethod::READ_LIVE, holotask::sources::HolofileSettings::LoadKind::Live},
      {LoadMethod::LOAD_IN_CPU, holotask::sources::HolofileSettings::LoadKind::CPUCached},
      {LoadMethod::LOAD_IN_GPU, holotask::sources::HolofileSettings::LoadKind::GPUCached},
  };

  // clang-format off
  std::vector<TDesc> holofile_read(holotask::sources::HolofileSettings s);
  std::vector<TDesc> memcpy(const TDesc &X, holotask::syncs::MemcpySettings s);
  std::vector<TDesc> batched_queue(const TDesc &X, holotask::asyncs::BatchQueueSettings s);
  std::vector<TDesc> convert(const TDesc &X, holotask::syncs::ConversionSettings s);
  std::vector<TDesc> pca(const TDesc &X, holotask::syncs::PcaSettings s);
  std::vector<TDesc> stft(const TDesc &X, holotask::syncs::StftSettings s);
  std::vector<TDesc> filter_2d(const TDesc &X, holotask::syncs::Filter2DSettings s);
  std::vector<TDesc> fresnel_diffraction(const TDesc &X, holotask::syncs::FresnelDiffractionSettings s);
  std::vector<TDesc> angular_spectrum(const TDesc &X, holotask::syncs::AngularSpectrumSettings s);
  std::vector<TDesc> reshape(const TDesc &X, holotask::syncs::ReshapeSettings s);
  std::vector<TDesc> slide_avg(const TDesc &X, holotask::asyncs::SlidingAverageSettings s);
  std::vector<TDesc> xy_raw_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  std::vector<TDesc> xy_processed_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  std::vector<TDesc> xz_processed_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  std::vector<TDesc> yz_processed_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  std::vector<TDesc> ametek_s710_euresys_coaxlink_octo(holotask::sources::AmetekS710EuresysCoaxlinkOctoSettings s);
  std::vector<TDesc> ametek_s711_euresys_coaxlink_qsfp_plus(holotask::sources::AmetekS711EuresysCoaxlinkQSFPSettings s);
  std::vector<TDesc> average(const TDesc &X, holotask::syncs::AverageSettings s);
  std::vector<TDesc> convolution(const TDesc &X, holotask::syncs::ConvolutionSettings s);
  std::vector<TDesc> crop(const TDesc &X, holotask::syncs::CropSettings s);
  std::vector<TDesc> fft_shift(const TDesc &X, holotask::syncs::FFTShiftSettings s);
  std::vector<TDesc> pct_clip(const TDesc &X, holotask::syncs::PctClipSettings s);
  std::vector<TDesc> registration(const TDesc &X, holotask::syncs::RegistrationSettings s);
  std::vector<TDesc> rotation(const TDesc &X, holotask::syncs::RotationSettings s);
  std::vector<TDesc> holofile_write(const TDesc &X, holotask::sinks::HolofileSettings s);
  // clang-format on
};

} // namespace holovibes::pipeline

#ifndef HOLOVIBES__GRAPH_BUILDER_V2_HXX__INCLUDED
#include "graph_builder_v2.hh"
#endif