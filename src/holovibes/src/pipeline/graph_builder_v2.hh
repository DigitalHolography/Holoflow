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
#include "holonp/abs.hh"
#include "holonp/add.hh"
#include "holonp/arange.hh"
#include "holonp/asarray.hh"
#include "holonp/ascontiguousarray.hh"
#include "holonp/assign.hh"
#include "holonp/concatenate.hh"
#include "holonp/conj.hh"
#include "holonp/copy.hh"
#include "holonp/div.hh"
#include "holonp/empty.hh"
#include "holonp/equal.hh"
#include "holonp/fft.hh"
#include "holonp/fft2.hh"
#include "holonp/fftshift.hh"
#include "holonp/irfft2.hh"
#include "holonp/max.hh"
#include "holonp/mean.hh"
#include "holonp/mean_abs.hh"
#include "holonp/meshgrid.hh"
#include "holonp/min.hh"
#include "holonp/mul.hh"
#include "holonp/normalize.hh"
#include "holonp/reshape.hh"
#include "holonp/rfft.hh"
#include "holonp/rfft2.hh"
#include "holonp/slice.hh"
#include "holonp/sub.hh"
#include "holonp/transpose.hh"
#include "holonp/where.hh"
#include "holonp/zeros.hh"
#include "holotask/asyncs/batch_queue.hh"
#include "holotask/asyncs/slide_avg.hh"
#include "holotask/sinks/holofile.hh"
#include "holotask/sources/ametek_s710_euresys_coaxlink_octo.hh"
#include "holotask/sources/ametek_s711_euresys_coaxlink_qsfp+.hh"
#include "holotask/sources/fresnel_qin.hh"
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
#include "holotask/syncs/rotation.hh"
#include "holotask/syncs/stft.hh"
#include "holotask/syncs/zernike.hh"
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
      return std::tuple<T>(s[I]...);
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
  std::vector<TDesc> make_nary_sync_node(std::string_view node_name, std::string_view kind,
                                         std::string_view reg_key, std::span<const TDesc> inputs,
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
  std::vector<TDesc> empty(holonp::EmptySettings s);
  std::vector<TDesc> zeros(holonp::ZerosSettings s);
  std::vector<TDesc> asarray(holonp::AsArraySettings s);
  std::vector<TDesc> ascontiguousarray(const TDesc &X, holonp::AsContiguousArraySettings s);
  std::vector<TDesc> copy(const TDesc &X, holonp::CopySettings s);
  std::vector<TDesc> memcpy(const TDesc &X, holotask::syncs::MemcpySettings s);
  std::vector<TDesc> batched_queue(const TDesc &X, holotask::asyncs::BatchQueueSettings s);
  std::vector<TDesc> convert(const TDesc &X, holotask::syncs::ConversionSettings s);
  std::vector<TDesc> pca(const TDesc &X, holotask::syncs::PcaSettings s);
  std::vector<TDesc> stft(const TDesc &X, holotask::syncs::StftSettings s);
  std::vector<TDesc> filter_2d(const TDesc &X, holotask::syncs::Filter2DSettings s);
  std::vector<TDesc> fresnel_diffraction(const TDesc &X, holotask::syncs::FresnelDiffractionSettings s);
  std::vector<TDesc> fresnel_qin(const TDesc &Z, holotask::sources::FresnelQinSettings s);
  std::vector<TDesc> angular_spectrum(const TDesc &X, holotask::syncs::AngularSpectrumSettings s);
  std::vector<TDesc> slide_avg(const TDesc &X, holotask::asyncs::SlidingAverageSettings s);
  std::vector<TDesc> xy_raw_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  std::vector<TDesc> xy_processed_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  std::vector<TDesc> xz_processed_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  std::vector<TDesc> yz_processed_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  std::vector<TDesc> shack_hartmann_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  std::vector<TDesc> shack_hartmann_xcorr_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  std::vector<TDesc> ametek_s710_euresys_coaxlink_octo(holotask::sources::AmetekS710EuresysCoaxlinkOctoSettings s);
  std::vector<TDesc> ametek_s711_euresys_coaxlink_qsfp_plus(holotask::sources::AmetekS711EuresysCoaxlinkQSFPSettings s);
  std::vector<TDesc> average(const TDesc &X, holotask::syncs::AverageSettings s);
  std::vector<TDesc> convolution(const TDesc &X, holotask::syncs::ConvolutionSettings s);
  std::vector<TDesc> crop(const TDesc &X, holotask::syncs::CropSettings s);
  std::vector<TDesc> fft_shift(const TDesc &X, holotask::syncs::FFTShiftSettings s);
  std::vector<TDesc> pct_clip(const TDesc &X, holotask::syncs::PctClipSettings s);
  std::vector<TDesc> registration(const TDesc &X, holotask::syncs::RegistrationSettings s);
  std::vector<TDesc> rotation(const TDesc &X, holotask::syncs::RotationSettings s);
  std::vector<TDesc> zernike(const TDesc &X, holotask::syncs::ZernikeSettings s);
  std::vector<TDesc> holofile_write(const TDesc &X, holotask::sinks::HolofileSettings s);
  std::vector<TDesc> concatenate(std::span<const TDesc> Xs, holonp::ConcatenateSettings s);
  std::vector<TDesc> transpose(const TDesc &X, holonp::TransposeSettings s);
  std::vector<TDesc> add(const TDesc &A, const TDesc &B, holonp::AddSettings s);
  std::vector<TDesc> div(const TDesc &A, const TDesc &B, holonp::DivSettings s);
  std::vector<TDesc> mul(const TDesc &A, const TDesc &B, holonp::MulSettings s);
  std::vector<TDesc> sub(const TDesc &A, const TDesc &B, holonp::SubSettings s);
  std::vector<TDesc> equal(const TDesc &A, const TDesc &B, holonp::EqualSettings s);
  std::vector<TDesc> where(const TDesc &Cond, const TDesc &X, const TDesc &Y,
                           holonp::WhereSettings s);
  std::vector<TDesc> rfft(const TDesc &X, holonp::RFFTSettings s);
  std::vector<TDesc> rfft2(const TDesc &X, holonp::RFFT2Settings s);
  std::vector<TDesc> irfft2(const TDesc &X, holonp::IRFFT2Settings s);
  std::vector<TDesc> assign(const TDesc &X, const TDesc &Y, holonp::AssignSettings s);
  std::vector<TDesc> slice(const TDesc &X, holonp::SliceSettings s);
  std::vector<TDesc> fft(const TDesc &X, holonp::FFTSettings s);
  std::vector<TDesc> fft2(const TDesc &X, holonp::FFT2Settings s);
  std::vector<TDesc> fftshift(const TDesc &X, holonp::FFTShiftSettings s);
  std::vector<TDesc> abs(const TDesc &X, holonp::AbsSettings s);
  std::vector<TDesc> mean(const TDesc &X, holonp::MeanSettings s);
  std::vector<TDesc> mean_abs(const TDesc &X, holonp::MeanAbsSettings s);
  std::vector<TDesc> min(const TDesc &X, holonp::MinSettings s);
  std::vector<TDesc> max(const TDesc &X, holonp::MaxSettings s);
  std::vector<TDesc> normalize(const TDesc &X, holonp::NormalizeSettings s);
  std::vector<TDesc> reshape(const TDesc &X, holonp::ReshapeSettings s);
  std::vector<TDesc> conj(const TDesc &X, holonp::ConjSettings s);
  // clang-format on
};

} // namespace holovibes::pipeline

#ifndef HOLOVIBES__GRAPH_BUILDER_V2_HXX__INCLUDED
#include "graph_builder_v2.hh"
#endif
