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

#include <map>

#include "graph_builder_tasks.hh"
#include "pipeline/settings.hh"

namespace holovibes::pipeline {

// Convenience alias — callers can write PhaseReference::LOCAL / GLOBAL without the holotask prefix.
using PhaseReference = holotask::syncs::STFDPhaseReference;

// GraphBuilder defines the holographic computation pipeline.
//
// It translates a Settings snapshot into a GraphSpec by wiring together the
// acquisition, preprocessing, time-frequency analysis, spatial propagation,
// and display stages via the task wrappers inherited from GraphBuilderTasks.
class GraphBuilder : public GraphBuilderTasks {
public:
  GraphBuilder(const Settings &settings, holoflow::core::Registry &registry);

  holoflow::core::GraphSpec build();

private:
  // Pipeline construction stages
  TDesc build_acquisition();

  // Sliding-window Fresnel propagation (analogous to STFT).
  //
  // Extracts overlapping windows of size (win_h x win_w) with the given strides from the last two
  // spatial dimensions of `field`, propagates each window, and returns the batched result with
  // shape [..., ny_win, nx_win, win_h, win_w].
  //
  // phase_ref == LOCAL:  each window is treated as an independent on-axis field.
  // phase_ref == GLOBAL: a corrective phase ramp is applied so that the result is equivalent to
  //                      first applying a global quadratic lens to the full field and then
  //                      propagating each window (plenoptic / Shack-Hartmann generalisation).
  TDesc short_time_fresnel_diffraction(const TDesc &field, size_t win_w, size_t win_h,
                                       size_t stride_x, size_t stride_y, float lam, float dx,
                                       float dy, float z_prop, PhaseReference phase_ref,
                                       bool skip_phase_shift = true);
  void  build_raw_record(const TDesc &H);
  bool  build_raw_view(const TDesc &H);
  TDesc build_preprocessing(TDesc H);
  TDesc build_time_frequency_analysis(TDesc H);
  TDesc build_shack_hartmann(TDesc FH, bool is_last_pass);
  TDesc build_spatial_propagation(const TDesc &FH);
  TDesc build_spatial_filter(const TDesc &FH_z);
  void  build_xy_view(const TDesc &FH_z);
  void  build_3d_cuts(const TDesc &FH_z);
  TDesc build_freq_weights();

  Settings s_;

  std::map<LoadMethod, holotask::sources::HolofileSettings::LoadKind> load_method_map_{
      {LoadMethod::READ_LIVE, holotask::sources::HolofileSettings::LoadKind::Live},
      {LoadMethod::LOAD_IN_CPU, holotask::sources::HolofileSettings::LoadKind::CPUCached},
      {LoadMethod::LOAD_IN_GPU, holotask::sources::HolofileSettings::LoadKind::GPUCached},
  };
};

} // namespace holovibes::pipeline
