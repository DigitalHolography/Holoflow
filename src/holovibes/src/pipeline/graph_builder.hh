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
  void  build_raw_record(const TDesc &H);
  bool  build_raw_view(const TDesc &H);
  TDesc build_preprocessing(TDesc H);
  TDesc build_time_frequency_analysis(TDesc H);
  TDesc build_shack_hartmann(TDesc FH);
  TDesc build_spatial_propagation(const TDesc &FH);
  void  build_xy_view(const TDesc &FH_z);
  void  build_3d_cuts(const TDesc &FH_z);

  Settings s_;

  std::map<LoadMethod, holotask::sources::HolofileSettings::LoadKind> load_method_map_{
      {LoadMethod::READ_LIVE, holotask::sources::HolofileSettings::LoadKind::Live},
      {LoadMethod::LOAD_IN_CPU, holotask::sources::HolofileSettings::LoadKind::CPUCached},
      {LoadMethod::LOAD_IN_GPU, holotask::sources::HolofileSettings::LoadKind::GPUCached},
  };
};

} // namespace holovibes::pipeline
