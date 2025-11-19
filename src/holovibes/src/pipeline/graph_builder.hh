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

#pragma once

#include "holoflow/runtime/graph_display.hh"
#include "settings_loader.hh"
#include <map>
#include <optional>

namespace holovibes::pipeline {

template <typename T>
concept JsonSerializable = requires(T a, nlohmann::json j) {
  { to_json(j, a) } -> std::same_as<void>;
  { from_json(j, a) } -> std::same_as<void>;
};

class GraphBuilder {
public:
  using V = boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                         boost::bidirectionalS>::vertex_descriptor;

  GraphBuilder(holoflow::core::GraphSpec &spec, const Settings &settings, int src_width,
               int src_height, bool opti_cpu_stride, bool opti_gpu_stride);

  void build();

private:
  // Core graph structure
  holoflow::core::GraphSpec &spec_;
  const Settings            &s_;
  int                        src_width_;
  int                        src_height_;
  bool                       opti_cpu_stride_;
  bool                       opti_gpu_stride_;
  std::string                current_section_name_;

  // Utilities to build the graph
  template <JsonSerializable S>
  V add_node(const std::string &name, const std::string &kind, const S &settings);

  template <JsonSerializable S>
  V add_node_after(const V &after, int out_idx, int in_idx, const std::string &name,
                   const std::string &kind, const S &settings);

  // Pipeline branches
  V    build_raw_branch(V cpu_in_queue);
  V    build_processed_branch(V parent);
  void build_xy_branch(V debounce_queue);
  void build_xz_branch(V debounce_queue);
  void build_yz_branch(V debounce_queue);

  // Node creation methods
  V add_source();
  V add_cpu_in_queue(V parent, int out_idx, int in_idx);
  V add_cpu_raw_queue(V parent, int out_idx, int in_idx);
  V add_raw_reshape(V parent, int out_idx, int in_idx);
  V add_record_queue(V parent, int out_idx, int in_idx);
  V add_cpu_cpu_cpy(V parent, int out_idx, int in_idx);
  V add_cpu_raw_view_cpy(V parent, int out_idx, int in_idx);
  V add_xy_raw_display(V parent, int out_idx, int in_idx);
  V add_raw_record(V parent, int out_idx, int in_idx);
  V add_cpu_gpu_cpy(V parent, int out_idx, int in_idx);
  V add_gpu_in_queue(V parent, int out_idx, int in_idx);
  V add_to_cf32(V parent, int out_idx, int in_idx);
  V add_spacial_transform(V parent, int out_idx, int in_idx);
  V add_spacial_filter(V parent, int out_idx, int in_idx);
  V add_time_queue(V parent, int out_idx, int in_idx);
  V add_time_transform(V parent, int out_idx, int in_idx);
  V add_to_f32(V parent, int out_idx, int in_idx);
  V add_debounce_queue(V parent, int out_idx, int in_idx);
  V add_xy_cut_avg(V parent, int out_idx, int in_idx);
  V add_fft_shift(V parent, int out_idx, int in_idx);
  V add_xy_registration(V parent, int out_idx, int in_idx);
  V add_xy_slide_avg(V parent, int out_idx, int in_idx);
  V add_xy_fps_limiter(V parent, int out_idx, int in_idx);
  V add_xy_convolution(V parent, int out_idx, int in_idx);
  V add_xy_pctclip(V parent, int out_idx, int in_idx);
  V add_xy_to_u8(V parent, int out_idx, int in_idx);
  V add_xy_gpu_out_queue(V parent, int out_idx, int in_idx);
  V add_xy_gpu_cpu_cpy(V parent, int out_idx, int in_idx);
  V add_xy_cpu_out_queue(V parent, int out_idx, int in_idx);
  V add_xy_processed_display(V parent, int out_idx, int in_idx);
  V add_xz_cut_avg(V parent, int out_idx, int in_idx);
  V add_xz_reshape(V parent, int out_idx, int in_idx);
  V add_xz_slide_avg(V parent, int out_idx, int in_idx);
  V add_xz_crop2frames(V parent, int out_idx, int in_idx);
  V add_xz_to_u8(V parent, int out_idx, int in_idx);
  V add_xz_gpu_out_queue(V parent, int out_idx, int in_idx);
  V add_xz_gpu_cpu_cpy(V parent, int out_idx, int in_idx);
  V add_xz_cpu_out_queue(V parent, int out_idx, int in_idx);
  V add_xz_processed_display(V parent, int out_idx, int in_idx);
  V add_yz_cut_avg(V parent, int out_idx, int in_idx);
  V add_yz_reshape(V parent, int out_idx, int in_idx);
  V add_yz_slide_avg(V parent, int out_idx, int in_idx);
  V add_yz_crop2frames(V parent, int out_idx, int in_idx);
  V add_yz_to_u8(V parent, int out_idx, int in_idx);
  V add_yz_rotation(V parent, int out_idx, int in_idx);
  V add_yz_gpu_out_queue(V parent, int out_idx, int in_idx);
  V add_yz_gpu_cpu_cpy(V parent, int out_idx, int in_idx);
  V add_yz_cpu_out_queue(V parent, int out_idx, int in_idx);
  V add_yz_processed_display(V parent, int out_idx, int in_idx);
};

} // namespace holovibes::pipeline