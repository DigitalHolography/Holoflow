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

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow/runtime/graph_exec.hh"
#include <vector>

namespace holoflow::runtime {

struct CompilerOutput {
  GraphPlan            graph;     ///< Compiled graph plan.
  std::vector<Section> sections;  ///< Execution sections.
  ExecResouces         resources; ///< Preallocated execution resources.
};

class Compiler {
public:
  Compiler(core::Registry &registry, std::filesystem::path log_dir = "");

  CompilerOutput compile(const core::GraphSpec &gspec, CompilerOutput *prev = nullptr);

private:
  void check_duplicate_names() const;
  void check_duplicate_edge_dst() const;
  void check_single_source() const;
  void check_single_input() const;
  void check_factories_registered() const;
  void build_graph_plan();
  void check_typing();
  void check_buffer_temporal_consistency();
  void check_buffer_spatial_consistency();
  void assign_tensor_ids();
  void create_tensor_buffers();
  void create_tensor_views();
  void assign_cuda_streams();
  void create_nodes_collection();
  void get_pes_roots();
  void assign_inputs_outputs();

private:
  core::Registry       &registry_;
  std::filesystem::path log_dir_;

  core::GraphSpec gspec_;
  CompilerOutput *prev_ = nullptr;
  CompilerOutput  out_;
};

} // namespace holoflow::runtime