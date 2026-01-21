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

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/runtime/graph_exec.hh"

namespace holoflow::runtime {

// -------------------------------------------------------------------------------------------------
// Compiler output
// -------------------------------------------------------------------------------------------------

struct CompilerOutput {
  GraphPlan            graph;
  std::vector<Section> sections;
  ExecResouces         resources;
};

// -------------------------------------------------------------------------------------------------
// Compiler
// -------------------------------------------------------------------------------------------------

class Compiler {
public:
  struct Config {
    std::filesystem::path log_dir;
    bool                  dump_dot_on_failure = true;
    bool                  verbose_tracing     = true;
  };

  explicit Compiler(core::Registry &registry, Config config = {});
  ~Compiler();

  [[nodiscard]] std::unique_ptr<CompilerOutput>
  compile(const core::GraphSpec &gspec, std::unique_ptr<CompilerOutput> prev = nullptr);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace holoflow::runtime

// #pragma once

// #include "holoflow/core/graph_spec.hh"
// #include "holoflow/core/registry.hh"
// #include "holoflow/runtime/graph_exec.hh"
// #include <string>
// #include <vector>

// namespace holoflow::runtime {

// struct CompilerOutput {
//   GraphPlan            graph;     ///< Compiled graph plan.
//   std::vector<Section> sections;  ///< Execution sections.
//   ExecResouces         resources; ///< Preallocated execution resources.
// };

// class Compiler {
// public:
//   Compiler(core::Registry &registry, const std::filesystem::path &log_dir = "");

//   std::unique_ptr<CompilerOutput> compile(const core::GraphSpec          &gspec,
//                                           std::unique_ptr<CompilerOutput> prev = nullptr);

// private:
//   struct StepTiming {
//     std::string name;
//     double      duration_ms;
//   };

//   struct NodeTiming {
//     std::string name;
//     double      duration_ms;
//   };

//   void write_profile_report(const std::vector<StepTiming> &step_timings,
//                             double                         total_duration_ms) const;

//   void check_duplicate_names();
//   void check_duplicate_edge_dst();
//   void check_single_source();
//   void check_single_input();
//   void check_factories_registered();
//   void build_graph_plan();
//   void check_typing();
//   void assign_tensor_ids();
//   void check_buffer_temporal_consistency();
//   void check_buffer_spatial_consistency();
//   void create_tensor_buffers();
//   void create_sections();
//   void assign_cuda_streams();
//   void create_nodes_collection();

// private:
//   std::vector<NodeTiming> node_timings_;

//   core::Registry       &registry_;
//   std::filesystem::path log_dir_;

//   core::GraphSpec                 gspec_;
//   std::unique_ptr<CompilerOutput> prev_ = nullptr;
//   std::unique_ptr<CompilerOutput> out_;
//   std::map<std::string, int>      sync_section_map_;
//   std::map<std::string, int>      async_prod_section_map_;
//   std::map<std::string, int>      async_cons_section_map_;
// };

// } // namespace holoflow::runtime