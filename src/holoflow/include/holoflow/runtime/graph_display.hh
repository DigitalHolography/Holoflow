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

#include <string>
// could not forward declare GraphSpec because it is an alias
#include "holoflow/core/graph_spec.hh"
namespace holoflow::runtime {

struct CompilerOutput;
struct GraphCompiledDumpPreferences {
  enum class Rankdir { LeftToRight, TopToBottom };
  enum class Layout { Normal, Stairs, Block, Snake };

  Rankdir rankdir                  = Rankdir::LeftToRight;
  Layout  layout                   = Layout::Normal;
  int     floating_point_precision = 17;
  bool    dump_node_name           = true;
  bool    dump_node_kind           = true;
  // TODO : remove node settings
  bool dump_node_settings    = true;
  bool dump_node_in_out_tids = true;

  bool dump_edge_indices      = true;
  bool dump_edge_descriptions = true;

  bool dump_section_info        = true;
  bool dump_section_stream_addr = true;
  bool dump_resource_info       = true;
};

/// Serialize a compiled graph (CompilerOutput) to Graphviz DOT format.
/// This prints:
///  - node labels with name/kind/settings/in_tids/out_tids/infer marker
///  - edge labels with out/in indices, tid and TDesc summary
///  - clusters for Sections (sync/async grouping)
///
/// @param out       Compiled graph output (non-owning reference).
/// @param prefs     Preferences for controlling the output format.
/// @return          DOT source as std::string.
std::string to_dot(const CompilerOutput &out, const GraphCompiledDumpPreferences &prefs = {},
                   std::string filename = "compiled");

} // namespace holoflow::runtime

namespace holoflow::core {

struct GraphSpecDumpPreferences {
  enum class Rankdir { LeftToRight, TopToBottom };

  Rankdir rankdir                  = Rankdir::LeftToRight;
  int     floating_point_precision = 17;
  bool    dump_node_name           = true;
  bool    dump_node_kind           = true;
  bool    dump_node_settings       = true;
  bool    dump_edge_indices        = true;
};

/// Serialize a graph specification to a dot format string.
/// @param g     Graph specification to serialize.
/// @return      Dot format representation of the graph specification.
std::string to_dot(const holoflow::core::GraphSpec &g,
                   const GraphSpecDumpPreferences  &dump_prefs = {});
} // namespace holoflow::core
