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

#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"

#include <string>

namespace holoflow::runtime {

struct GraphCompiledDumpPreferences {
  enum class Rankdir { LeftToRight, TopToBottom };

  Rankdir rankdir        = Rankdir::LeftToRight;
  bool    dump_node_name = true;
  bool    dump_node_kind = true;
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
std::string to_dot(const CompilerOutput &out, const GraphCompiledDumpPreferences &prefs = {});

} // namespace holoflow::runtime
