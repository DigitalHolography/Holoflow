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

#include "holoflow/core/graph_spec.hh"

#include <algorithm>
#include <boost/graph/graph_traits.hpp>
#include <sstream>
#include <string>

namespace holoflow::core {

static std::string escape_for_label(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      break;
    default:
      out += c;
      break;
    }
  }
  return out;
}

static void write_graph_header(std::ostringstream &ss) {
  ss << "digraph holoflow_graph {\n";
  ss << "  rankdir=LR;\n";
  ss << "  node [shape=box, fontname=\"Helvetica\"];\n";
  ss << "  edge [fontname=\"Helvetica\"];\n\n";
}

static void write_nodes(std::ostringstream &ss, const GraphSpec &g) {
  using vertex_iter_t = boost::graph_traits<GraphSpec>::vertex_iterator;
  vertex_iter_t vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi) {
    auto            v  = *vi;
    const NodeSpec &ns = g[v];

    std::ostringstream label;
    if (!ns.name.empty())
      label << ns.name;
    else
      label << "(unnamed)";

    if (!ns.kind.empty())
      label << "\n(" << ns.kind << ")";

    try {
      if (!ns.settings.is_null() && !(ns.settings.is_object() && ns.settings.empty())) {
        std::string settings_dump = ns.settings.dump(2);
        label << "\n" << settings_dump;
      }
    } catch (...) {
      // ignore bad settings
    }

    std::string esc_label = escape_for_label(label.str());

    ss << "  v" << v << " [label=\"" << esc_label << "\"];\n";
  }
  ss << "\n";
}

static void write_edges(std::ostringstream &ss, const GraphSpec &g) {
  using edge_iter_t = boost::graph_traits<GraphSpec>::edge_iterator;
  edge_iter_t ei, ei_end;
  for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
    auto            e  = *ei;
    auto            s  = boost::source(e, g);
    auto            t  = boost::target(e, g);
    const EdgeSpec &es = g[e];

    ss << "  v" << s << " -> v" << t << " [taillabel=\""
       << escape_for_label(std::to_string(es.out_idx)) << "\""
       << " headlabel=\"" << escape_for_label(std::to_string(es.in_idx)) << "\"];\n";
  }
}

std::string to_dot(const GraphSpec &g) {
  std::ostringstream ss;
  write_graph_header(ss);
  write_nodes(ss, g);
  write_edges(ss, g);
  ss << "}\n";
  return ss.str();
}

} // namespace holoflow::core