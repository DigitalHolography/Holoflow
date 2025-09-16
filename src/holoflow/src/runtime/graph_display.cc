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

#include "holoflow/runtime/graph_display.hh"

#include "holoflow/core/tensor.hh"

#include <boost/graph/graph_traits.hpp>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>

namespace holoflow::runtime {

namespace {

// reuse escape helper
std::string escape_for_label(const std::string &s) {
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

std::string tdesc_to_string(const core::TDesc &d) {
  std::ostringstream ss;
  ss << "shape " << d.shape.size();
  ss << " x " << std::string(to_string(d.dtype));
  ss << " (" << d.rank() << "D)";
  ss << " mem=" << std::string(holoflow::core::to_string(d.mem_loc));
  ss << " elems=" << d.num_elements() << " bytes=" << d.num_bytes();

  return ss.str();
}

std::string tensor_summary(const core::Tensor &t) {
  std::ostringstream ss;
  const core::TDesc &d = t.desc();
  ss << tdesc_to_string(d);
  const void *ptr = t.data();
  ss << " ptr=" << reinterpret_cast<uintptr_t>(ptr);
  return ss.str();
}
} // namespace

static void write_compiled_graph_header(std::ostringstream &ss,
                                        const std::string  &title = "holoflow_compiled_graph") {
  ss << "digraph " << title << " {\n";
  ss << "  rankdir=LR;\n";
  ss << "  node [fontname=\"Helvetica\"];\n";
  ss << "  edge [fontname=\"Helvetica\"];\n\n";
}

static void write_compiled_nodes(std::ostringstream &ss, const runtime::GraphPlan &g,
                                 const std::vector<runtime::Section> &sections,
                                 core::Registry &registry, std::size_t settings_max_len = 300) {
  using vertex_iter_t = boost::graph_traits<runtime::GraphPlan>::vertex_iterator;
  vertex_iter_t vi, vi_end;

  std::unordered_set<runtime::GraphPlan::vertex_descriptor> async_producers;
  std::unordered_set<runtime::GraphPlan::vertex_descriptor> async_consumers;
  for (const auto &sec : sections) {
    for (auto vd : sec.async_prod)
      async_producers.insert(vd);
    for (auto vd : sec.async_cons)
      async_consumers.insert(vd);
  }

  for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi) {
    auto                     v  = *vi;
    const runtime::NodePlan &np = g[v];

    std::ostringstream label;

    if (!np.spec.name.empty())
      label << np.spec.name;
    else
      label << "(unnamed)";

    if (!np.spec.kind.empty())
      label << "\n(" << np.spec.kind << ")";

    try {
      if (!np.spec.settings.is_null() &&
          !(np.spec.settings.is_object() && np.spec.settings.empty())) {
        std::string sd = np.spec.settings.dump();
        if (sd.size() > settings_max_len)
          sd = sd.substr(0, settings_max_len) + "...";
        // raw JSON might contain quotes/newlines -> we escape later
        label << "\n" << sd;
      }
    } catch (...) {
    }

    label << "\n[infer: ";
    label << "present]";
    label << "\n" << "in_tids:[";
    for (size_t i = 0; i < np.in_tids.size(); ++i) {
      if (i)
        label << ",";
      label << np.in_tids[i];
    }
    label << "]";

    label << "\n" << "out_tids:[";
    for (size_t i = 0; i < np.out_tids.size(); ++i) {
      if (i)
        label << ",";
      label << np.out_tids[i];
    }
    label << "]";

    std::string esc_label = escape_for_label(label.str());

    std::string shape = "record";
    try {
      registry.get_async(np.spec.kind);
      shape = "octagon";
    } catch (...) {
      shape = "record";
    }

    std::string style_attrs;
    if (async_producers.count(v) && async_consumers.count(v)) {
      style_attrs = ", style=bold";
    } else if (async_producers.count(v)) {
      style_attrs = ", color=blue, style=dashed";
    } else if (async_consumers.count(v)) {
      style_attrs = ", color=red, style=dashed";
    }

    ss << "  v" << v << " [label=\"" << esc_label << "\", shape=" << shape << style_attrs << "];\n";
  }
  ss << "\n";
}

static void write_compiled_edges(std::ostringstream &ss, const runtime::GraphPlan &g,
                                 std::size_t desc_max_len = 200) {
  using edge_iter_t = boost::graph_traits<runtime::GraphPlan>::edge_iterator;
  edge_iter_t ei, ei_end;
  for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
    auto                     e  = *ei;
    auto                     s  = boost::source(e, g);
    auto                     t  = boost::target(e, g);
    const runtime::EdgePlan &ep = g[e];

    std::ostringstream elabel;
    elabel << "out:" << ep.spec.out_idx << " in:" << ep.spec.in_idx << " tid:" << ep.tid;

    try {
      std::string desc_str = tdesc_to_string(ep.desc);
      if (desc_str.size() > desc_max_len)
        desc_str = desc_str.substr(0, desc_max_len) + "...";
      elabel << " desc=" << desc_str;
    } catch (...) {
      // ignore
    }

    std::string esc_elabel = escape_for_label(elabel.str());
    ss << "  v" << s << " -> v" << t << " [label=\"" << esc_elabel << "\"];\n";
  }
  ss << "\n";
}

static void write_compiled_sections(std::ostringstream                  &ss,
                                    const std::vector<runtime::Section> &sections) {
  for (const auto &sec : sections) {
    ss << "  subgraph cluster_section_" << sec.id << " {\n";
    ss << "    label = \""
       << escape_for_label("Section " + std::to_string(sec.id) + ": " + sec.name) << "\";\n";
    ss << "    labelloc = \"t\";\n";
    ss << "    color = gray;\n";
    ss << "    fontsize = 10;\n";
    ss << "    // stream_ptr=" << reinterpret_cast<uintptr_t>(sec.stream) << "\n";

    ss << "    // sync_topo\n";
    for (auto vd : sec.sync_topo)
      ss << "    v" << vd << ";\n";

    if (!sec.async_prod.empty()) {
      ss << "    // async_producers\n";
      for (auto vd : sec.async_prod)
        ss << "    v" << vd << ";\n";
    }
    if (!sec.async_cons.empty()) {
      ss << "    // async_consumers\n";
      for (auto vd : sec.async_cons)
        ss << "    v" << vd << ";\n";
    }

    ss << "  }\n\n";
  }
}

static void write_compiled_resources(std::ostringstream &ss, const runtime::ExecResouces &res) {
  ss << "  // --- resources summary ---\n";

  ss << "  // streams: ";
  bool first = true;
  for (const auto &p : res.streams) {
    if (!first)
      ss << ", ";
    ss << p.first << "(addr=" << reinterpret_cast<uintptr_t>(std::addressof(p.second)) << ")";
    first = false;
  }
  ss << "\n";

  ss << "  // tasks: ";
  first = true;
  for (const auto &p : res.tasks) {
    if (!first)
      ss << ", ";
    ss << p.first;
    first = false;
  }
  ss << "\n";

  ss << "  // tensors:\n";
  for (const auto &p : res.tensors) {
    int                 tid = p.first;
    const core::Tensor &t   = p.second;
    ss << "  //   tid=" << tid << " : " << escape_for_label(tensor_summary(t)) << "\n";
  }
  ss << "\n";
}

std::string to_dot(const CompilerOutput &out, core::Registry &registry) {
  std::ostringstream ss;
  write_compiled_graph_header(ss, "holoflow_compiled");

  try {
    write_compiled_resources(ss, out.resources);
  } catch (...) {
  }

  try {
    write_compiled_nodes(ss, out.graph, out.sections, registry);
    write_compiled_edges(ss, out.graph);
  } catch (...) {
  }

  try {
    write_compiled_sections(ss, out.sections);
  } catch (...) {
  }

  ss << "}\n";
  return ss.str();
}

} // namespace holoflow::runtime
