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
#include <charconv>
#include <cmath>
#include <cstdint>
#include <format>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "holoflow/runtime/compiler.hh"

namespace {

using namespace holoflow::core;
using GraphCompiledDumpPreferences = holoflow::runtime::GraphCompiledDumpPreferences;

static std::string replace_newlines_escaped_with_l(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (auto it = s.begin(); it != s.end(); ++it) {
    const char c = *it;
    if (c == '\\') {
      if (it + 1 != s.end() && *(it + 1) == 'n') {
        out += "\\l";
        ++it;
      } else {
        out += "\\";
      }
    } else {
      out += c;
    }
  }
  return out;
}

static std::string replace_newlines_with_l(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '\n') {
      out += "\\l";
    } else {
      out += c;
    }
  }
  return out;
}

// reuse escape helper
static std::string escape_for_label(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (auto it = s.begin(); it != s.end(); ++it) {
    const char c = *it;
    switch (c) {
    case '\\':
      if (it + 1 != s.end() && *(it + 1) == 'l') {
        out += "\\l";
        ++it;
      } else {
        out += "\\\\";
      }
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

static void round_json_floating_point_values(nlohmann::json &value, int precision) {
  if (value.is_array() || value.is_object()) {
    for (auto &child : value) {
      round_json_floating_point_values(child, precision);
    }
    return;
  }

  if (!value.is_number_float()) {
    return;
  }

  const double number = value.get<double>();
  if (!std::isfinite(number)) {
    return;
  }

  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), number,
                                    std::chars_format::scientific, precision);
  if (result.ec != std::errc{}) {
    return;
  }

  double rounded = number;
  const auto parsed = std::from_chars(buffer, result.ptr, rounded,
                                      std::chars_format::scientific);
  if (parsed.ec == std::errc{}) {
    value = rounded;
  }
}

static std::string dump_json_with_floating_point_precision(const nlohmann::json &value,
                                                           int precision) {
  auto rounded = value;
  round_json_floating_point_values(rounded, precision);
  return rounded.dump(2);
}

std::string tdesc_to_string(const TDesc &d) {
  std::ostringstream ss;
  ss << "{" << "\\n";
  ss << "  shape: " << escape_for_label(nlohmann::json(d.shape).dump()) << ",\\n";
  ss << "  dtype: " << escape_for_label(nlohmann::json(d.dtype).dump()) << ",\\n";
  ss << "  mem_loc: " << escape_for_label(nlohmann::json(d.mem_loc).dump()) << ",\\n";
  ss << "  strides: " << escape_for_label(nlohmann::json(d.strides).dump()) << "\\n";
  ss << "}";

  return ss.str();
}

std::string format_tdesc(const TDesc &d) {
  std::ostringstream ss;
  ss << "{\\n";
  ss << "  shape: " << escape_for_label(nlohmann::json(d.shape).dump()) << ",\\n";
  ss << "  dtype: " << escape_for_label(nlohmann::json(d.dtype).dump()) << "\\n";
  ss << "  mem_loc: " << escape_for_label(nlohmann::json(d.mem_loc).dump()) << "\\n";
  ss << "  strides: " << escape_for_label(nlohmann::json(d.strides).dump()) << "\\n";
  ss << "  offset: " << d.offset << "\\n";
  ss << "}";
  return ss.str();
}

} // namespace
namespace holoflow::runtime {

static void write_compiled_graph_header(std::ostringstream                 &ss,
                                        const GraphCompiledDumpPreferences &prefs,
                                        const std::string &title = "holoflow_compiled_graph") {
  ss << "digraph " << title << " {\n";
  if (prefs.rankdir == GraphCompiledDumpPreferences::Rankdir::LeftToRight)
    ss << "  rankdir=LR;\n";
  else
    ss << "  rankdir=TB;\n";

  ss << "  compound=true;\n";
  ss << "  node [fontname=\"Helvetica\", shape=box, style=filled];\n";
  ss << "  edge [fontname=\"Helvetica\"];\n\n";
}

static void write_compiled_nodes(std::ostringstream &ss, const runtime::GraphPlan &g,
                                 const holoflow::runtime::ExecResouces &res,
                                 const GraphCompiledDumpPreferences    &prefs) {

  auto fmt_id = [&](int tid) -> std::string {
    if (res.tid_to_sid.contains(tid)) {
      return std::format("{}(s:{})", tid, res.tid_to_sid.at(tid));
    }
    return std::to_string(tid);
  };

  auto get_visual_id = [&](size_t v, bool is_source) -> std::string {
    if (g[v].infer.kind == core::TaskKind::Async) {
      return is_source ? std::format("v{}_out", v) : std::format("v{}_in", v);
    }
    return std::format("v{}", v);
  };

  std::string ids_line = "";
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &np = g[v];

    std::ostringstream label_base;
    if (prefs.dump_node_name)
      label_base << (np.spec.name.empty() ? "(unnamed)" : np.spec.name);

    if (prefs.dump_node_settings && np.spec.debug && !np.spec.settings.is_null() &&
        !(np.spec.settings.is_object() && np.spec.settings.empty())) {
      if (!label_base.str().empty()) {
        label_base << "\n";
      }
      label_base << replace_newlines_with_l(dump_json_with_floating_point_precision(
                        np.spec.settings, prefs.floating_point_precision))
                 << "\\l";
    }

    if (prefs.dump_node_in_out_tids) {
      std::string in_str = "[";
      for (size_t i = 0; i < np.in_tids.size(); ++i) {
        in_str += (i ? "," : "") + fmt_id(np.in_tids[i]);
      }
      in_str += "]";

      std::string out_str = "[";
      for (size_t i = 0; i < np.out_tids.size(); ++i) {
        const int   out_tid = np.out_tids[i];
        std::string id_text = fmt_id(out_tid);

        bool is_alias = false;
        if (res.tid_to_sid.count(out_tid)) {
          const size_t out_sid = res.tid_to_sid.at(out_tid);
          for (int in_tid : np.in_tids) {
            if (res.tid_to_sid.count(in_tid) && res.tid_to_sid.at(in_tid) == out_sid) {
              is_alias = true;
              break;
            }
          }
        }

        out_str += (i ? "," : "") + id_text + (is_alias ? "*" : "");
      }
      out_str += "]";

      ids_line = "\nIn: " + in_str + "\nOut: " + out_str;
    }

    if (prefs.dump_node_kind) {
      if (np.infer.kind == core::TaskKind::Async) {
        const std::string label_in = label_base.str() + "\n(Producer/Write)" + ids_line + "\n";
        ss << std::format("  v{}_in [label=\"{}\", shape=invhouse, fillcolor=\"#e6f2ff\", "
                          "color=\"#0066cc\", style=\"filled,dashed\"];\n",
                          v, escape_for_label(label_in));

        const std::string label_out = label_base.str() + "\n(Consumer/Read)" + ids_line + "\n";
        ss << std::format("  v{}_out [label=\"{}\", shape=house, fillcolor=\"#ffe6e6\", "
                          "color=\"#cc0000\", style=\"filled,dashed\"];\n",
                          v, escape_for_label(label_out));

        ss << std::format("  v{}_in -> v{}_out [style=dotted, color=\"#888888\", penwidth=2, "
                          "arrowh=none, label=\"Async Signal\"];\n",
                          v, v);
      } else {
        const std::string label = label_base.str() + "\n(" + np.spec.kind + ")" + ids_line + "\n";
        ss << std::format("  v{} [label=\"{}\", fillcolor=\"#ccffcc\"];\n", v,
                          escape_for_label(label));
      }
    }
  }
}

static void write_compiled_edges(std::ostringstream &ss, const runtime::GraphPlan &g,
                                 const holoflow::runtime::ExecResouces &res,
                                 const GraphCompiledDumpPreferences    &prefs) {

  auto get_visual_id = [&](size_t v, bool is_source) -> std::string {
    if (g[v].infer.kind == core::TaskKind::Async) {
      return is_source ? std::format("v{}_out", v) : std::format("v{}_in", v);
    }
    return std::format("v{}", v);
  };

  for (auto e : boost::make_iterator_range(boost::edges(g))) {
    const auto  u  = boost::source(e, g);
    const auto  v  = boost::target(e, g);
    const auto &ep = g[e];

    const std::string u_vis = get_visual_id(u, true);
    const std::string v_vis = get_visual_id(v, false);

    std::ostringstream edge_lbl;
    edge_lbl << "tid:" << ep.tid;
    if (res.tid_to_sid.count(ep.tid)) {
      edge_lbl << " (s:" << res.tid_to_sid.at(ep.tid) << ")";
    }
    edge_lbl << "\\n" << format_tdesc(ep.desc);

    ss << std::format("  {} -> {} ", u_vis, v_vis);
    if (prefs.dump_edge_indices) {
      ss << std::format("[taillabel=\"{}\", headlabel=\"{}\"]", ep.spec.out_idx, ep.spec.in_idx);
    }
    if (prefs.dump_edge_descriptions) {
      auto formated = replace_newlines_escaped_with_l(edge_lbl.str());
      ss << std::format("[label=\"{}\\l\"]", formated);
    }
    ss << ";\n";
  }
}

static void write_compiled_sections(std::ostringstream                  &ss,
                                    const std::vector<runtime::Section> &sections,
                                    const GraphCompiledDumpPreferences  &prefs) {
  for (const auto &sec : sections) {
    ss << std::format("  subgraph cluster_section_{} {{\n", sec.id);

    ss << std::format("    label=\"Section {}", sec.id);
    if (prefs.dump_section_stream_addr) {
      ss << std::format("(Stream {})", (void *)sec.stream);
    }
    ss << std::format("\\l\";\n");

    ss << "    style=rounded; color=gray; bgcolor=\"#f8f8f8\";\n";

    for (auto vd : sec.sync_topo) {
      ss << std::format("    v{};\n", vd);
    }
    for (auto vd : sec.async_prod) {
      ss << std::format("    v{}_in;\n", vd);
    }
    for (auto vd : sec.async_cons) {
      ss << std::format("    v{}_out;\n", vd);
    }

    ss << "  }\n";
  }
}

std::string to_dot(const CompilerOutput &out, const GraphCompiledDumpPreferences &prefs,
                   std::string filename) {
  std::ostringstream ss;
  write_compiled_graph_header(ss, prefs, filename);

  write_compiled_nodes(ss, out.graph, out.resources, prefs);
  ss << "\n";
  write_compiled_edges(ss, out.graph, out.resources, prefs);
  ss << "\n";
  if (prefs.dump_section_info) {
    write_compiled_sections(ss, out.sections, prefs);
  }
  ss << "}\n";
  return ss.str();
}

} // namespace holoflow::runtime

namespace holoflow::core {

static void write_graph_header(std::ostringstream &ss, const GraphSpecDumpPreferences &dump_prefs) {
  ss << "digraph holoflow_graph {\n";
  if (dump_prefs.rankdir == GraphSpecDumpPreferences::Rankdir::LeftToRight) {
    ss << "  rankdir=LR;\n";
  } else {
    ss << "  rankdir=TB;\n";
  }
  ss << "  node [shape=box, fontname=\"Helvetica\"];\n";
  ss << "  edge [fontname=\"Helvetica\"];\n\n";
}

static void write_nodes(std::ostringstream &ss, const GraphSpec &g,
                        const GraphSpecDumpPreferences &dump_prefs) {
  using vertex_iter_t = boost::graph_traits<GraphSpec>::vertex_iterator;
  vertex_iter_t vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi) {
    auto            v  = *vi;
    const NodeSpec &ns = g[v];

    std::ostringstream label;
    if (dump_prefs.dump_node_name) {
      if (!ns.name.empty())
        label << ns.name;
      else
        label << "(unnamed)";
    }

    if (dump_prefs.dump_node_kind && !ns.kind.empty()) {
      if (!label.str().empty())
        label << "\n";
      label << "(" << ns.kind << ")\n";
    }

    if (dump_prefs.dump_node_settings && ns.debug && !ns.settings.is_null() &&
        !(ns.settings.is_object() && ns.settings.empty())) {
      std::string settings_dump = dump_json_with_floating_point_precision(
          ns.settings, dump_prefs.floating_point_precision);
      label << replace_newlines_with_l(settings_dump) << "\\l";
    }

    std::string esc_label = escape_for_label(label.str());
    ss << "  v" << v << " [label=\"" << esc_label << "\"];\n";
  }
  ss << "\n";
}

static void write_edges(std::ostringstream &ss, const GraphSpec &g,
                        const GraphSpecDumpPreferences &dump_prefs) {
  using edge_iter_t = boost::graph_traits<GraphSpec>::edge_iterator;
  edge_iter_t ei, ei_end;
  for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
    auto            e  = *ei;
    auto            s  = boost::source(e, g);
    auto            t  = boost::target(e, g);
    const EdgeSpec &es = g[e];

    ss << "  v" << s << " -> v" << t;
    if (dump_prefs.dump_edge_indices) {
      ss << " [taillabel=\"" << escape_for_label(std::to_string(es.out_idx)) << "\""
         << " headlabel=\"" << escape_for_label(std::to_string(es.in_idx)) << "\"]";
    }
    ss << ";\n";
  }
}

std::string to_dot(const GraphSpec &g, const GraphSpecDumpPreferences &dump_prefs) {
  std::ostringstream ss;
  write_graph_header(ss, dump_prefs);
  write_nodes(ss, g, dump_prefs);
  write_edges(ss, g, dump_prefs);
  ss << "}\n";
  return ss.str();
}

} // namespace holoflow::core
