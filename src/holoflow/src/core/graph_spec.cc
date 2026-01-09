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
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bug.hh"

namespace holoflow::core {

namespace {

[[noreturn]] void json_error(const std::string &msg) {
  throw std::runtime_error("GraphSpec JSON: " + msg);
}

inline void require(bool cond, const std::string &msg) {
  if (!cond)
    json_error(msg);
}

inline bool has_object_key(const nlohmann::json &j, const char *key) {
  return j.is_object() && j.contains(key) && !j.at(key).is_null();
}

inline std::vector<std::string> sorted_object_keys(const nlohmann::json &obj) {
  std::vector<std::string> keys;
  keys.reserve(obj.size());
  for (auto it = obj.begin(); it != obj.end(); ++it)
    keys.push_back(it.key());
  std::ranges::sort(keys);
  return keys;
}

} // namespace

nlohmann::json to_json(const GraphSpec &g, const GraphSpecWriteOptions &opts) {
  // clang-format off
  HOLOFLOW_CHECK(opts.include_node_names, "include_node_names=false not supported in this format");
  HOLOFLOW_CHECK(opts.include_node_kinds, "include_node_kinds=false not supported in this format");
  HOLOFLOW_CHECK(opts.include_node_settings, "include_node_settings=false not supported in this format");
  HOLOFLOW_CHECK(opts.include_edge_indices, "include_edge_indices=false not supported in this format");
  // clang-format on

  nlohmann::json j;
  j["nodes"] = nlohmann::json::object();
  j["edges"] = nlohmann::json::array();

  // Nodes
  for (auto vd : boost::make_iterator_range(vertices(g))) {
    const auto &n = g[vd];
    require(!n.name.empty(), "node has empty name");
    nlohmann::json node_obj = nlohmann::json::object();

    if (opts.include_node_kinds) {
      node_obj["type"] = n.kind;
    }

    if (opts.include_node_settings) {
      node_obj["params"] = n.settings.is_null() ? nlohmann::json::object() : n.settings;
    }

    if (!n.debug) {
      node_obj["debug"] = false;
    }

    const std::string key = n.name;
    j["nodes"][key]       = std::move(node_obj);
  }

  // Edges
  for (auto ed : boost::make_iterator_range(edges(g))) {
    const auto  src = source(ed, g);
    const auto  dst = target(ed, g);
    const auto &e   = g[ed];

    const auto &src_name = g[src].name;
    const auto &dst_name = g[dst].name;

    require(!src_name.empty() && !dst_name.empty(), "edge connects unnamed node(s)");

    nlohmann::json edge_obj;
    edge_obj["from"] = src_name;
    edge_obj["to"]   = dst_name;
    edge_obj["out"]  = e.out_idx;
    edge_obj["in"]   = e.in_idx;
    j["edges"].push_back(std::move(edge_obj));
  }

  return j;
}

GraphSpec from_json(const nlohmann::json &j) {
  require(j.is_object(), "root must be an object");
  require(has_object_key(j, "nodes"), "missing required key 'nodes'");
  require(j.at("nodes").is_object(), "'nodes' must be an object");

  const auto &jnodes = j.at("nodes");
  const auto &jedges = j.contains("edges") ? j.at("edges") : nlohmann::json::array();
  require(jedges.is_array(), "'edges' must be an array if present");

  GraphSpec g;
  using Vertex = boost::graph_traits<GraphSpec>::vertex_descriptor;
  std::unordered_map<std::string, Vertex> name_to_v;
  name_to_v.reserve(jnodes.size());

  // Deterministic vertex numbering: sort node keys.
  for (const auto &name : sorted_object_keys(jnodes)) {
    require(!name.empty(), "node key (name) must not be empty");

    const auto &node_json = jnodes.at(name);
    require(node_json.is_object(), "node '" + name + "' must be an object");
    require(has_object_key(node_json, "type"), "node '" + name + "': missing 'type'");
    require(node_json.at("type").is_string(), "node '" + name + "': 'type' must be a string");
    require(has_object_key(node_json, "params"), "node '" + name + "': missing 'params'");

    NodeSpec spec;
    spec.name = name;
    spec.kind = node_json.at("type").get<std::string>();

    spec.settings = node_json.at("params");
    require(spec.settings.is_object() || spec.settings.is_array() || spec.settings.is_primitive() ||
                spec.settings.is_null(),
            "node '" + name + "': 'params' must be valid JSON");

    if (spec.settings.is_null())
      spec.settings = nlohmann::json::object();

    if (node_json.contains("debug")) {
      require(node_json.at("debug").is_boolean(), "node '" + name + "': 'debug' must be boolean");
      spec.debug = node_json.at("debug").get<bool>();
    } else {
      spec.debug = true;
    }

    const auto v              = add_vertex(std::move(spec), g);
    const auto [it, inserted] = name_to_v.emplace(name, v);
    require(inserted, "duplicate node name '" + name + "'");
  }

  // Add edges
  for (std::size_t i = 0; i < jedges.size(); ++i) {
    const auto &e = jedges.at(i);
    require(e.is_object(), "edge[" + std::to_string(i) + "] must be an object");
    require(has_object_key(e, "from"), "edge[" + std::to_string(i) + "]: missing 'from'");
    require(has_object_key(e, "to"), "edge[" + std::to_string(i) + "]: missing 'to'");
    require(e.at("from").is_string(), "edge[" + std::to_string(i) + "]: 'from' must be string");
    require(e.at("to").is_string(), "edge[" + std::to_string(i) + "]: 'to' must be string");

    const auto from    = e.at("from").get<std::string>();
    const auto to      = e.at("to").get<std::string>();
    const auto it_from = name_to_v.find(from);
    const auto it_to   = name_to_v.find(to);

    require(it_from != name_to_v.end(),
            "edge[" + std::to_string(i) + "]: unknown 'from' node '" + from + "'");
    require(it_to != name_to_v.end(),
            "edge[" + std::to_string(i) + "]: unknown 'to' node '" + to + "'");

    EdgeSpec es{};
    require(e.at("out").is_number_integer(), "edge[" + std::to_string(i) + "]: 'out' must be int");
    require(e.at("in").is_number_integer(), "edge[" + std::to_string(i) + "]: 'in' must be int");
    es.out_idx = e.at("out").get<int>();
    es.in_idx  = e.at("in").get<int>();
    add_edge(it_from->second, it_to->second, es, g);
  }

  return g;
}

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
      if (ns.debug && !ns.settings.is_null() && !(ns.settings.is_object() && ns.settings.empty())) {
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