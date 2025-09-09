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

#include "holoflow/runtime/compiler.hh"

#include <algorithm>
#include <array>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <vector>

#include "boost/range/iterator_range_core.hpp"
#include "bug.hh"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow/runtime/graph_exec.hh"

namespace holoflow::runtime {

void Compiler::check_duplicate_names() const {
  std::set<std::string> names;
  for (const auto &v : boost::make_iterator_range(boost::vertices(gspec_))) {
    const auto &ns = gspec_[v];
    if (!names.insert(ns.name).second) {
      throw std::logic_error("Duplicate node name: " + ns.name);
    }
  }
}

void Compiler::check_duplicate_edge_dst() const {
  std::set<std::string> dsts;
  for (const auto &e : boost::make_iterator_range(boost::edges(gspec_))) {
    const auto &es    = gspec_[e];
    const auto  dst   = gspec_[boost::target(e, gspec_)];
    auto        label = fmt::format("{}:{}", dst.name, es.out_idx);
    if (!dsts.insert(label).second) {
      throw std::logic_error("Duplicate edge destination: " + label);
    }
  }
}

void Compiler::check_factories_registered() const {
  for (const auto &v : boost::make_iterator_range(boost::vertices(gspec_))) {
    const auto &ns = gspec_[v];
    if (!registry_.is_registered(ns.kind)) {
      throw std::logic_error("No factory registered for kind: " + ns.kind);
    }
  }
}

void Compiler::check_single_source() const {
  int source_count = 0;
  for (const auto &v : boost::make_iterator_range(boost::vertices(gspec_))) {
    if (boost::in_degree(v, gspec_) == 0) {
      source_count++;
    }
  }
  if (source_count == 0) {
    throw std::logic_error("No source nodes found in the graph");
  }
  if (source_count > 1) {
    throw std::logic_error("Multiple source nodes found in the graph");
  }
}

void Compiler::check_single_input() const {
  for (const auto &v : boost::make_iterator_range(boost::vertices(gspec_))) {
    if (boost::in_degree(v, gspec_) > 1) {
      const auto &np = gspec_[v];
      throw std::logic_error("Node " + np.name + " has multiple inputs");
    }
  }
}

void Compiler::build_graph_plan() {
  using VGraphSpec = core::GraphSpec::vertex_descriptor;
  using VGraphPlan = GraphPlan::vertex_descriptor;
  using VMap       = std::map<VGraphSpec, VGraphPlan>;

  auto vertices = boost::make_iterator_range(boost::vertices(gspec_));
  auto edges    = boost::make_iterator_range(boost::edges(gspec_));
  out_.graph    = GraphPlan();
  int  nid      = 0;
  VMap vmap;

  // Build vertices
  for (const auto &vs : vertices) {
    const auto &ns = gspec_[vs];
    NodePlan    np;
    np.id   = nid++;
    np.spec = ns;
    auto vp = boost::add_vertex(np, out_.graph);
    vmap.insert({vs, vp});
  }

  // Build edges
  for (const auto &e : edges) {
    const auto &es    = gspec_[e];
    const auto  vsrcp = vmap.at(boost::source(e, gspec_));
    const auto  vdstp = vmap.at(boost::target(e, gspec_));
    EdgePlan    ep;
    ep.spec = es;
    boost::add_edge(vsrcp, vdstp, ep, out_.graph);
  }
}

void Compiler::check_typing() {
  struct CheckTypingVisitor : boost::default_bfs_visitor {
  public:
    CheckTypingVisitor(GraphPlan &g, const core::Registry &reg) : g_(g), reg_(reg) {}

    void discover_vertex(GraphPlan::vertex_descriptor v, const GraphPlan &) {
      auto &np         = g_[v];
      auto  in_degree  = boost::in_degree(v, g_);
      auto  in_edges   = boost::in_edges(v, g_);
      auto  out_degree = boost::out_degree(v, g_);
      auto  out_edges  = boost::out_edges(v, g_);

      // Gather input descs by input index
      std::vector<core::TDesc> idescs(in_degree);
      for (auto e : boost::make_iterator_range(in_edges)) {
        const auto &ep            = g_[e];
        idescs.at(ep.spec.in_idx) = ep.desc;
      }

      // Call factory
      const auto &factory = reg_.get(np.spec.kind);
      const auto  infer   = factory.infer(idescs, np.spec.settings);
      np.infer            = infer;

      // Assign outgoing edge descs
      // Verify every out_idx in [0, out_degree) is connected
      std::set<int> seen;
      for (auto e : boost::make_iterator_range(out_edges)) {
        auto &ep = g_[e];
        ep.desc  = infer.output_descs.at(ep.spec.out_idx);
        seen.insert(ep.spec.out_idx);
      }

      if (seen.size() != out_degree) {
        throw std::logic_error("Not all outputs of node " + np.spec.name + " are connected");
      }
    }

  private:
    GraphPlan            &g_;
    const core::Registry &reg_;
  };

  using ColorMap = std::vector<boost::default_color_type>;
  ColorMap color_map(boost::num_vertices(out_.graph));
  auto     color_map_p = boost::make_iterator_property_map(color_map.begin(),
                                                           boost::get(boost::vertex_index, out_.graph));

  CheckTypingVisitor visitor(out_.graph, registry_);
  for (auto v : boost::make_iterator_range(boost::vertices(out_.graph))) {
    bool is_source = boost::in_degree(v, out_.graph) == 0;
    if (is_source && color_map_p[v] == boost::white_color) {
      boost::queue<GraphPlan::vertex_descriptor> q;
      boost::breadth_first_visit(out_.graph, v, q, visitor, color_map_p);
    }
  }
}

void Compiler::assign_tensor_ids() {
  struct AssignTensorIdsVisitor : boost::default_dfs_visitor {
    AssignTensorIdsVisitor(GraphPlan &g) : g_(g) {}

    void discover_vertex(GraphPlan::vertex_descriptor v, const GraphPlan &) {
      auto &np        = g_[v];
      auto  in_edges  = boost::make_iterator_range(boost::in_edges(v, g_));
      auto  out_edges = boost::make_iterator_range(boost::out_edges(v, g_));

      // Map input tids by input index
      std::vector<int> input_tids(np.infer.input_descs.size(), -1);
      for (auto e : in_edges) {
        const auto &ep                = g_[e];
        input_tids.at(ep.spec.in_idx) = ep.tid;
      }

      // Precompute in-place output tids
      int              n_out = np.infer.output_descs.size();
      std::vector<int> inplace_tids(n_out, -1);
      for (const auto &ip : np.infer.in_place) {
        inplace_tids.at(ip.out_idx) = input_tids.at(ip.in_idx);
      }

      // Group outgoing edges by output index
      using E = GraphPlan::edge_descriptor;
      std::vector<std::vector<E>> groups(n_out);
      for (auto e : out_edges) {
        const auto &ep = g_[e];
        groups.at(ep.spec.out_idx).push_back(e);
      }

      // Assign tids per output group
      for (int i = 0; i < n_out; i++) {
        int inplace_tid = inplace_tids.at(i);
        int tid         = inplace_tid != -1 ? inplace_tid : next_id_++;
        for (const auto &e : groups.at(i)) {
          auto &ep = g_[e];
          ep.tid   = tid;
        }
      }
    }

  private:
    GraphPlan &g_;
    int        next_id_ = 0;
  };

  using ColorMap = std::vector<boost::default_color_type>;
  ColorMap color_map(boost::num_vertices(out_.graph));
  auto     color_map_p = boost::make_iterator_property_map(color_map.begin(),
                                                           boost::get(boost::vertex_index, out_.graph));

  AssignTensorIdsVisitor visitor(out_.graph);
  for (auto v : boost::make_iterator_range(boost::vertices(out_.graph))) {
    bool is_source = boost::in_degree(v, out_.graph) == 0;
    if (is_source && color_map_p[v] == boost::white_color) {
      boost::depth_first_visit(out_.graph, v, visitor, color_map_p);
    }
  }
}

void Compiler::check_buffer_temporal_consistency() {
  for (const auto &v : boost::make_iterator_range(boost::vertices(out_.graph))) {
    const auto &np        = out_.graph[v];
    const int   n_out     = np.infer.output_descs.size();
    const auto  out_edges = boost::make_iterator_range(boost::out_edges(v, out_.graph));

    // Count temporal restrictions per output index
    std::vector<int> restricted(n_out, 0);
    for (const auto &e : out_edges) {
      const auto &ep  = out_.graph[e];
      const auto &dst = out_.graph[boost::target(e, out_.graph)];

      const bool owned   = dst.infer.owned_inputs.at(ep.spec.in_idx);
      const bool inplace = std::ranges::any_of(
          dst.infer.in_place, [&](const auto &ip) { return ip.in_idx == ep.spec.out_idx; });

      if (owned || inplace) {
        restricted.at(ep.spec.out_idx)++;
      }
    }

    // Check no output is temporally restricted more than once
    for (int i = 0; i < n_out; i++) {
      if (restricted.at(i) > 1) {
        throw std::logic_error(fmt::format(
            "Output {} of node {} is temporally restricted more than once", i, np.spec.name));
      }
    }
  }
}

void Compiler::check_buffer_spatial_consistency() {
  // Count distinct tids
  int nb_tids = 0;
  for (const auto &v : boost::make_iterator_range(boost::vertices(out_.graph))) {
    const auto &np   = out_.graph[v];
    const auto  tids = std::array{std::span{np.in_tids}, std::span{np.out_tids}} | std::views::join;
    int         max  = std::ranges::max(tids);
    nb_tids          = std::max(nb_tids, max + 1);
  }

  // Count owner per tid
  std::vector<int> tid_owners(nb_tids, 0);
  for (const auto &v : boost::make_iterator_range(boost::vertices(out_.graph))) {
    const auto &np         = out_.graph[v];
    int         nb_inputs  = np.infer.input_descs.size();
    int         nb_outputs = np.infer.output_descs.size();

    for (int i = 0; i < nb_inputs; i++) {
      if (np.infer.owned_inputs.at(i)) {
        int tid = np.in_tids.at(i);
        tid_owners.at(tid)++;
      }
    }

    for (int i = 0; i < nb_outputs; i++) {
      if (np.infer.owned_outputs.at(i)) {
        int tid = np.out_tids.at(i);
        tid_owners.at(tid)++;
      }
    }
  }

  // Check no tid has more than one owner
  for (int tid = 0; tid < nb_tids; tid++) {
    if (tid_owners.at(tid) > 1) {
      throw std::logic_error(fmt::format("Tensor id {} has more than one owner", tid));
    }
  }
}

} // namespace holoflow::runtime