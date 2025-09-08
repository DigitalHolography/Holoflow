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

#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>

#include "bug.hh"
#include "holoflow/core/graph_spec.hh"
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

void Compiler::build_graph_plan() {
  using VGraphSpec = core::GraphSpec::vertex_descriptor;
  using VGraphPlan = GraphPlan::vertex_descriptor;
  using VMap       = std::map<VGraphSpec, VGraphPlan>;

  out_.graph = GraphPlan();
  int  nid   = 0;
  VMap vmap;

  for (auto vs : boost::make_iterator_range(boost::vertices(gspec_))) {
    const auto &ns = gspec_[vs];
    NodePlan    np;
    np.id   = nid++;
    np.spec = ns;
    auto vp = boost::add_vertex(np, out_.graph);
    vmap.insert({vs, vp});
  }

  for (auto e : boost::make_iterator_range(boost::edges(gspec_))) {
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
    void discover_vertex(GraphPlan::vertex_descriptor v, const GraphPlan &) {
      auto                    &np = graph[v];
      std::vector<core::TDesc> idescs;
    }

  public:
    GraphPlan            &graph;
    const core::Registry &registry;
  };
}

} // namespace holoflow::runtime