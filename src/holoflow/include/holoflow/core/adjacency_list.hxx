// Copyright 2026 Digital Holography Foundation
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

#include "bug.hh"
#include "adjacency_list.hh"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace holoflow::core {

// -------------------------------------------------------------------------------------------------
// Exceptions
// -------------------------------------------------------------------------------------------------

DetectedCycleError::DetectedCycleError() : std::runtime_error("detected unexpected cycle") {}

// -------------------------------------------------------------------------------------------------
// OutEdge
// -------------------------------------------------------------------------------------------------

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::OutEdge::OutEdge(
    const VDescriptor &target, const EProps &props)
    : target_{target}, properties_{props} {}

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
inline VDescriptor
AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::OutEdge::get_target() {
  return target_;
}

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
inline EProps &
AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::OutEdge::get_properties() {
  return properties_;
}

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
inline const EProps &
AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::OutEdge::get_properties()
    const {
  return properties_;
}

// -------------------------------------------------------------------------------------------------
// InEdge
// -------------------------------------------------------------------------------------------------

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::InEdge::InEdge(
    const VDescriptor &source, const EProps &props)
    : source_{source}, properties_{props} {}

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
inline VDescriptor
AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::InEdge::get_source() {
  return source;
}

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
inline EProps &
AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::InEdge::get_properties() {
  return properties_;
}

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
inline const EProps &
AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::InEdge::get_properties() const {
  return properties_;
}

// -------------------------------------------------------------------------------------------------
// Vertex
// -------------------------------------------------------------------------------------------------

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::Vertex::Vertex(
    const VProps &props)
    : properties{props} {}

// -------------------------------------------------------------------------------------------------
// AdjacencyList
// -------------------------------------------------------------------------------------------------

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
inline void AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::add_vertex(
    const VProps &props) {
  adjacency_list_.emplace_back(Vertex(props));
}

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
void AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::add_edge(
    VDescriptor source, const EProps &edge_props, VDescriptor target) {
  auto &v = adjacency_list_[source];
  v.out_edges.emplace_back(target, edge_props);

  auto &v2 = adjacency_list_[target];
  v.in_edges.emplace_back(source, edge_props);
}

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
VProps &AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::operator[](
    VDescriptor descriptor) {
  return adjacency_list_[descriptor].properties;
}

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
const VProps &AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::operator[](
    VDescriptor descriptor) const {
  return adjacency_list_[descriptor].properties;
}

template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor>
size_t AdjacencyList<VContainer, EContainer, VProps, EProps, VDescriptor>::num_vertices() const {
  return adjacency_list_.size();
}

template <typename G>
  requires Graph<G>
static G::VertexDescriptor
select_umarked_node(const G &g, const typename G::VertexContainer<bool> &permanent_mark,
                    const typename G::VertexContainer<bool> &temp_mark) {
  for (auto vd : g.make_vertex_range()) {
    if (!permanent_mark[vd] && !temp_mark[vd])
      return vd;
  }

  HOLOFLOW_UNREACHABLE();
}

template <typename G, typename Inserter>
  requires Graph<G> && std::output_iterator<Inserter, typename G::VertexDescriptor>
static void visit(const G &g, typename G::VertexDescriptor n,
                  const typename G::VertexContainer<bool> &permanent_mark,
                  const typename G::VertexContainer<bool> &temp_mark, Inserter it) {

  if (permanent_mark[n])
    return;
  else if (temp_mark[n])
    throw DetectedCycleError();

  temp_mark[n] = true;

  auto nodes =
      g.make_vertex_range() | std::ranges::views::filter([const & ](G::VertexDescriptor d) {
        return std::ranges::any_of(g.make_in_edges_range(d),
                                   [const & ](G::InEdge &e) { return e.get_source() == n; });
      });
  std::ranges::for_each(
      nodes, [const & ](G::VertexDescriptor d) { visit(g, d, permanent_mark, temp_mark) });

  permanent_mark[n] = true;

  *it = n;
  it++;
}

template <typename G, typename Inserter>
  requires Graph<G> && std::output_iterator<Inserter, typename G::VertexDescriptor>
void topological_sort(const G &g, Inserter inserter) {
  auto permanent_mark = G::VertexContainer<bool>{};
  auto temporary_mark = G::VertexContainer<bool>{};

  while (permanent_mark.size() != g.num_vertices()) {
    auto n = select_umarked_node(g, permanent_mark, temporary_mark);
    visit(g, n, permanent_mark, temporary_mark, inserter);
  }
}

} // namespace holoflow::core