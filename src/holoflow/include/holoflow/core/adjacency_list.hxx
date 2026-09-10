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

#include <algorithm>
#include <queue>
#include <stdexcept>

#include "adjacency_list.hh"

namespace holoflow::core {

// -------------------------------------------------------------------------------------------------
// Exceptions
// -------------------------------------------------------------------------------------------------

DetectedCycleError::DetectedCycleError() : std::runtime_error("detected unexpected cycle") {}

// -------------------------------------------------------------------------------------------------
// OutEdge
// -------------------------------------------------------------------------------------------------

template <typename VProps, typename EProps>
AdjacencyList<VProps, EProps>::OutEdge::OutEdge(const VertexDescriptor &target, const EProps &props)
    : target_{target}, properties_{props} {}

template <typename VProps, typename EProps>
inline AdjacencyList<VProps, EProps>::VertexDescriptor
AdjacencyList<VProps, EProps>::OutEdge::get_target() const {
  return target_;
}

template <typename VProps, typename EProps>
inline EProps &AdjacencyList<VProps, EProps>::OutEdge::get_properties() {
  return properties_;
}

template <typename VProps, typename EProps>
inline const EProps &AdjacencyList<VProps, EProps>::OutEdge::get_properties() const {
  return properties_;
}

// -------------------------------------------------------------------------------------------------
// InEdge
// -------------------------------------------------------------------------------------------------

template <typename VProps, typename EProps>
AdjacencyList<VProps, EProps>::InEdge::InEdge(const VertexDescriptor &source, const EProps &props)
    : source_{source}, properties_{props} {}

template <typename VProps, typename EProps>
inline AdjacencyList<VProps, EProps>::VertexDescriptor
AdjacencyList<VProps, EProps>::InEdge::get_source() const {
  return source_;
}

template <typename VProps, typename EProps>
inline EProps &AdjacencyList<VProps, EProps>::InEdge::get_properties() {
  return properties_;
}

template <typename VProps, typename EProps>
inline const EProps &AdjacencyList<VProps, EProps>::InEdge::get_properties() const {
  return properties_;
}

// -------------------------------------------------------------------------------------------------
// Vertex
// -------------------------------------------------------------------------------------------------

template <typename VProps, typename EProps>
AdjacencyList<VProps, EProps>::Vertex::Vertex(const VProps &props) : properties{props} {}

template <typename VProps, typename EProps>
inline size_t AdjacencyList<VProps, EProps>::Vertex::out_degree() const {
  return out_edges.size();
}

template <typename VProps, typename EProps>
size_t AdjacencyList<VProps, EProps>::Vertex::in_degree() const {
  return in_edges.size();
}
// -------------------------------------------------------------------------------------------------
// AdjacencyList
// -------------------------------------------------------------------------------------------------

template <typename VProps, typename EProps>
inline AdjacencyList<VProps, EProps>::VertexDescriptor
AdjacencyList<VProps, EProps>::add_vertex(const VProps &props) {
  adjacency_list_.emplace_back(Vertex(props));
  return num_vertices() - 1; // TODO fix for non integral VertexDescriptor
}

template <typename VProps, typename EProps>
void AdjacencyList<VProps, EProps>::add_edge(VertexDescriptor source, const EProps &edge_props,
                                             VertexDescriptor target) {
  auto &v = adjacency_list_[source];
  v.out_edges.emplace_back(target, edge_props);

  auto &v2 = adjacency_list_[target];
  v2.in_edges.emplace_back(source, edge_props);
}

template <typename VProps, typename EProps>
VProps &AdjacencyList<VProps, EProps>::operator[](VertexDescriptor descriptor) {
  return adjacency_list_[descriptor].properties;
}

template <typename VProps, typename EProps>
const VProps &AdjacencyList<VProps, EProps>::operator[](VertexDescriptor descriptor) const {
  return adjacency_list_[descriptor].properties;
}

template <typename VProps, typename EProps>
size_t AdjacencyList<VProps, EProps>::num_vertices() const {
  return adjacency_list_.size();
}

template <typename VProps, typename EProps>
inline auto AdjacencyList<VProps, EProps>::make_vertices_range() const {
  return std::ranges::views::iota(static_cast<size_t>(0), num_vertices());
}

template <typename VProps, typename EProps>
inline auto AdjacencyList<VProps, EProps>::make_edges_range() const {
  auto indices = std::views::iota(std::size_t{0}, adjacency_list_.size());

  return indices | std::views::transform(Vertex::VertexTransform(adjacency_list_)) |
         std::views::join;
}

template <typename VProps, typename EProps>
inline const AdjacencyList<VProps,
                           EProps>::EContainer<typename AdjacencyList<VProps, EProps>::OutEdge> &
AdjacencyList<VProps, EProps>::make_out_edges_range(VertexDescriptor d) const {
  return adjacency_list_[d].out_edges;
}

template <typename VProps, typename EProps>
inline const AdjacencyList<VProps,
                           EProps>::EContainer<typename AdjacencyList<VProps, EProps>::InEdge> &
AdjacencyList<VProps, EProps>::make_in_edges_range(VertexDescriptor d) const {
  return adjacency_list_[d].in_edges;
}

template <typename VProps, typename EProps>
inline size_t AdjacencyList<VProps, EProps>::in_degree(VertexDescriptor d) const {
  return adjacency_list_[d].in_degree();
}

template <typename VProps, typename EProps>
inline size_t AdjacencyList<VProps, EProps>::out_degree(VertexDescriptor d) const {
  return adjacency_list_[d].out_degree();
}

// -------------------------------------------------------------------------------------------------
// Topological sort
// -------------------------------------------------------------------------------------------------

template <typename G, typename Inserter>
  requires Graph<G> && std::output_iterator<Inserter, typename G::VertexDescriptor>
void topological_sort(const G &g, Inserter inserter) {
  auto degree_range =
      std::ranges::views::transform(g.make_vertices_range(), [&](auto d) { return g.in_degree(d); });
  auto in_degrees = G::template VertexContainer<size_t>(degree_range.begin(), degree_range.end());

  auto   nodes_with_no_incoming_edge = std::queue<size_t>();
  size_t count                       = 0;

  for (size_t i = 0; i < in_degrees.size(); i++) {
    if (in_degrees[i] == 0)
      nodes_with_no_incoming_edge.emplace(i);
  }

  while (nodes_with_no_incoming_edge.size() > 0) {
    auto n = nodes_with_no_incoming_edge.front();
    nodes_with_no_incoming_edge.pop();

    *inserter = n;
    inserter++;
    count++;

    for (auto e : g.make_out_edges_range(n)) {
      in_degrees[e.get_target()]--;
      if (in_degrees[e.get_target()] == 0)
        nodes_with_no_incoming_edge.emplace(e.get_target());
    }
  }

  if (count != g.num_vertices())
    throw DetectedCycleError();
}

} // namespace holoflow::core