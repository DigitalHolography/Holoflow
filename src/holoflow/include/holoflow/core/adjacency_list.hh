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

#include <concepts>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace holoflow::core {

class DetectedCycleError : public std::runtime_error {
public:
  explicit DetectedCycleError() : std::runtime_error("detected unexpected cycle") {}
};

// TODO add make_*_range
template <typename G>
concept Graph = requires(G g, const G gc, G::VertexProperties vp, G::VertexDescriptor d,
                         G::EdgeProperties ep, G::Edge e) {
  typename G::VertexDescriptor;
  typename G::VertexProperties;
  typename G::EdgeProperties;

  typename G::Edge;
  { e.source } -> std::convertible_to<typename G::VertexDescriptor>;
  { e.target } -> std::convertible_to<typename G::VertexDescriptor>;
  { e.properties } -> std::convertible_to<typename G::EdgeProperties>;

  typename G::template VContainer<bool>;

  { g.add_vertex(vp) } -> std::same_as<typename G::VertexDescriptor>;

  g.add_edge(d, ep, d);
  { g.num_vertices() } -> std::convertible_to<size_t>;
  { g[d] } -> std::same_as<typename G::VertexProperties &>;
  { gc[d] } -> std::same_as<const typename G::VertexProperties &>;

  { g.in_degree(d) } -> std::same_as<size_t>;
  { g.out_degree(d) } -> std::same_as<size_t>;

  typename G::VertexIterator;
  std::forward_iterator<typename G::VertexIterator>;
  typename G::EdgeIterator;
  std::forward_iterator<typename G::EdgeIterator>;
};

template <typename VProps, typename EProps> class AdjacencyList {
public:
  using VertexDescriptor                 = size_t;
  using EdgeDescriptor                   = size_t;
  template <typename T> using EContainer = std::vector<T>;
  template <typename T> using VContainer = std::vector<T>;

  struct Edge {
    VertexDescriptor source;
    VertexDescriptor target;
    EProps           properties;
  };

  struct Vertex {
    Vertex(const VProps &props);

    inline size_t out_degree() const;
    inline size_t in_degree() const;

    VProps                     properties;
    EContainer<EdgeDescriptor> out_edges;
    EContainer<EdgeDescriptor> in_edges;
  };

  struct EdgeTransform {
    AdjacencyList *graph;

    Edge &operator()(EdgeDescriptor descriptor) { return graph->edge(descriptor); }
  };
  struct ConstEdgeTransform {
    const AdjacencyList *graph;

    const Edge &operator()(EdgeDescriptor descriptor) const { return graph->edge(descriptor); }
  };

  using VertexProperties = VProps;
  using EdgeProperties   = EProps;

  using EdgeDescriptorContainer = EContainer<EdgeDescriptor>;

  using EdgeRange =
      std::ranges::transform_view<std::ranges::ref_view<EdgeDescriptorContainer>, EdgeTransform>;

  using ConstEdgeRange =
      std::ranges::transform_view<std::ranges::ref_view<const EdgeDescriptorContainer>,
                                  ConstEdgeTransform>;
  using EdgeIterator = EContainer<Edge>::iterator;
  using VertexIterator =
      decltype(std::ranges::iota_view<VertexDescriptor, VertexDescriptor>(0, 1).begin());

  template <typename T> using VertexContainer = VContainer<T>;

  inline VertexDescriptor add_vertex(const VProps &props); // TODO add move semantic
  EdgeDescriptor          add_edge(VertexDescriptor source, const EProps &edge_props,
                                   VertexDescriptor target);

  inline size_t num_vertices() const;
  inline size_t num_edges() const;

  VProps       &operator[](VertexDescriptor descriptor);
  const VProps &operator[](VertexDescriptor descriptor) const;

  EProps       &edge_properties(EdgeDescriptor descriptor);
  const EProps &edge_properties(EdgeDescriptor descriptor) const;

  Edge       &edge(EdgeDescriptor descriptor);
  const Edge &edge(EdgeDescriptor descriptor) const;

  inline size_t in_degree(VertexDescriptor descriptor) const;
  inline size_t out_degree(VertexDescriptor descriptor) const;

  inline auto                    make_vertices_range() const;
  inline const EContainer<Edge> &make_edges_range() const;
  inline ConstEdgeRange          make_out_edges_range(VertexDescriptor d) const;
  inline ConstEdgeRange          make_in_edges_range(VertexDescriptor d) const;

  inline EContainer<Edge> &make_edges_range();
  inline EdgeRange         make_out_edges_range(VertexDescriptor d);
  inline EdgeRange         make_in_edges_range(VertexDescriptor d);

private:
  VContainer<Vertex> adjacency_list_;
  EContainer<Edge>   edges_;
};

template <typename G, typename Inserter>
  requires Graph<G> && std::output_iterator<Inserter, typename G::VertexDescriptor>
void topological_sort(const G &g, Inserter inserter);

} // namespace holoflow::core

#include "adjacency_list.hxx"