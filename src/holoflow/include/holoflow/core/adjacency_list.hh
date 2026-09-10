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
  explicit DetectedCycleError();
};

template <typename G>
concept Graph =
    requires(G g, const G gc, G::VertexProperties vp, G::VertexDescriptor d, G::EdgeProperties ep) {
      typename G::VertexDescriptor;
      typename G::VertexProperties;
      typename G::EdgeProperties;

      typename G::InEdge;
      { G::InEdge(0, ep).get_source() } -> std::same_as<typename G::VertexDescriptor>;
      typename G::OutEdge;
      { G::OutEdge(0, ep).get_target() } -> std::same_as<typename G::VertexDescriptor>;

      typename G::template VContainer<bool>;

      { g.add_vertex(vp) } -> std::same_as<typename G::VertexDescriptor>;

      g.add_edge(d, ep, d);
      { g.num_vertices() } -> std::convertible_to<size_t>;
      { g[d] } -> std::same_as<typename G::VertexProperties &>;
      { gc[d] } -> std::same_as<const typename G::VertexProperties &>;

      { g.in_degree(d) } -> std::same_as<size_t>;
      { g.out_degree(d) } -> std::same_as<size_t>;

      typename G::VerticesIterator;
      std::forward_iterator<typename G::VerticesIterator>;
      typename G::EdgesIterator;
      std::forward_iterator<typename G::EdgesIterator>;
      typename G::InEdgesIterator;
      std::forward_iterator<typename G::InEdgesIterator>;
      typename G::OutEdgesIterator;
      std::forward_iterator<typename G::OutEdgesIterator>;
    };

template <typename VProps, typename EProps> class AdjacencyList {
public:
  using VertexDescriptor                 = size_t;
  template <typename T> using EContainer = std::vector<T>;
  template <typename T> using VContainer = std::vector<T>;

  using Edge = std::tuple<VertexDescriptor, const EProps &, VertexDescriptor>;

  class OutEdge {
  public:
    OutEdge(const VertexDescriptor &target, const EProps &props);

    inline VertexDescriptor get_target() const;
    inline const EProps    &get_properties() const;
    inline EProps          &get_properties();
    struct EdgeTransform {
      std::size_t index;

      // TODO move transformer in .hxx
      Edge operator()(const OutEdge &e) const {
        return std::tuple{index, e.get_properties(), e.get_target()};
      }
    };

  private:
    VertexDescriptor target_;
    EProps           properties_;
  };

  class InEdge {
  public:
    InEdge(const VertexDescriptor &source, const EProps &props);

    inline VertexDescriptor get_source() const;
    inline const EProps    &get_properties() const;
    inline EProps          &get_properties();

  private:
    VertexDescriptor source_;
    const EProps    &properties_;
  };

  struct Vertex {
    Vertex(const VProps &props);

    struct VertexTransform {
      const VContainer<Vertex> &adjacency_list;
      // TODO move transformer in .hxx
      auto operator()(std::size_t i) const {
        const auto &v = adjacency_list[i];

        return v.out_edges | std::views::transform(OutEdge::EdgeTransform(i));
      }
    };

    inline size_t       out_degree() const;
    inline size_t       in_degree() const;
    VProps              properties;
    EContainer<OutEdge> out_edges;
    EContainer<InEdge>  in_edges;
  };

  using VertexProperties = VProps;
  using EdgeProperties   = EProps;

  template <typename T> using VertexContainer = VContainer<T>;

  inline VertexDescriptor add_vertex(const VProps &props); // TODO add move semantic
  void add_edge(VertexDescriptor source, const EProps &edge_props, VertexDescriptor target);

  inline size_t num_vertices() const;

  VProps       &operator[](VertexDescriptor descriptor);
  const VProps &operator[](VertexDescriptor descriptor) const;

  inline size_t in_degree(VertexDescriptor descriptor) const;
  inline size_t out_degree(VertexDescriptor descriptor) const;

  inline auto                       make_vertices_range() const;
  inline auto                       make_edges_range() const;
  inline const EContainer<OutEdge> &make_out_edges_range(VertexDescriptor d) const;
  inline const EContainer<InEdge>  &make_in_edges_range(VertexDescriptor d) const;

  using VerticesIterator =
      decltype(std::ranges::iota_view<VertexDescriptor, VertexDescriptor>(0, 1).begin());
  using Indices    = std::ranges::iota_view<std::size_t, std::size_t>;
  using InnerRange = decltype(std::declval<typename Vertex::VertexTransform>()(std::size_t{}));
  using OuterRange = std::ranges::transform_view<Indices, typename Vertex::VertexTransform>;
  using EdgesRange = std::ranges::join_view<OuterRange>;

  using EdgesIterator    = std::ranges::iterator_t<EdgesRange>;
  using OutEdgesIterator = EContainer<OutEdge>::iterator;
  using InEdgesIterator  = EContainer<InEdge>::iterator;

private:
  VContainer<Vertex> adjacency_list_;
};

template <typename G, typename Inserter>
  requires Graph<G> && std::output_iterator<Inserter, typename G::VertexDescriptor>
void topological_sort(const G &g, Inserter inserter);

} // namespace holoflow::core

#include "adjacency_list.hxx"