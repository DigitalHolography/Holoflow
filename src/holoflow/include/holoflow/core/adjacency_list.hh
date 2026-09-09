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

namespace holoflow::core {

class DetectedCycleError : std::runtime_error {
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
      typename G::OutEdge; // TODO add constaint such as get_source etc

      typename G::VertexContainer;
      typename G::EdgeContainer;

      g.add_vertex(vp);
      g.add_edge(d, ep, d);
      { g.num_vertices() } -> std::convertible_to<size_t>;
      { g[d] } -> std::same_as<typename G::VertexProperties &>;
      { gc[d] } -> std::same_as<const typename G::VertexProperties &>;
      typename G::VertexIterator;
      std::forward_iterator<typename G::VertexIterator>;
      typename G::InEdgeIterator;
      std::forward_iterator<typename G::InEdgeIterator>;
      typename G::OutEdgeIterator;
      std::forward_iterator<typename G::OutEdgeIterator>;
    };

// TODO add concepts for containers
template <template <typename> class VContainer, template <typename> class EContainer,
          typename VProps, typename EProps, typename VDescriptor = size_t>
class AdjacencyList {
public:
  using VertexProperties = VProps;
  using EdgeProperties   = EProps;
  using VertexDescriptor = VDescriptor;

  using VertexIterator  = declval<decltype(this)>().make_vertex_range().begin();
  using OutEdgeIterator = EContainer<OutEdge>::iterator;
  using InEdgeIterator  = EContainer<InEdge>::iterator;

  using VertexContainer = VContainer;
  using EdgeContainer   = EContainer;

  class OutEdge {
  public:
    OutEdge(const VDescriptor &target, const EProps &props);

    inline VDescriptor   get_target();
    inline const EProps &get_properties() const;
    inline EProps       &get_properties();

  private:
    VDescriptor target_;
    EProps      properties_;
  };

  class InEdge {
  public:
    InEdge(const VDescriptor &source, const EProps &props);

    inline VDescriptor   get_source();
    inline const EProps &get_properties() const;
    inline EProps       &get_properties();

  private:
    VDescriptor   source_;
    const EProps &properties_;
  };

  struct Vertex {
    Vertex(const VProps &props);
    Vprops              properties;
    EContainer<OutEdge> out_edges;
    EContainer<InEdge>  in_edges;
  }

  inline void
       add_vertex(const VProps &props);
  void add_edge(VDescriptor source, const EProps &edge_props, VDescriptor target);

  inline size_t num_vertices() const;

  VProps       &operator[](VDescriptor descriptor);
  const VProps &operator[](VDescriptor descriptor) const;

  inline auto make_vertex_range() { return std::ranges::views::iota(0, num_vertices()); }

  inline auto make_out_edges_range(VDescriptor d) {
    return std::tie(adjacency_list_[d].out_edges.begin(), adjacency_list_[d].out_edges.end());
  }

  inline auto make_in_edges_range(VDescriptor d) {
    return std::tie(adjacency_list_[d].in_edges.begin(), adjacency_list_[d].out_in.end());
  }

private:
  VContainer<Vertex> adjacency_list_;
};

template <typename G, typename Inserter>
  requires Graph<G> && std::output_iterator<Inserter, typename G::VertexDescriptor>
void topological_sort(const G &g, Inserter inserter);

} // namespace holoflow::core

#include "adjacency_list.hxx"