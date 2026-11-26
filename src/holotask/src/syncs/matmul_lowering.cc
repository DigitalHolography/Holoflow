// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holotask/syncs/matmul_lowering.hh"

#include <array>
#include <map>
#include <set>

namespace holotask::syncs {

std::optional<holoflow::core::GraphSpec>
lower_matmul_tasks(const holoflow::runtime::GraphPlan &graph,
                   const holoflow::core::Registry     &registry) {
  namespace core = holoflow::core;
  using Vertex   = holoflow::runtime::GraphPlan::vertex_descriptor;
  using Port     = std::pair<Vertex, int>;
  if (!registry.is_sync_registered("FusedMatmul"))
    return {};
  const auto input = [&](Vertex v, int port) -> std::optional<Port> {
    for (auto e : boost::make_iterator_range(in_edges(v, graph)))
      if (graph[e].spec.in_idx == port)
        return Port{source(e, graph), graph[e].spec.out_idx};
    return {};
  };
  const auto is = [&](Port p, std::string_view kind) {
    return p.second == 0 && graph[p.first].spec.kind == kind;
  };
  const auto cast = [&](Port p, std::string_view dtype) -> std::optional<Port> {
    if (is(p, "Conversion")) {
      const auto &settings = graph[p.first].spec.settings;
      if (settings.value("target", std::string{}) == dtype &&
          settings.value("strategy", std::string("Real")) == "Real")
        return input(p.first, 0);
    }
    return {};
  };
  const auto                         &factory = registry.get_sync("FusedMatmul");
  std::map<Vertex, core::NodeSpec>    replacements;
  std::map<Vertex, std::vector<Port>> rewired;
  std::set<Vertex>                    removed;
  const auto                          desc = [&](Port p) -> const core::TDesc                          &{
    return graph[p.first].infer.output_descs.at(p.second);
  };
  const auto install = [&](Vertex v, bool gram, std::vector<Port> operands) {
    auto spec     = graph[v].spec;
    spec.kind     = "FusedMatmul";
    spec.settings = {{"gram", gram}};
    replacements.emplace(v, std::move(spec));
    rewired.emplace(v, std::move(operands));
  };
  const auto eligible = [&](bool gram, std::span<const core::TDesc> descriptors) {
    try {
      (void)factory.infer(descriptors, {{"gram", gram}});
      return true;
    } catch (const std::invalid_argument &) {
      return false;
    }
  };
  const auto single = [&](Port p) { return out_degree(p.first, graph) == 1; };
  for (auto v : boost::make_iterator_range(vertices(graph))) {
    if (graph[v].spec.kind != "Matmul")
      continue;
    const auto a = input(v, 0), b = input(v, 1);
    if (!a || !b)
      continue;
    const auto is_matrix_transpose = [&](Port p) {
      return is(p, "Transpose") && graph[p.first].spec.settings.value("axes", std::vector<int>{}) ==
                                       std::vector<int>{0, 2, 1};
    };
    if (auto x = cast(*a, "F32"); x && is_matrix_transpose(*b) && input(b->first, 0) == a &&
                                  eligible(true, std::array{desc(*x)})) {
      install(v, true, {*x});
      if (single(*b)) {
        removed.insert(b->first);
        if (out_degree(a->first, graph) == 2)
          removed.insert(a->first);
      }
      continue;
    }
    const auto x = cast(*b, "F16");
    if (!x || !is(*a, "Slice") || !single(*a))
      continue;
    const auto transpose = input(a->first, 0);
    if (!transpose || !is_matrix_transpose(*transpose) || !single(*transpose))
      continue;
    const auto coefficients_cast = input(transpose->first, 0);
    if (!coefficients_cast || !single(*coefficients_cast))
      continue;
    const auto coefficients = cast(*coefficients_cast, "F16");
    if (!coefficients || desc(*coefficients).dtype != core::DType::F32)
      continue;
    auto view = registry.get_sync("Transpose")
                    .infer(std::array{desc(*coefficients)}, graph[transpose->first].spec.settings);
    view      = registry.get_sync("Slice").infer(view.output_descs, graph[a->first].spec.settings);
    if (!eligible(false, std::array{view.output_descs[0], desc(*x)}))
      continue;
    install(v, false, {*a, *x});
    rewired.emplace(transpose->first, std::vector<Port>{*coefficients});
    removed.insert(coefficients_cast->first);
    if (single(*b))
      removed.insert(b->first);
  }
  if (replacements.empty())
    return {};
  core::GraphSpec                                      spec;
  std::map<Vertex, core::GraphSpec::vertex_descriptor> vertices_by_id;
  for (auto v : boost::make_iterator_range(vertices(graph)))
    if (!removed.contains(v))
      vertices_by_id.emplace(
          v, add_vertex(replacements.contains(v) ? replacements.at(v) : graph[v].spec, spec));
  for (auto e : boost::make_iterator_range(edges(graph))) {
    const auto from = source(e, graph), to = target(e, graph);
    if (!removed.contains(from) && !removed.contains(to) && !rewired.contains(to))
      add_edge(vertices_by_id.at(from), vertices_by_id.at(to), graph[e].spec, spec);
  }
  for (const auto &[to, operands] : rewired)
    for (size_t i = 0; i < operands.size(); ++i) {
      const auto [from, port] = operands[i];
      add_edge(vertices_by_id.at(from), vertices_by_id.at(to),
               core::EdgeSpec{port, static_cast<int>(i)}, spec);
    }
  return spec;
}

} // namespace holotask::syncs
