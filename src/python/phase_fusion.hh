// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"
#include "holotask/syncs/correct_phase.hh"

#include <algorithm>
#include <map>
#include <set>

namespace holoflow::python {

inline std::optional<core::GraphSpec> lower_phase_tasks(const runtime::GraphPlan &graph,
                                                        const core::Registry     &registry) {
  using Vertex  = runtime::GraphPlan::vertex_descriptor;
  using Port    = std::pair<Vertex, int>;
  using Factory = holotask::syncs::FusedPhaseCorrectionFactory;
  struct Replacement {
    core::NodeSpec    node;
    std::vector<Port> inputs;
  };
  std::map<Vertex, Replacement> replacements;
  std::set<Vertex>              removed;
  const auto                    input = [&](Vertex node, int port) -> std::optional<Port> {
    for (auto edge : boost::make_iterator_range(in_edges(node, graph))) {
      if (graph[edge].spec.in_idx == port)
        return Port{source(edge, graph), graph[edge].spec.out_idx};
    }
    return {};
  };
  const auto desc = [&](Port port) -> const core::TDesc & {
    return graph[port.first].infer.output_descs.at(port.second);
  };
  const auto exclusive = [&](Port port, std::string_view kind) {
    return port.second == 0 && graph[port.first].spec.kind == kind &&
           out_degree(port.first, graph) == 1;
  };

  for (auto terminal : boost::make_iterator_range(vertices(graph))) {
    if (graph[terminal].spec.kind != "Multiply")
      continue;
    auto field    = input(terminal, 0);
    auto rotation = input(terminal, 1);
    if (!field || !rotation || !exclusive(*rotation, "Exp"))
      continue;
    auto angle = input(rotation->first, 0);
    if (!angle || !exclusive(*angle, "Multiply"))
      continue;
    auto scale = input(angle->first, 0);
    auto phase = input(angle->first, 1);
    if (!scale || !phase || graph[scale->first].spec.kind != "AsArray")
      continue;
    const auto &scalar = graph[scale->first].spec.settings;
    if (desc(*scale).dtype != core::DType::CF32 || desc(*scale).num_elements() != 1 ||
        scalar.value("value", 1.0) != 0.0 || scalar.value("imag", 0.0) != -1.0)
      continue;

    std::vector<Vertex> absorbed{rotation->first, angle->first};
    std::vector<Port>   terms;
    auto                add_term = [&](Port term) {
      if (!exclusive(term, "Multiply") || terms.size() == 2 * Factory::max_terms)
        return false;
      auto basis       = input(term.first, 0);
      auto coefficient = input(term.first, 1);
      if (!basis || !coefficient ||
          graph[basis->first].infer.constness != core::ConstInferenceState::Constant ||
          graph[coefficient->first].spec.kind != "Slice")
        return false;
      absorbed.push_back(term.first);
      terms.insert(terms.end(), {*coefficient, *basis});
      return true;
    };

    // The demo builds a left fold. Walk from the last term back to the first.
    while (phase && exclusive(*phase, "Add")) {
      auto right = input(phase->first, 1);
      if (!right || !add_term(*right)) {
        phase.reset();
        break;
      }
      absorbed.push_back(phase->first);
      phase = input(phase->first, 0);
    }
    if (!phase || !add_term(*phase))
      continue;
    auto output_node = terminal;
    bool output_cf16 = false;
    if (exclusive(*field, "Conversion") && desc(*field).dtype == core::DType::F32 &&
        graph[field->first].spec.settings.value("strategy", "") == std::string_view("Real")) {
      auto original = input(field->first, 0);
      if (original && desc(*original).dtype == core::DType::F16 &&
          desc(*original).shape == desc(*field).shape &&
          desc(*original).strides ==
              core::TDesc(desc(*original).shape, core::DType::F16, desc(*original).mem_loc)
                  .strides) {
        absorbed.push_back(field->first);
        field = original;
        if (out_degree(terminal, graph) == 1) {
          const auto  edge     = *out_edges(terminal, graph).first;
          const auto  consumer = target(edge, graph);
          const auto &cast     = graph[consumer].spec;
          if (cast.kind == "Conversion" && graph[edge].spec.in_idx == 0 &&
              cast.settings.value("strategy", "") == std::string_view("Real") &&
              desc({consumer, 0}).dtype == core::DType::CF16) {
            output_node = consumer;
            output_cf16 = true;
            absorbed.push_back(terminal);
          }
        }
      }
    }
    std::ranges::reverse(terms);
    terms.insert(terms.begin(), *field);
    std::vector<core::TDesc> inputs;
    for (auto port : terms)
      inputs.push_back(desc(port));

    auto node     = graph[output_node].spec;
    node.kind     = "FusedPhaseCorrection";
    node.settings = {{"scalar_real", scalar.at("value")}, {"output_cf16", output_cf16}};
    try {
      const auto output = registry.get_sync(node.kind).infer(inputs, node.settings).output_descs[0];
      const auto &original = desc({output_node, 0});
      if (output.shape != original.shape || output.dtype != original.dtype ||
          output.strides != original.strides || output.offset != original.offset ||
          output.mem_loc != original.mem_loc)
        continue;
    } catch (const std::invalid_argument &) {
      continue;
    }

    auto names = nlohmann::json::array();
    for (auto vertex : absorbed)
      names.push_back(graph[vertex].spec.name);
    names.push_back(node.name);
    node.settings["absorbed"] = std::move(names);
    removed.insert(absorbed.begin(), absorbed.end());
    replacements.emplace(output_node, Replacement{std::move(node), std::move(terms)});
  }

  if (replacements.empty())
    return {};
  core::GraphSpec                                      result;
  std::map<Vertex, core::GraphSpec::vertex_descriptor> mapped;
  for (auto vertex : boost::make_iterator_range(vertices(graph))) {
    if (removed.contains(vertex))
      continue;
    const auto replacement = replacements.find(vertex);
    mapped.emplace(vertex, add_vertex(replacement == replacements.end() ? graph[vertex].spec
                                                                        : replacement->second.node,
                                      result));
  }
  for (auto edge : boost::make_iterator_range(edges(graph))) {
    const auto from = source(edge, graph), to = target(edge, graph);
    if (!removed.contains(from) && !removed.contains(to) && !replacements.contains(to))
      add_edge(mapped.at(from), mapped.at(to), graph[edge].spec, result);
  }
  for (const auto &[vertex, replacement] : replacements) {
    for (size_t i = 0; i < replacement.inputs.size(); ++i) {
      const auto [from, port] = replacement.inputs[i];
      add_edge(mapped.at(from), mapped.at(vertex), core::EdgeSpec{port, static_cast<int>(i)},
               result);
    }
  }
  return result;
}

} // namespace holoflow::python
