// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"
#include "holotask/syncs/flatfield.hh"

#include <array>
#include <map>
#include <set>

namespace holoflow::python {

inline std::optional<core::GraphSpec>
lower_spatial_flatfield_tasks(const runtime::GraphPlan &graph, const core::Registry &registry) {
  using Vertex  = runtime::GraphPlan::vertex_descriptor;
  using Port    = std::pair<Vertex, int>;
  using Factory = holotask::syncs::FusedSpatialFlatfieldFactory;
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

  struct Pass {
    Port                  source, weights;
    std::array<Vertex, 4> absorbed;
  };
  const auto match_pass = [&](Port sum, int axis) -> std::optional<Pass> {
    if (!exclusive(sum, "Sum"))
      return {};
    const auto &reduce = graph[sum.first].spec.settings;
    if (reduce.value("axis", nlohmann::json{}) != nlohmann::json::array({-1}) ||
        reduce.value("keepdims", false))
      return {};
    auto mul = input(sum.first, 0);
    if (!mul || !exclusive(*mul, "Multiply"))
      return {};
    auto windows = input(mul->first, 0);
    auto weights = input(mul->first, 1);
    if (!windows || !weights || !exclusive(*windows, "SlidingWindowView") ||
        graph[weights->first].infer.constness != core::ConstInferenceState::Constant)
      return {};
    auto pad = input(windows->first, 0);
    if (!pad || !exclusive(*pad, "Pad"))
      return {};
    auto source = input(pad->first, 0);
    if (!source)
      return {};

    const auto &image  = desc(*source);
    const auto &kernel = desc(*weights);
    if (image.rank() < 2 || kernel.rank() != 1 || kernel.shape[0] % 2 != 1 ||
        desc(sum).shape != image.shape)
      return {};
    const auto &view         = graph[windows->first].spec.settings;
    const int   spatial_axis = static_cast<int>(image.rank()) + axis;
    const int   view_axis    = view.at("axis").get<int>();
    if ((view_axis != axis && view_axis != spatial_axis) ||
        view.at("window_shape") != kernel.shape[0])
      return {};
    const auto &padding = graph[pad->first].spec.settings;
    const auto  radius  = kernel.shape[0] / 2;
    auto        widths  = nlohmann::json::array();
    for (size_t i = 0; i < image.rank(); ++i)
      widths.push_back(static_cast<int>(i) == spatial_axis ? nlohmann::json::array({radius, radius})
                                                           : nlohmann::json::array({0, 0}));
    if (padding.at("mode") != "edge" || padding.at("pad_width") != widths)
      return {};
    return Pass{*source, *weights, {pad->first, windows->first, mul->first, sum.first}};
  };

  for (auto terminal : boost::make_iterator_range(vertices(graph))) {
    if (graph[terminal].spec.kind != "Subtract")
      continue;
    auto image      = input(terminal, 0);
    auto background = input(terminal, 1);
    if (!image || !background)
      continue;
    const auto vertical = match_pass(*background, -2);
    if (!vertical)
      continue;
    const auto horizontal = match_pass(vertical->source, -1);
    if (!horizontal || horizontal->source != *image)
      continue;

    std::vector<Port> ports{*image, horizontal->weights, vertical->weights};
    const std::array  inputs{desc(ports[0]), desc(ports[1]), desc(ports[2])};
    auto              node = graph[terminal].spec;
    node.kind              = "FusedSpatialFlatfield";
    node.settings          = nlohmann::json::object();
    try {
      const auto output = registry.get_sync(node.kind).infer(inputs, node.settings).output_descs[0];
      const auto &original = desc({terminal, 0});
      if (output.shape != original.shape || output.dtype != original.dtype ||
          output.strides != original.strides || output.offset != original.offset ||
          output.mem_loc != original.mem_loc)
        continue;
    } catch (const std::invalid_argument &) {
      continue;
    }

    auto names = nlohmann::json::array();
    for (const auto &pass : {*horizontal, *vertical}) {
      for (auto vertex : pass.absorbed) {
        names.push_back(graph[vertex].spec.name);
        removed.insert(vertex);
      }
    }
    names.push_back(node.name);
    node.settings["absorbed"] = std::move(names);
    replacements.emplace(terminal, Replacement{std::move(node), std::move(ports)});
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
