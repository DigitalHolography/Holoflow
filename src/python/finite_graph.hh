// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0.

#pragma once

#include "array_collector.hh"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include <boost/graph/topological_sort.hpp>
#include <optional>
#include <ranges>
#include <set>

namespace holoflow::python {

// Validate the finite subset before allocating native tasks. Counts describe tensor emissions,
// not frames: reshape and reductions change tensor shape, not the emission count.
inline core::GraphSpec prepare_finite_graph(core::GraphSpec spec, core::Registry &registry) {
  bool finite = false, collecting = false;
  for (auto v : boost::make_iterator_range(vertices(spec))) {
    finite |= spec[v].kind == "Holofile" && !spec[v].settings.value("loop", true);
    collecting |= spec[v].kind == "ArrayCollector";
  }
  if (!finite && !collecting)
    return spec;
  if (!finite)
    throw std::invalid_argument("NumPy collection requires finite changing sources (loop=False)");
  const std::set<std::string> supported = {"Holofile",
                                           "ShortTimeFresnelDiffraction",
                                           "CausalSlidingAverage",
                                           "ShackHartmannSlopes",
                                           "ZernikeFromSlopes",
                                           "ZernikePhase",
                                           "CorrectPhase",
                                           "Normalize",
                                           "Wrap2Pi",
                                           "Copy",
                                           "Transpose",
                                           "Slice",
                                           "ArrayCollector",
                                           "Void",
                                           "Throughput",
                                           "Memcpy",
                                           "Conversion",
                                           "Reshape",
                                           "Mean",
                                           "Pca",
                                           "FresnelDiffraction",
                                           "Filter2D",
                                           "Flatfield",
                                           "BatchQueue",
                                           "SlidingAverage",
                                           "DualReaderBatchQueue",
                                           "Arange",
                                           "AsArray",
                                           "Meshgrid",
                                           "Square",
                                           "Add",
                                           "Multiply",
                                           "Exp",
                                           "FFT2",
                                           "Abs",
                                           "FFTShiftNp"};
  struct Metadata {
    std::vector<core::TDesc> outputs;
    std::optional<size_t>    emissions;
    bool                     constant = false;
  };
  std::vector<Metadata>                           metadata(num_vertices(spec));
  std::vector<core::GraphSpec::vertex_descriptor> topo;
  boost::topological_sort(spec, std::back_inserter(topo));
  for (auto v : std::views::reverse(topo)) {
    auto &node = spec[v];
    if (!supported.contains(node.kind))
      throw std::invalid_argument("No finite-stream contract for operator " + node.kind);
    const auto                                      degree = in_degree(v, spec);
    std::vector<core::TDesc>                        inputs(degree);
    std::vector<core::GraphSpec::vertex_descriptor> upstream(degree);
    for (auto e : boost::make_iterator_range(in_edges(v, spec))) {
      const auto port   = spec[e].in_idx;
      upstream.at(port) = source(e, spec);
      inputs.at(port)   = metadata.at(source(e, spec)).outputs.at(spec[e].out_idx);
    }
    auto &m    = metadata[v];
    m.constant = node.kind == "Arange" || node.kind == "AsArray";
    if (degree && node.kind != "ArrayCollector" && node.kind != "Void") {
      m.constant = true;
      for (auto u : upstream)
        m.constant &= metadata[u].constant;
    }
    for (auto u : upstream) {
      if (!metadata[u].constant) {
        m.emissions = metadata[u].emissions;
        break;
      }
    }
    if (node.kind == "Holofile") {
      if (node.settings.value("loop", true))
        throw std::invalid_argument(
            "Finite execution requires every changing source to use loop=False");
      const auto n = node.settings.at("end_frame").get<size_t>() -
                     node.settings.at("start_frame").get<size_t>();
      m.emissions  = n / node.settings.at("batch_size").get<size_t>();
    } else if (node.kind == "BatchQueue" && m.emissions) {
      m.emissions = checked_product(*m.emissions, inputs[0].shape[0]) /
                    node.settings.at("output_stride").get<size_t>();
    } else if (node.kind == "DualReaderBatchQueue" && m.emissions) {
      m.emissions = checked_product(*m.emissions, inputs[0].shape[0]);
    } else if (node.kind == "SlidingAverage") {
      if (inputs.size() > 1)
        m.emissions.reset();
      else if (m.emissions) {
        const auto discarded = node.settings.value("discard_first", size_t{0});
        const auto n         = *m.emissions > discarded ? *m.emissions - discarded : 0;
        const auto window    = node.settings.at("window_size").get<size_t>();
        m.emissions          = n >= window ? n - window + 1 : 0;
      }
    } else if (node.kind == "ArrayCollector") {
      if (metadata[upstream[0]].constant)
        throw std::invalid_argument("NumPy collector requires a changing stream");
      if (node.settings["length"].is_null()) {
        if (!m.emissions)
          throw std::invalid_argument("Cannot infer NumPy collection length; supply length=");
        node.settings["length"] = checked_product(
            *m.emissions, inputs[0].shape.at(node.settings.at("axis").get<size_t>()));
      }
      const auto       &producer = spec[upstream[0]];
      const core::TDesc packed(inputs[0].shape, inputs[0].dtype, inputs[0].mem_loc);
      node.settings["direct"] =
          node.settings.at("axis") == 0 && inputs[0].mem_loc == core::MemLoc::Host &&
          inputs[0].offset == 0 && inputs[0].strides == packed.strides &&
          out_degree(upstream[0], spec) == 1 &&
          (producer.kind == "Memcpy" ||
           (producer.kind == "Holofile" && producer.settings.at("load_kind") == "Live"));
    }
    const auto inferred = registry.is_sync_registered(node.kind)
                              ? registry.get_sync(node.kind).infer(inputs, node.settings)
                              : registry.get_async(node.kind).infer(inputs, node.settings);
    m.outputs           = inferred.output_descs;
  }
  return spec;
}

} // namespace holoflow::python
