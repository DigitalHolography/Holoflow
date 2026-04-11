#pragma once

#ifndef HOLOVIBES__GRAPH_BUILDER_TRACER_HXX__INCLUDED
#include "graph_builder_tracer.hh"
#endif

#include "bug.hh"

namespace holovibes::pipeline {

template <class InferResult>
std::vector<GraphBuilderTracer::TDesc>
GraphBuilderTracer::wrap_infer_outputs(std::string_view node_id, V vertex,
                                       const InferResult &infer) {
  std::vector<TDesc> ys;
  ys.reserve(infer.output_descs.size());
  int out_idx = 0;
  for (const auto &base : infer.output_descs) {
    auto y     = TDesc::from_core(base);
    y.producer = TDesc::Producer{
        .node_id = std::string{node_id},
        .out_idx = out_idx,
        .vertex  = vertex,
    };
    ys.push_back(std::move(y));
    ++out_idx;
  }
  return ys;
}

template <typename SettingsT>
std::vector<GraphBuilderTracer::TDesc>
GraphBuilderTracer::make_source_sync_node(std::string_view node_name, std::string_view kind,
                                          std::string_view reg_key, const SettingsT &s,
                                          bool debug) {
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto       v       = boost::add_vertex(node_spec, g_);
  auto      &factory = reg_.get_sync(std::string{reg_key});
  const auto infer   = factory.infer(std::span<const holoflow::core::TDesc>{}, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

template <typename SettingsT>
std::vector<GraphBuilderTracer::TDesc>
GraphBuilderTracer::make_unary_sync_node(std::string_view node_name, std::string_view kind,
                                         std::string_view reg_key, const TDesc &X,
                                         const SettingsT &s, bool debug) {
  HOLOVIBES_CHECK(X.producer.has_value());
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, 0}, g_);

  auto      &factory     = reg_.get_sync(std::string{reg_key});
  const auto core_inputs = to_core_descs(std::span{&X, 1});
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

template <typename SettingsT>
std::vector<GraphBuilderTracer::TDesc>
GraphBuilderTracer::make_nary_sync_node(std::string_view node_name, std::string_view kind,
                                        std::string_view reg_key, std::span<const TDesc> inputs,
                                        const SettingsT &s, bool debug) {
  HOLOVIBES_CHECK(!inputs.empty());
  for (const auto &X : inputs) {
    HOLOVIBES_CHECK(X.producer.has_value());
  }
  HOLOVIBES_CHECK(reg_.is_sync_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &X = inputs[i];
    boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, static_cast<int>(i)}, g_);
  }

  auto      &factory     = reg_.get_sync(std::string{reg_key});
  const auto core_inputs = to_core_descs(inputs);
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

template <typename SettingsT>
std::vector<GraphBuilderTracer::TDesc>
GraphBuilderTracer::make_unary_async_node(std::string_view node_name, std::string_view kind,
                                          std::string_view reg_key, const TDesc &X,
                                          const SettingsT &s, bool debug) {
  HOLOVIBES_CHECK(X.producer.has_value());
  HOLOVIBES_CHECK(reg_.is_async_registered(std::string{reg_key}));

  holoflow::core::NodeSpec node_spec{
      .name     = std::string{node_name} + "_" + std::to_string(unique_id_counter_++),
      .kind     = std::string{kind},
      .settings = nlohmann::json(s),
      .debug    = debug,
  };

  auto v = boost::add_vertex(node_spec, g_);
  boost::add_edge(X.producer->vertex, v, {X.producer->out_idx, 0}, g_);

  auto      &factory     = reg_.get_async(std::string{reg_key});
  const auto core_inputs = to_core_descs(std::span{&X, 1});
  const auto infer       = factory.infer(core_inputs, nlohmann::json(s));
  return wrap_infer_outputs(node_name, v, infer);
}

} // namespace holovibes::pipeline
