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

#include <optional>
#include <span>
#include <stack>
#include <string>
#include <vector>

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/core/tasks.hh"

namespace holovibes::pipeline {

// GraphBuilderTracer is the tensor-descriptor tracing layer used to construct a GraphSpec.
//
// It maintains a TDesc (tensor descriptor) type that augments the core descriptor with
// producer/consumer provenance, and provides generic template helpers that register nodes
// in the graph and return traced output descriptors. Subclasses use these helpers to build
// strongly-typed wrappers for individual task kinds.
class GraphBuilderTracer {
public:
  explicit GraphBuilderTracer(holoflow::core::Registry &registry);

protected:
  using V      = holoflow::core::GraphSpec::vertex_descriptor;
  using NodeId = std::string;

  class TDesc : public holoflow::core::TDesc {
  public:
    struct Producer {
      NodeId node_id;
      int    out_idx;
      V      vertex;
    };

    struct Consumer {
      NodeId node_id;
      int    in_idx;
      V      vertex;
    };

    std::optional<Producer> producer;
    std::vector<Consumer>   consumers;

    [[nodiscard]] holoflow::core::TDesc as_core() const;
    [[nodiscard]] static TDesc          from_core(const holoflow::core::TDesc &base);
  };

  // Convert a span of TDesc to the core representation.
  [[nodiscard]] static std::vector<holoflow::core::TDesc> to_core_descs(std::span<const TDesc> src);

  // Wrap the output descriptors returned by a factory's infer() call into traced TDescs.
  template <class InferResult>
  [[nodiscard]] static std::vector<TDesc> wrap_infer_outputs(std::string_view node_id, V vertex,
                                                             const InferResult &infer);

  // Add a source (zero-input) sync node and return its output descriptors.
  template <typename SettingsT>
  std::vector<TDesc> make_source_sync_node(std::string_view node_name, std::string_view kind,
                                           std::string_view reg_key, const SettingsT &s,
                                           bool debug = true);

  // Add a unary sync node (one input) and return its output descriptors.
  template <typename SettingsT>
  std::vector<TDesc> make_unary_sync_node(std::string_view node_name, std::string_view kind,
                                          std::string_view reg_key, const TDesc &X,
                                          const SettingsT &s, bool debug = true);

  // Add an n-ary sync node (multiple inputs) and return its output descriptors.
  template <typename SettingsT>
  std::vector<TDesc> make_nary_sync_node(std::string_view node_name, std::string_view kind,
                                         std::string_view reg_key, std::span<const TDesc> inputs,
                                         const SettingsT &s, bool debug = true);

  // Add a unary async node (one input) and return its output descriptors.
  template <typename SettingsT>
  std::vector<TDesc> make_unary_async_node(std::string_view node_name, std::string_view kind,
                                           std::string_view reg_key, const TDesc &X,
                                           const SettingsT &s, bool debug = true);

  holoflow::core::Registry &reg_;
  holoflow::core::GraphSpec g_;
  std::stack<std::string>   scope_;
  size_t                    unique_id_counter_ = 0;
};

} // namespace holovibes::pipeline

#include "graph_builder_tracer.hxx"
