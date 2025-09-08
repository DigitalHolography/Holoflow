// Copyright 2025 Digital Holography Foundation
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

#include <boost/graph/adjacency_list.hpp>
#include <nlohmann/json.hpp>
#include <string>

namespace holoflow::core {

/// Specification of a computation node.
struct NodeSpec {
  std::string    name;     ///< Unique node name.
  std::string    kind;     ///< Task kind.
  nlohmann::json settings; ///< Task configuration.
};

/// Specification of a directed edge between two nodes.
struct EdgeSpec {
  int out_idx; ///< Output index of the source node.
  int in_idx;  ///< Input index of the destination node.
};

/// Directed graph specification of a computation pipeline.
using GraphSpec = boost::adjacency_list<boost::vecS,           // OutEdgeList
                                        boost::vecS,           // VertexList
                                        boost::bidirectionalS, // Directed graph
                                        NodeSpec,              // Vertex properties
                                        EdgeSpec               // Edge properties
                                        >;

/// Options for serializing a graph specification.
struct GraphSpecWriteOptions {
  bool include_node_names    = true; ///< Whether to include node names in the output.
  bool include_node_kinds    = true; ///< Whether to include node kinds in the output.
  bool include_node_settings = true; ///< Whether to include node settings in the output.
  bool include_edge_indices  = true; ///< Whether to include edge indices in the output.
};

/// Serialize a graph specification to JSON.
/// @param g     Graph specification to serialize.
/// @param opts  Serialization options.
/// @return      JSON representation of the graph specification.
nlohmann::json to_json(const GraphSpec &g, const GraphSpecWriteOptions &opts = {});

/// Deserialize a graph specification from JSON.
/// @param j  JSON representation of the graph specification.
/// @return   Deserialized graph specification.
GraphSpec from_json(const nlohmann::json &j);

/// Serialize a graph specification to a dot format string.
/// @param g     Graph specification to serialize.
/// @return      Dot format representation of the graph specification.
std::string to_dot(const GraphSpec &g);

} // namespace holoflow::core