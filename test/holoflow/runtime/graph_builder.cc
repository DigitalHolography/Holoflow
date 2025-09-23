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


#include "graph_builder.hh"

#include <string>
#include <vector>

#include <boost/graph/graph_traits.hpp>
#include <nlohmann/json.hpp>

namespace holoflow::test {

  GraphBuilder &GraphBuilder::add_node(const std::string &id, const std::string &name, const std::string &kind, const nlohmann::json &settings) {
    core::NodeSpec node;
    node.name     = name;
    node.kind     = kind;
    node.settings = settings;
    auto v        = boost::add_vertex(node, graph_);
    nodes_.emplace(id, v);
    return *this;
  }

  GraphBuilder &GraphBuilder::add_node(const std::string &name, const std::string &kind) {
    return add_node(name, name, kind);
  }

  GraphBuilder &GraphBuilder::add_edge(const std::string &src_id, const std::string &dst_id, int out_idx,
                         int in_idx) {
    core::EdgeSpec edge{out_idx, in_idx};
    boost::add_edge(nodes_.at(src_id), nodes_.at(dst_id), edge, graph_);
    return *this;
  }

  core::GraphSpec GraphBuilder::finish() { return std::move(graph_); }


} // namespace holoflow::test