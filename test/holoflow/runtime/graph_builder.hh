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

#include <string>
#include <nlohmann/json.hpp>

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"

namespace holoflow::test {

class GraphBuilder {
public:
  GraphBuilder &add_node(const std::string &id, const std::string &name, const std::string &kind, const nlohmann::json &settings = {});
  GraphBuilder &add_node(const std::string &name, const std::string &kind);
  GraphBuilder &add_edge(const std::string &src_id, const std::string &dst_id, int out_idx = 0,
                         int in_idx = 0);

  core::GraphSpec finish();

private:
  core::GraphSpec                                                        graph_;
  std::map<std::string, core::GraphSpec::vertex_descriptor, std::less<>> nodes_;
};

} // namespace holoflow::test