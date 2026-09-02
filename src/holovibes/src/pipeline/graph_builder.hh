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

#include <memory>

#include "holoflow/core/graph_spec.hh"

namespace holoflow::core {
class Registry;
}

namespace holovibes::pipeline {

struct Settings;

// GraphBuilder defines the holographic computation pipeline.
//
// It translates a Settings snapshot into a GraphSpec by wiring together the acquisition,
// preprocessing, time-frequency analysis, spatial propagation, and display stages.
class GraphBuilder {
public:
  GraphBuilder(const Settings &settings, holoflow::core::Registry &registry);
  ~GraphBuilder();

  GraphBuilder(GraphBuilder &&) noexcept;
  GraphBuilder &operator=(GraphBuilder &&) noexcept;

  GraphBuilder(const GraphBuilder &)            = delete;
  GraphBuilder &operator=(const GraphBuilder &) = delete;

  holoflow::core::GraphSpec build();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace holovibes::pipeline
