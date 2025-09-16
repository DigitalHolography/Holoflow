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

#include <QApplication>
#include <iostream>

#include "boost/graph/compressed_sparse_row_graph.hpp"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/graph_exec.hh"
#include "spdlog/common.h"
#include "tasks/display_tensor.hh"
#include "tasks/holofile.hh"
#include "ui/tensor_display_widget.hh"

int main() {
  spdlog::set_level(spdlog::level::debug);

  QApplication                       app(__argc, __argv);
  holovibes::ui::TensorDisplayWidget widget;
  widget.show();

  auto holofile_factory = std::make_unique<holovibes::tasks::HolofileFactory>();
  auto display_factory  = std::make_unique<holovibes::tasks::DisplayTensorFactory>(&widget);
  holoflow::core::Registry registry;
  registry.register_sync("Holofile", std::move(holofile_factory));
  registry.register_sync("DisplayTensor", std::move(display_factory));
  holoflow::runtime::Compiler compiler(registry);

  holoflow::core::GraphSpec spec;

  const holovibes::tasks::HolofileSettings holofile_settings = {
      .path        = "D:\\InputData\\250220_GUJ0206_L.holo",
      .load_kind   = holovibes::tasks::HolofileSettings::LoadKind::Live,
      .start_frame = 0,
      .end_frame   = 2048,
      .batch_size  = 1,
  };

  const holoflow::core::NodeSpec holofile_node = {
      .name     = "reader",
      .kind     = "Holofile",
      .settings = nlohmann::json(holofile_settings),
  };

  auto v0 = boost::add_vertex(holofile_node, spec);

  const holovibes::tasks::DisplayTensorSettings display_settings = {
      .refresh_rate_hz = 60.0f,
  };

  const holoflow::core::NodeSpec display_node = {
      .name     = "display",
      .kind     = "DisplayTensor",
      .settings = nlohmann::json(display_settings),
  };

  auto v1 = boost::add_vertex(display_node, spec);

  holoflow::core::EdgeSpec edge = {
      .out_idx = 0,
      .in_idx  = 0,
  };

  boost::add_edge(v0, v1, edge, spec);

  auto                         res = compiler.compile(spec);
  holoflow::runtime::Scheduler scheduler(res->graph, res->sections, res->resources);
  scheduler.start();

  return app.exec();
}