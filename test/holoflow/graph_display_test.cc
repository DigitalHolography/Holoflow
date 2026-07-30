// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <memory>

#include "holoflow/runtime/graph_display.hh"
#include "support/math_tasks.hh"

TEST(CompiledGraphDisplayTest, RendersSyncAsyncEdgesSectionsAndResources) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_async("bridge", std::make_unique<holoflow::test::AsyncBridgeFactory>(state));

  holoflow::runtime::CompilerOutput output;
  holoflow::runtime::NodePlan       source{
            .spec     = {"source", "sync", {{"text", "quoted\"\nvalue"}}},
            .infer    = {{}, {}, {}, {}, {}, holoflow::core::TaskKind::Sync},
            .in_tids  = {},
            .out_tids = {0},
  };
  holoflow::runtime::NodePlan bridge{
      .spec     = {"bridge", "bridge", {}},
      .infer    = {{}, {}, {}, {}, {}, holoflow::core::TaskKind::Async},
      .in_tids  = {0},
      .out_tids = {1},
  };
  auto source_v = add_vertex(source, output.graph);
  auto bridge_v = add_vertex(bridge, output.graph);
  add_edge(source_v, bridge_v,
           holoflow::runtime::EdgePlan{
               {0, 0},
               holoflow::core::TDesc({4}, holoflow::core::DType::F32, holoflow::core::MemLoc::Host),
               0},
           output.graph);
  output.sections.push_back({.id         = 3,
                             .name       = "producer",
                             .stream     = nullptr,
                             .sync_topo  = {source_v},
                             .async_cons = {},
                             .async_prod = {bridge_v}});
  output.resources.tensor_descs.emplace(
      0, holoflow::core::TDesc({4}, holoflow::core::DType::F32, holoflow::core::MemLoc::Host));
  output.resources.tid_to_sid.emplace(0, 9);

  const auto dot = holoflow::runtime::to_dot(output, registry);
  EXPECT_NE(dot.find("digraph holoflow_compiled"), std::string::npos);
  EXPECT_NE(dot.find("shape=octagon"), std::string::npos);
  EXPECT_NE(dot.find("color=blue"), std::string::npos);
  EXPECT_NE(dot.find("tid:0"), std::string::npos);
  EXPECT_NE(dot.find("Section 3: producer"), std::string::npos);
  EXPECT_NE(dot.find("quoted"), std::string::npos);
  EXPECT_NE(dot.find("value"), std::string::npos);
}

TEST(CompiledGraphDisplayTest, RendersEmptyOutput) {
  holoflow::runtime::CompilerOutput output;
  holoflow::core::Registry          registry;
  const auto                        dot = holoflow::runtime::to_dot(output, registry);
  EXPECT_NE(dot.find("// streams:"), std::string::npos);
  EXPECT_NE(dot.find("// tasks:"), std::string::npos);
  EXPECT_NE(dot.find("}\n"), std::string::npos);
}
