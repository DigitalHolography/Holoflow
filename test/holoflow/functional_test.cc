// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <memory>

#include "holoflow/runtime/compiler.hh"
#include "support/math_tasks.hh"

namespace {

using holoflow::core::EdgeSpec;
using holoflow::core::GraphSpec;
using holoflow::core::NodeSpec;

GraphSpec sync_math_graph() {
  GraphSpec graph;
  auto      lhs   = graph.add_vertex(NodeSpec{"lhs", "lhs", {}});
  auto      rhs   = graph.add_vertex(NodeSpec{"rhs", "rhs", {}});
  auto      add   = graph.add_vertex(NodeSpec{"add", "add", {}});
  auto      scale = graph.add_vertex(NodeSpec{"scale", "scale", {}});
  auto      sink  = graph.add_vertex(NodeSpec{"sink", "sink", {}});
  graph.add_edge(lhs, EdgeSpec{0, 0}, add);
  graph.add_edge(rhs, EdgeSpec{0, 1}, add);
  graph.add_edge(add, EdgeSpec{0, 0}, scale);
  graph.add_edge(scale, EdgeSpec{0, 0}, sink);
  return graph;
}

GraphSpec async_math_graph() {
  GraphSpec graph;
  auto      source = graph.add_vertex(NodeSpec{"source", "source", {}});
  auto      bridge = graph.add_vertex(NodeSpec{"bridge", "bridge", {}});
  auto      scale  = graph.add_vertex(NodeSpec{"scale", "scale", {}});
  auto      sink   = graph.add_vertex(NodeSpec{"sink", "sink", {}});
  graph.add_edge(source, EdgeSpec{0, 0}, bridge);
  graph.add_edge(bridge, EdgeSpec{0, 0}, scale);
  graph.add_edge(scale, EdgeSpec{0, 0}, sink);
  return graph;
}

} // namespace

TEST(FunctionalPipelineTest, CompilesAndExecutesHostVectorMath) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_sync("lhs", std::make_unique<holoflow::test::VectorSourceFactory>(
                                    std::vector<float>{1, 2, 3, 4}, state));
  registry.register_sync("rhs", std::make_unique<holoflow::test::VectorSourceFactory>(
                                    std::vector<float>{10, 20, 30, 40}, state));
  registry.register_sync("add", std::make_unique<holoflow::test::AddFactory>(state));
  registry.register_sync("scale", std::make_unique<holoflow::test::ScaleFactory>(0.5F, state));
  registry.register_sync("sink", std::make_unique<holoflow::test::CollectFactory>(state));

  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto                         output = compiler.compile(sync_math_graph());
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);

  scheduler.start();
  scheduler.wait();

  EXPECT_EQ(state->collected, (std::vector<float>{5.5F, 11.F, 16.5F, 22.F}));
  EXPECT_EQ(state->source_calls, 2);
  EXPECT_EQ(state->add_calls, 1);
  EXPECT_EQ(state->scale_calls, 1);
  EXPECT_EQ(state->sink_calls, 1);
  EXPECT_NE(state->last_sync_stream, nullptr);
}

TEST(FunctionalPipelineTest, ExecutesAcrossAnAsyncBoundary) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<holoflow::test::VectorSourceFactory>(
                                       std::vector<float>{2, 4, 6, 8}, state));
  registry.register_async("bridge", std::make_unique<holoflow::test::AsyncBridgeFactory>(state));
  registry.register_sync("scale", std::make_unique<holoflow::test::ScaleFactory>(3.F, state));
  registry.register_sync("sink", std::make_unique<holoflow::test::CollectFactory>(state));

  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(async_math_graph());
  ASSERT_EQ(output->sections.size(), 2);
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);

  scheduler.start();
  scheduler.wait();

  EXPECT_EQ(state->collected, (std::vector<float>{6, 12, 18, 24}));
  EXPECT_GT(state->async_push_calls, 0);
  EXPECT_GT(state->async_pop_calls, 0);
  EXPECT_NE(state->producer_stream, nullptr);
  EXPECT_NE(state->consumer_stream, nullptr);
  EXPECT_NE(state->producer_stream, state->consumer_stream);
}
