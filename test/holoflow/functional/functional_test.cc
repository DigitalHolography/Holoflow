// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <boost/graph/adjacency_list.hpp>
#include <memory>

#include "holoflow/runtime/compiler.hh"
#include "support/math_tasks.hh"

namespace {

using holoflow::core::EdgeSpec;
using holoflow::core::GraphSpec;
using holoflow::core::NodeSpec;

GraphSpec sync_math_graph() {
  GraphSpec graph;
  auto      lhs   = add_vertex(NodeSpec{"lhs", "lhs", {}}, graph);
  auto      rhs   = add_vertex(NodeSpec{"rhs", "rhs", {}}, graph);
  auto      add   = add_vertex(NodeSpec{"add", "add", {}}, graph);
  auto      scale = add_vertex(NodeSpec{"scale", "scale", {}}, graph);
  auto      sink  = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(lhs, add, EdgeSpec{0, 0}, graph);
  add_edge(rhs, add, EdgeSpec{0, 1}, graph);
  add_edge(add, scale, EdgeSpec{0, 0}, graph);
  add_edge(scale, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

GraphSpec async_math_graph() {
  GraphSpec graph;
  auto      source = add_vertex(NodeSpec{"source", "source", {}}, graph);
  auto      bridge = add_vertex(NodeSpec{"bridge", "bridge", {}}, graph);
  auto      scale  = add_vertex(NodeSpec{"scale", "scale", {}}, graph);
  auto      sink   = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(source, bridge, EdgeSpec{0, 0}, graph);
  add_edge(bridge, scale, EdgeSpec{0, 0}, graph);
  add_edge(scale, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

GraphSpec representative_streaming_graph() {
  GraphSpec graph;
  auto      source          = add_vertex(NodeSpec{"source", "sequence", {}}, graph);
  auto      double_bridge   = add_vertex(NodeSpec{"double-bridge", "bridge", {}}, graph);
  auto      triple_bridge   = add_vertex(NodeSpec{"triple-bridge", "bridge", {}}, graph);
  auto      double_branch   = add_vertex(NodeSpec{"double", "double", {}}, graph);
  auto      triple_branch   = add_vertex(NodeSpec{"triple", "triple", {}}, graph);
  auto      combine         = add_vertex(NodeSpec{"combine", "add", {}}, graph);
  auto      normalize       = add_vertex(NodeSpec{"normalize", "normalize", {}}, graph);
  auto      history         = add_vertex(NodeSpec{"history", "history", {}}, graph);
  add_edge(source, double_bridge, EdgeSpec{0, 0}, graph);
  add_edge(source, triple_bridge, EdgeSpec{0, 0}, graph);
  add_edge(double_bridge, double_branch, EdgeSpec{0, 0}, graph);
  add_edge(triple_bridge, triple_branch, EdgeSpec{0, 0}, graph);
  add_edge(double_branch, combine, EdgeSpec{0, 0}, graph);
  add_edge(triple_branch, combine, EdgeSpec{0, 1}, graph);
  add_edge(combine, normalize, EdgeSpec{0, 0}, graph);
  add_edge(normalize, history, EdgeSpec{0, 0}, graph);
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

TEST(FunctionalPipelineTest, ExecutesSmallPipelineWithinFloatingPointTolerance) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_sync("lhs", std::make_unique<holoflow::test::VectorSourceFactory>(
                                    std::vector<float>{0.1F, 0.2F}, state));
  registry.register_sync("rhs", std::make_unique<holoflow::test::VectorSourceFactory>(
                                    std::vector<float>{0.3F, 0.4F}, state));
  registry.register_sync("add", std::make_unique<holoflow::test::AddFactory>(state));
  registry.register_sync("scale", std::make_unique<holoflow::test::ScaleFactory>(0.7F, state));
  registry.register_sync("sink", std::make_unique<holoflow::test::CollectFactory>(state));

  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(sync_math_graph());
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);

  scheduler.start();
  scheduler.wait();

  ASSERT_EQ(state->collected.size(), 2);
  EXPECT_NEAR(state->collected[0], (0.1 + 0.3) * 0.7, 1e-6);
  EXPECT_NEAR(state->collected[1], (0.2 + 0.4) * 0.7, 1e-6);
  EXPECT_EQ(state->source_calls, 2);
  EXPECT_EQ(state->add_calls, 1);
  EXPECT_EQ(state->scale_calls, 1);
  EXPECT_EQ(state->sink_calls, 1);
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

TEST(FunctionalPipelineTest, ExecutesRepresentativeStreamingGraphWithinTolerance) {
  constexpr size_t frame_count = 3;
  const std::vector<float> base{0.1F, 0.2F};
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_sync("sequence", std::make_unique<holoflow::test::SequenceSourceFactory>(
                                         base, state));
  registry.register_async("bridge", std::make_unique<holoflow::test::AsyncBridgeFactory>(
                                      state, 0x5EEDU, 3));
  registry.register_sync("double", std::make_unique<holoflow::test::ScaleFactory>(2.F, state));
  registry.register_sync("triple", std::make_unique<holoflow::test::ScaleFactory>(3.F, state));
  registry.register_sync("add", std::make_unique<holoflow::test::AddFactory>(state));
  registry.register_sync("normalize",
                         std::make_unique<holoflow::test::ScaleFactory>(0.1F, state));
  registry.register_sync("history", std::make_unique<holoflow::test::HistoryCollectFactory>(
                                           frame_count, state));

  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(representative_streaming_graph());
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);

  scheduler.start();
  scheduler.wait();

  ASSERT_EQ(state->collected_frames.size(), frame_count);
  for (size_t frame = 0; frame < frame_count; ++frame) {
    ASSERT_EQ(state->collected_frames[frame].size(), base.size());
    for (size_t element = 0; element < base.size(); ++element) {
      const auto expected = (base[element] + static_cast<float>(frame)) * 0.5F;
      EXPECT_NEAR(state->collected_frames[frame][element], expected, 1e-6F)
          << "frame=" << frame << " element=" << element;
    }
  }
  // Asynchronous branches may allow a few extra source frames to be in flight when the history
  // sink reaches its requested frame count and returns Eof.
  EXPECT_GE(state->source_calls, frame_count);
  EXPECT_EQ(state->add_calls, frame_count);
  EXPECT_EQ(state->scale_calls, 3 * frame_count);
  EXPECT_EQ(state->sink_calls, frame_count);
  EXPECT_GT(state->async_push_calls, 0);
  EXPECT_GT(state->async_pop_calls, 0);
}
