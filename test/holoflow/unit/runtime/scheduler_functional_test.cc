// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <boost/graph/adjacency_list.hpp>
#include <chrono>
#include <memory>
#include <optional>
#include <ranges>
#include <thread>

#include "holoflow/runtime/compiler.hh"
#include "support/math_tasks.hh"

namespace {

using holoflow::core::EdgeSpec;
using holoflow::core::GraphSpec;
using holoflow::core::NodeSpec;
using holoflow::runtime::CompilerOutput;
using holoflow::runtime::GraphPlan;
using holoflow::runtime::Scheduler;
using holoflow::runtime::Section;

constexpr auto scheduler_timeout = std::chrono::seconds{5};

GraphSpec serial_async_graph() {
  GraphSpec graph;
  auto      source   = add_vertex(NodeSpec{"source", "source", {}}, graph);
  auto      bridge_a = add_vertex(NodeSpec{"bridge-a", "bridge", {}}, graph);
  auto      scale_a  = add_vertex(NodeSpec{"scale-a", "double", {}}, graph);
  auto      bridge_b = add_vertex(NodeSpec{"bridge-b", "bridge", {}}, graph);
  auto      scale_b  = add_vertex(NodeSpec{"scale-b", "triple", {}}, graph);
  auto      sink     = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(source, bridge_a, EdgeSpec{0, 0}, graph);
  add_edge(bridge_a, scale_a, EdgeSpec{0, 0}, graph);
  add_edge(scale_a, bridge_b, EdgeSpec{0, 0}, graph);
  add_edge(bridge_b, scale_b, EdgeSpec{0, 0}, graph);
  add_edge(scale_b, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

GraphSpec parallel_fan_in_graph() {
  GraphSpec graph;
  auto      left         = add_vertex(NodeSpec{"left", "left", {}}, graph);
  auto      right        = add_vertex(NodeSpec{"right", "right", {}}, graph);
  auto      left_bridge  = add_vertex(NodeSpec{"left-bridge", "bridge", {}}, graph);
  auto      right_bridge = add_vertex(NodeSpec{"right-bridge", "bridge", {}}, graph);
  auto      left_scale   = add_vertex(NodeSpec{"left-scale", "double", {}}, graph);
  auto      right_scale  = add_vertex(NodeSpec{"right-scale", "half", {}}, graph);
  auto      add          = add_vertex(NodeSpec{"add", "add", {}}, graph);
  auto      sink         = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(left, left_bridge, EdgeSpec{0, 0}, graph);
  add_edge(right, right_bridge, EdgeSpec{0, 0}, graph);
  add_edge(left_bridge, left_scale, EdgeSpec{0, 0}, graph);
  add_edge(right_bridge, right_scale, EdgeSpec{0, 0}, graph);
  add_edge(left_scale, add, EdgeSpec{0, 0}, graph);
  add_edge(right_scale, add, EdgeSpec{0, 1}, graph);
  add_edge(add, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

GraphSpec diamond_graph() {
  GraphSpec graph;
  auto      source   = add_vertex(NodeSpec{"source", "source", {}}, graph);
  auto      bridge_a = add_vertex(NodeSpec{"bridge-a", "bridge", {}}, graph);
  auto      bridge_b = add_vertex(NodeSpec{"bridge-b", "bridge", {}}, graph);
  auto      scale_a  = add_vertex(NodeSpec{"scale-a", "double", {}}, graph);
  auto      scale_b  = add_vertex(NodeSpec{"scale-b", "triple", {}}, graph);
  auto      add      = add_vertex(NodeSpec{"add", "add", {}}, graph);
  auto      sink     = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(source, bridge_a, EdgeSpec{0, 0}, graph);
  add_edge(source, bridge_b, EdgeSpec{0, 0}, graph);
  add_edge(bridge_a, scale_a, EdgeSpec{0, 0}, graph);
  add_edge(bridge_b, scale_b, EdgeSpec{0, 0}, graph);
  add_edge(scale_a, add, EdgeSpec{0, 0}, graph);
  add_edge(scale_b, add, EdgeSpec{0, 1}, graph);
  add_edge(add, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

GraphSpec multi_frame_graph() {
  GraphSpec graph;
  auto      source = add_vertex(NodeSpec{"source", "sequence", {}}, graph);
  auto      bridge = add_vertex(NodeSpec{"bridge", "bridge", {}}, graph);
  auto      scale  = add_vertex(NodeSpec{"scale", "scale", {}}, graph);
  auto      sink   = add_vertex(NodeSpec{"sink", "history", {}}, graph);
  add_edge(source, bridge, EdgeSpec{0, 0}, graph);
  add_edge(bridge, scale, EdgeSpec{0, 0}, graph);
  add_edge(scale, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

GraphSpec consecutive_async_graph() {
  GraphSpec graph;
  auto      source   = add_vertex(NodeSpec{"source", "source", {}}, graph);
  auto      bridge_a = add_vertex(NodeSpec{"bridge-a", "bridge", {}}, graph);
  auto      bridge_b = add_vertex(NodeSpec{"bridge-b", "bridge", {}}, graph);
  auto      sink     = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(source, bridge_a, EdgeSpec{0, 0}, graph);
  add_edge(bridge_a, bridge_b, EdgeSpec{0, 0}, graph);
  add_edge(bridge_b, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

bool contains_node(const GraphPlan &graph, const auto &vertices, std::string_view name) {
  return std::ranges::any_of(vertices,
                             [&](auto vertex) { return graph[vertex].spec.name == name; });
}

const Section *sync_section_for(const CompilerOutput &output, std::string_view name) {
  auto it = std::ranges::find_if(output.sections, [&](const Section &section) {
    return contains_node(output.graph, section.sync_topo, name);
  });
  return it == output.sections.end() ? nullptr : std::addressof(*it);
}

bool run_to_completion(Scheduler &scheduler) {
  std::atomic<bool> completed{false};
  std::atomic<bool> timed_out{false};
  scheduler.start();
  std::jthread watchdog([&](std::stop_token stop_token) {
    const auto deadline = std::chrono::steady_clock::now() + scheduler_timeout;
    while (!stop_token.stop_requested() && !completed.load() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    if (!stop_token.stop_requested() && !completed.load()) {
      timed_out.store(true);
      scheduler.request_stop();
    }
  });
  scheduler.wait();
  completed.store(true);
  watchdog.request_stop();
  return !timed_out.load();
}

bool run_until_progress_then_cancel(Scheduler                                        &scheduler,
                                    const std::shared_ptr<holoflow::test::MathState> &state,
                                    int minimum_frames) {
  scheduler.start();
  const auto deadline = std::chrono::steady_clock::now() + scheduler_timeout;
  while (state->sink_calls.load() < minimum_frames && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  const bool progressed = state->sink_calls.load() >= minimum_frames;
  scheduler.request_stop();
  scheduler.wait();
  return progressed;
}

void expect_frames(const holoflow::test::MathState &state, std::span<const float> base,
                   float factor, size_t frame_count) {
  ASSERT_EQ(state.collected_frames.size(), frame_count);
  for (size_t frame = 0; frame < frame_count; ++frame) {
    ASSERT_EQ(state.collected_frames[frame].size(), base.size());
    for (size_t element = 0; element < base.size(); ++element) {
      EXPECT_FLOAT_EQ(state.collected_frames[frame][element],
                      (base[element] + static_cast<float>(frame)) * factor)
          << "frame=" << frame << " element=" << element;
    }
  }
}

std::unique_ptr<CompilerOutput> compile_multi_frame_graph(
    holoflow::core::Registry &registry, const std::shared_ptr<holoflow::test::MathState> &state,
    std::span<const float> base, size_t frame_count, uint32_t seed, bool never_finish = false) {
  registry.register_sync("sequence", std::make_unique<holoflow::test::SequenceSourceFactory>(
                                         std::vector<float>(base.begin(), base.end()), state));
  registry.register_async(
      "bridge", std::make_unique<holoflow::test::AsyncBridgeFactory>(state, seed, 7, 2, 2));
  registry.register_sync("scale", std::make_unique<holoflow::test::ScaleFactory>(2.F, state));
  registry.register_sync("history", std::make_unique<holoflow::test::HistoryCollectFactory>(
                                        never_finish ? 0 : frame_count, state));
  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  return compiler.compile(multi_frame_graph());
}

} // namespace

TEST(SchedulerFunctionalTest, ExecutesSerialThreeSectionPipeline) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<holoflow::test::VectorSourceFactory>(
                                       std::vector<float>{1, 2, 3, 4}, state));
  registry.register_async("bridge", std::make_unique<holoflow::test::AsyncBridgeFactory>(state));
  registry.register_sync("double", std::make_unique<holoflow::test::ScaleFactory>(2.F, state));
  registry.register_sync("triple", std::make_unique<holoflow::test::ScaleFactory>(3.F, state));
  registry.register_sync("sink", std::make_unique<holoflow::test::CollectFactory>(state));
  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(serial_async_graph());

  ASSERT_EQ(output->sections.size(), 3);
  const auto *source_section = sync_section_for(*output, "source");
  const auto *middle_section = sync_section_for(*output, "scale-a");
  const auto *sink_section   = sync_section_for(*output, "sink");
  ASSERT_NE(source_section, nullptr);
  ASSERT_NE(middle_section, nullptr);
  ASSERT_NE(sink_section, nullptr);
  EXPECT_FALSE(source_section->has_synchronizing_async_producer);
  EXPECT_FALSE(middle_section->has_synchronizing_async_producer);
  EXPECT_TRUE(contains_node(output->graph, source_section->async_prod, "bridge-a"));
  EXPECT_TRUE(contains_node(output->graph, middle_section->async_cons, "bridge-a"));
  EXPECT_TRUE(contains_node(output->graph, middle_section->async_prod, "bridge-b"));
  EXPECT_TRUE(contains_node(output->graph, sink_section->async_cons, "bridge-b"));

  Scheduler scheduler(output->graph, output->sections, output->resources);
  ASSERT_TRUE(run_to_completion(scheduler));
  EXPECT_EQ(state->collected, (std::vector<float>{6, 12, 18, 24}));
  EXPECT_GE(state->async_push_calls, 2);
  EXPECT_GE(state->async_pop_calls, 2);
}

TEST(SchedulerFunctionalTest, ExecutesParallelProducerSectionsAndFanIn) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_sync("left", std::make_unique<holoflow::test::VectorSourceFactory>(
                                     std::vector<float>{1, 2}, state));
  registry.register_sync("right", std::make_unique<holoflow::test::VectorSourceFactory>(
                                      std::vector<float>{10, 20}, state));
  registry.register_async("bridge", std::make_unique<holoflow::test::AsyncBridgeFactory>(state));
  registry.register_sync("double", std::make_unique<holoflow::test::ScaleFactory>(2.F, state));
  registry.register_sync("half", std::make_unique<holoflow::test::ScaleFactory>(0.5F, state));
  registry.register_sync("add", std::make_unique<holoflow::test::AddFactory>(state));
  registry.register_sync("sink", std::make_unique<holoflow::test::CollectFactory>(state));
  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(parallel_fan_in_graph());

  ASSERT_EQ(output->sections.size(), 3);
  const auto *left_section  = sync_section_for(*output, "left");
  const auto *right_section = sync_section_for(*output, "right");
  const auto *sink_section  = sync_section_for(*output, "sink");
  ASSERT_NE(left_section, nullptr);
  ASSERT_NE(right_section, nullptr);
  ASSERT_NE(sink_section, nullptr);
  EXPECT_NE(left_section->id, right_section->id);
  EXPECT_TRUE(contains_node(output->graph, left_section->async_prod, "left-bridge"));
  EXPECT_TRUE(contains_node(output->graph, right_section->async_prod, "right-bridge"));
  EXPECT_TRUE(contains_node(output->graph, sink_section->async_cons, "left-bridge"));
  EXPECT_TRUE(contains_node(output->graph, sink_section->async_cons, "right-bridge"));

  Scheduler scheduler(output->graph, output->sections, output->resources);
  ASSERT_TRUE(run_to_completion(scheduler));
  EXPECT_EQ(state->collected, (std::vector<float>{7, 14}));
}

TEST(SchedulerFunctionalTest, ExecutesAsyncDiamondFanOutAndFanIn) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<holoflow::test::VectorSourceFactory>(
                                       std::vector<float>{1, 2, 3}, state));
  registry.register_async("bridge", std::make_unique<holoflow::test::AsyncBridgeFactory>(state));
  registry.register_sync("double", std::make_unique<holoflow::test::ScaleFactory>(2.F, state));
  registry.register_sync("triple", std::make_unique<holoflow::test::ScaleFactory>(3.F, state));
  registry.register_sync("add", std::make_unique<holoflow::test::AddFactory>(state));
  registry.register_sync("sink", std::make_unique<holoflow::test::CollectFactory>(state));
  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(diamond_graph());

  ASSERT_EQ(output->sections.size(), 2);
  const auto *source_section = sync_section_for(*output, "source");
  const auto *sink_section   = sync_section_for(*output, "sink");
  ASSERT_NE(source_section, nullptr);
  ASSERT_NE(sink_section, nullptr);
  EXPECT_EQ(source_section->async_prod.size(), 2);
  EXPECT_EQ(sink_section->async_cons.size(), 2);

  Scheduler scheduler(output->graph, output->sections, output->resources);
  ASSERT_TRUE(run_to_completion(scheduler));
  EXPECT_EQ(state->collected, (std::vector<float>{5, 10, 15}));
}

TEST(SchedulerFunctionalTest, PreservesEveryFrameUnderBackpressure) {
  constexpr size_t         frame_count = 64;
  const std::vector<float> base{0, 10, 20, 30};
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  auto      output = compile_multi_frame_graph(registry, state, base, frame_count, 0xC0FFEEU);
  Scheduler scheduler(output->graph, output->sections, output->resources);

  ASSERT_TRUE(run_to_completion(scheduler));
  expect_frames(*state, base, 2.F, frame_count);
  EXPECT_GT(state->async_push_not_ready, 0);
  EXPECT_GT(state->async_pop_not_ready, 0);
}

class SchedulerJitterStressTest : public testing::TestWithParam<uint32_t> {};

TEST_P(SchedulerJitterStressTest, PreservesFramesForFixedSeed) {
  constexpr size_t         frame_count = 32;
  const std::vector<float> base{1, 2, 4, 8, 16, 32, 64, 128};
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  auto      output = compile_multi_frame_graph(registry, state, base, frame_count, GetParam());
  Scheduler scheduler(output->graph, output->sections, output->resources);

  ASSERT_TRUE(run_to_completion(scheduler));
  expect_frames(*state, base, 2.F, frame_count);
}

INSTANTIATE_TEST_SUITE_P(FixedSeeds, SchedulerJitterStressTest,
                         testing::Values(0x00000000U, 0x00000001U, 0x00000002U, 0x00000003U,
                                         0x12345678U, 0x23456789U, 0x3456789AU, 0x456789ABU,
                                         0x56789ABCU, 0x6789ABCDU, 0x789ABCDEU, 0x89ABCDEFU,
                                         0x9ABCDEF0U, 0xABCDEF01U, 0xC001D00DU, 0xFFFFFFFFU));

TEST(SchedulerFunctionalTest, CancelsAProgressingThreeSectionPipeline) {
  const std::vector<float> base{1, 2, 3, 4};
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  auto      output = compile_multi_frame_graph(registry, state, base, 0, 0xA5A5A5A5U, true);
  Scheduler scheduler(output->graph, output->sections, output->resources);

  EXPECT_TRUE(run_until_progress_then_cancel(scheduler, state, 16));
  EXPECT_TRUE(scheduler.stop_requested());
  EXPECT_FALSE(scheduler.is_running());
}

TEST(CompilerTest, RejectsUnsupportedConsecutiveAsyncNodes) {
  auto                     state = std::make_shared<holoflow::test::MathState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<holoflow::test::VectorSourceFactory>(
                                       std::vector<float>{1, 2}, state));
  registry.register_async("bridge", std::make_unique<holoflow::test::AsyncBridgeFactory>(state));
  registry.register_sync("sink", std::make_unique<holoflow::test::CollectFactory>(state));
  holoflow::runtime::Compiler compiler(
      registry,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});

  EXPECT_THROW((void)compiler.compile(consecutive_async_graph()), std::runtime_error);
}
