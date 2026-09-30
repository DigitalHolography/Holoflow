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
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include "holoflow/runtime/compiler.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::EdgeSpec;
using holoflow::core::GraphSpec;
using holoflow::core::InferResult;
using holoflow::core::MemLoc;
using holoflow::core::NodeSpec;
using holoflow::core::OpResult;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

TDesc scalar_desc() { return TDesc({1}, DType::F32, MemLoc::Host); }

struct ExecutionState {
  std::mutex             mutex;
  std::vector<std::string> order;
  float                  result = 0.F;
  int                    terminal_calls = 0;

  void record(std::string node) {
    std::lock_guard lock(mutex);
    order.push_back(std::move(node));
  }
};

class SourceTask final : public holoflow::core::ISyncTask {
public:
  explicit SourceTask(std::shared_ptr<ExecutionState> state) : state_(std::move(state)) {}

  OpResult execute(holoflow::core::SyncCtx &ctx) override {
    *reinterpret_cast<float *>(ctx.outputs[0].data()) = 10.F;
    state_->record("source");
    return OpResult::Ok;
  }

private:
  std::shared_ptr<ExecutionState> state_;
};

class SourceFactory final : public holoflow::core::ISyncTaskFactory {
public:
  explicit SourceFactory(std::shared_ptr<ExecutionState> state) : state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (!inputs.empty()) {
      throw std::invalid_argument("source has no inputs");
    }
    return {.input_descs   = {},
            .output_descs  = {scalar_desc()},
            .in_place      = {},
            .owned_inputs  = {},
            .owned_outputs = {false},
            .kind          = TaskKind::Sync};
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<SourceTask>(state_);
  }

private:
  std::shared_ptr<ExecutionState> state_;
};

class TransformTask final : public holoflow::core::ISyncTask {
public:
  TransformTask(std::string name, std::function<float(float)> transform,
                std::shared_ptr<ExecutionState> state)
      : name_(std::move(name)), transform_(std::move(transform)), state_(std::move(state)) {}

  OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto input = *reinterpret_cast<const float *>(ctx.inputs[0].data());
    *reinterpret_cast<float *>(ctx.outputs[0].data()) = transform_(input);
    state_->record(name_);
    return OpResult::Ok;
  }

private:
  std::string                     name_;
  std::function<float(float)>     transform_;
  std::shared_ptr<ExecutionState> state_;
};

class TransformFactory final : public holoflow::core::ISyncTaskFactory {
public:
  TransformFactory(std::string name, std::function<float(float)> transform,
                   std::shared_ptr<ExecutionState> state, bool in_place = false)
      : name_(std::move(name)), transform_(std::move(transform)), state_(std::move(state)),
        in_place_(in_place) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("transform requires one input");
    }
    return {.input_descs   = {inputs[0]},
            .output_descs  = {inputs[0]},
            .in_place      = in_place_ ? std::vector<holoflow::core::InPlace>{{0, 0}}
                                       : std::vector<holoflow::core::InPlace>{},
            .owned_inputs  = {false},
            .owned_outputs = {false},
            .kind          = TaskKind::Sync};
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<TransformTask>(name_, transform_, state_);
  }

private:
  std::string                     name_;
  std::function<float(float)>     transform_;
  std::shared_ptr<ExecutionState> state_;
  bool                            in_place_;
};

class SinkTask final : public holoflow::core::ISyncTask {
public:
  explicit SinkTask(std::shared_ptr<ExecutionState> state) : state_(std::move(state)) {}

  OpResult execute(holoflow::core::SyncCtx &ctx) override {
    state_->result = *reinterpret_cast<const float *>(ctx.inputs[0].data());
    state_->record("sink");
    return OpResult::Eof;
  }

private:
  std::shared_ptr<ExecutionState> state_;
};

class SinkFactory final : public holoflow::core::ISyncTaskFactory {
public:
  explicit SinkFactory(std::shared_ptr<ExecutionState> state) : state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("sink requires one input");
    }
    return {.input_descs   = {inputs[0]},
            .output_descs  = {},
            .in_place      = {},
            .owned_inputs  = {false},
            .owned_outputs = {},
            .kind          = TaskKind::Sync};
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<SinkTask>(state_);
  }

private:
  std::shared_ptr<ExecutionState> state_;
};

class TerminalTask final : public holoflow::core::ISyncTask {
public:
  explicit TerminalTask(std::shared_ptr<ExecutionState> state) : state_(std::move(state)) {}

  OpResult execute(holoflow::core::SyncCtx &) override {
    {
      std::lock_guard lock(state_->mutex);
      ++state_->terminal_calls;
    }
    return OpResult::Eof;
  }

private:
  std::shared_ptr<ExecutionState> state_;
};

class TerminalFactory final : public holoflow::core::ISyncTaskFactory {
public:
  explicit TerminalFactory(std::shared_ptr<ExecutionState> state) : state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (!inputs.empty()) {
      throw std::invalid_argument("terminal has no inputs");
    }
    return {.input_descs   = {},
            .output_descs  = {},
            .in_place      = {},
            .owned_inputs  = {},
            .owned_outputs = {},
            .kind          = TaskKind::Sync};
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<TerminalTask>(state_);
  }

private:
  std::shared_ptr<ExecutionState> state_;
};

holoflow::core::Registry pipeline_registry(const std::shared_ptr<ExecutionState> &state,
                                           bool in_place = false) {
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<SourceFactory>(state));
  registry.register_sync("subtract", std::make_unique<TransformFactory>(
                                           "subtract", [](float value) { return value - 3.F; }, state,
                                           in_place));
  registry.register_sync("scale", std::make_unique<TransformFactory>(
                                       "scale", [](float value) { return value * 2.F; }, state));
  registry.register_sync("sink", std::make_unique<SinkFactory>(state));
  return registry;
}

GraphSpec pipeline_graph() {
  GraphSpec graph;
  const auto source   = add_vertex(NodeSpec{"source", "source", {}}, graph);
  const auto subtract = add_vertex(NodeSpec{"subtract", "subtract", {}}, graph);
  const auto scale    = add_vertex(NodeSpec{"scale", "scale", {}}, graph);
  const auto sink     = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(source, subtract, EdgeSpec{0, 0}, graph);
  add_edge(subtract, scale, EdgeSpec{0, 0}, graph);
  add_edge(scale, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

std::vector<std::string> section_node_names(const holoflow::runtime::CompilerOutput &output) {
  if (output.sections.size() != 1) {
    return {};
  }
  std::vector<std::string> names;
  for (const auto vertex : output.sections.front().sync_topo) {
    names.push_back(output.graph[vertex].spec.name);
  }
  return names;
}

} // namespace

TEST(CompilerTest, PreservesTopologicalOrderForNonCommutativePipeline) {
  auto state = std::make_shared<ExecutionState>();
  auto registry = pipeline_registry(state);
  holoflow::runtime::Compiler compiler(
      registry, {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});

  const auto output = compiler.compile(pipeline_graph());

  ASSERT_NE(output, nullptr);
  EXPECT_EQ(section_node_names(*output),
            (std::vector<std::string>{"source", "subtract", "scale", "sink"}));
}

TEST(CompilerTest, ReusesStorageForInPlaceOutput) {
  auto state = std::make_shared<ExecutionState>();
  auto registry = pipeline_registry(state, true);
  holoflow::runtime::Compiler compiler(
      registry, {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});

  const auto output = compiler.compile(pipeline_graph());
  ASSERT_NE(output, nullptr);

  holoflow::runtime::GraphPlan::vertex_descriptor subtract = 0;
  for (const auto vertex : boost::make_iterator_range(boost::vertices(output->graph))) {
    if (output->graph[vertex].spec.name == "subtract") {
      subtract = vertex;
      break;
    }
  }
  ASSERT_EQ(output->graph[subtract].spec.name, "subtract");
  ASSERT_EQ(output->graph[subtract].in_tids.size(), 1);
  ASSERT_EQ(output->graph[subtract].out_tids.size(), 1);
  EXPECT_EQ(output->resources.tid_to_sid.at(output->graph[subtract].in_tids[0]),
            output->resources.tid_to_sid.at(output->graph[subtract].out_tids[0]));
}

TEST(CompilerTest, RejectsAnEdgeTargetingAUnavailableInputPort) {
  auto state = std::make_shared<ExecutionState>();
  auto registry = pipeline_registry(state);
  holoflow::runtime::Compiler compiler(registry, {.dump_dot_on_failure = false});

  GraphSpec graph;
  const auto source = add_vertex(NodeSpec{"source", "source", {}}, graph);
  const auto sink   = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(source, sink, EdgeSpec{0, 1}, graph);

  EXPECT_THROW((void)compiler.compile(graph), std::runtime_error);
}

TEST(CompilerTest, RejectsNegativeTensorPorts) {
  auto state = std::make_shared<ExecutionState>();
  auto registry = pipeline_registry(state);
  holoflow::runtime::Compiler compiler(registry, {.dump_dot_on_failure = false});

  for (const auto ports : {EdgeSpec{-1, 0}, EdgeSpec{0, -1}}) {
    GraphSpec graph;
    const auto source = add_vertex(NodeSpec{"source", "source", {}}, graph);
    const auto sink   = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
    add_edge(source, sink, ports, graph);

    EXPECT_THROW((void)compiler.compile(graph), std::runtime_error);
  }
}

TEST(SchedulerTest, ExecutesNonCommutativeOperationsInCompiledOrder) {
  auto state = std::make_shared<ExecutionState>();
  auto registry = pipeline_registry(state);
  holoflow::runtime::Compiler compiler(
      registry, {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(pipeline_graph());

  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources,
                                         std::chrono::milliseconds{1});
  scheduler.start();
  scheduler.wait();

  EXPECT_EQ(state->order,
            (std::vector<std::string>{"source", "subtract", "scale", "sink"}));
  EXPECT_FLOAT_EQ(state->result, 14.F);
}

TEST(SchedulerTest, HandlesAnEmptyCompiledGraphLifecycle) {
  holoflow::core::Registry registry;
  holoflow::runtime::Compiler compiler(registry, {.dump_dot_on_failure = false});
  auto output = compiler.compile({});

  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  ASSERT_TRUE(scheduler.is_running());
  scheduler.request_stop();
  scheduler.wait();

  EXPECT_FALSE(scheduler.is_running());
  EXPECT_TRUE(scheduler.stop_requested());
  EXPECT_TRUE(scheduler.metrics().empty());
}

TEST(SchedulerTest, ExecutesAnIsolatedTerminalNodeExactlyOnce) {
  auto state = std::make_shared<ExecutionState>();
  holoflow::core::Registry registry;
  registry.register_sync("terminal", std::make_unique<TerminalFactory>(state));

  GraphSpec graph;
  add_vertex(NodeSpec{"terminal", "terminal", {}}, graph);
  holoflow::runtime::Compiler compiler(
      registry, {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(graph);

  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  EXPECT_EQ(state->terminal_calls, 1);
  EXPECT_TRUE(scheduler.stop_requested());
  EXPECT_FALSE(scheduler.is_running());
}

TEST(SchedulerTest, ExecutesDisconnectedTerminalNodesExactlyOnceEach) {
  auto state = std::make_shared<ExecutionState>();
  holoflow::core::Registry registry;
  registry.register_sync("terminal", std::make_unique<TerminalFactory>(state));

  GraphSpec graph;
  add_vertex(NodeSpec{"first", "terminal", {}}, graph);
  add_vertex(NodeSpec{"second", "terminal", {}}, graph);
  holoflow::runtime::Compiler compiler(
      registry, {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});
  auto output = compiler.compile(graph);

  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  EXPECT_EQ(state->terminal_calls, 2);
  EXPECT_TRUE(scheduler.stop_requested());
  EXPECT_FALSE(scheduler.is_running());
}
