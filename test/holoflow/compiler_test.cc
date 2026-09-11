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

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/graph_exec.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::GraphSpec;
using holoflow::core::InferResult;
using holoflow::core::MemLoc;
using holoflow::core::NodeSpec;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

class NoopTask final : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
    return holoflow::core::OpResult::Ok;
  }
};

class NoopAsyncTask final : public holoflow::core::IAsyncTask {
public:
  holoflow::core::OpResult try_push(holoflow::core::AsyncPushCtx &) override {
    return holoflow::core::OpResult::Ok;
  }
  holoflow::core::OpResult try_pop(holoflow::core::AsyncPopCtx &) override {
    return holoflow::core::OpResult::Ok;
  }
};

class AsyncFactory final : public holoflow::core::IAsyncTaskFactory {
public:
  explicit AsyncFactory(bool synchronizes_producer_stream)
      : synchronizes_producer_stream_(synchronizes_producer_stream) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("async task requires one input");
    }
    return {
        .input_descs                  = {inputs[0]},
        .output_descs                 = {inputs[0]},
        .in_place                     = {},
        .owned_inputs                 = {false},
        .owned_outputs                = {false},
        .kind                         = TaskKind::Async,
        .synchronizes_producer_stream = synchronizes_producer_stream_,
    };
  }

  std::unique_ptr<holoflow::core::IAsyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::AsyncCreateCtx &) const override {
    return std::make_unique<NoopAsyncTask>();
  }

private:
  bool synchronizes_producer_stream_;
};

class SourceFactory final : public holoflow::core::ISyncTaskFactory {
public:
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (!inputs.empty()) {
      throw std::invalid_argument("source has no inputs");
    }
    return {
        .input_descs   = {},
        .output_descs  = {TDesc({4}, DType::F32, MemLoc::Host)},
        .in_place      = {},
        .owned_inputs  = {},
        .owned_outputs = {false},
        .kind          = TaskKind::Sync,
    };
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<NoopTask>();
  }
};

class UnaryFactory final : public holoflow::core::ISyncTaskFactory {
public:
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("unary requires one input");
    }
    return {
        .input_descs   = {inputs[0]},
        .output_descs  = {inputs[0]},
        .in_place      = {},
        .owned_inputs  = {false},
        .owned_outputs = {false},
        .kind          = TaskKind::Sync,
    };
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<NoopTask>();
  }
};

class SinkFactory final : public holoflow::core::ISyncTaskFactory {
public:
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("sink requires one input");
    }
    return {
        .input_descs   = {inputs[0]},
        .output_descs  = {},
        .in_place      = {},
        .owned_inputs  = {false},
        .owned_outputs = {},
        .kind          = TaskKind::Sync,
    };
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<NoopTask>();
  }
};

holoflow::core::Registry registry() {
  holoflow::core::Registry value;
  value.register_sync("source", std::make_unique<SourceFactory>());
  value.register_sync("unary", std::make_unique<UnaryFactory>());
  value.register_sync("sink", std::make_unique<SinkFactory>());
  return value;
}

GraphSpec linear_graph(int out_idx = 0) {
  GraphSpec  graph;
  const auto source = graph.add_vertex(
      NodeSpec{.name = "source", .kind = "source", .settings = nlohmann::json::object()});
  const auto unary = graph.add_vertex(
      NodeSpec{.name = "unary", .kind = "unary", .settings = nlohmann::json::object()});
  const auto sink = graph.add_vertex(
      NodeSpec{.name = "sink", .kind = "sink", .settings = nlohmann::json::object()});
  graph.add_edge(source, holoflow::core::EdgeSpec{.out_idx = out_idx, .in_idx = 0}, unary);
  graph.add_edge(unary, holoflow::core::EdgeSpec{.out_idx = 0, .in_idx = 0}, sink);
  return graph;
}

} // namespace

TEST(CompilerTest, CompilesLinearGraphIntoTasksSectionsAndStorage) {
  auto                        factories = registry();
  holoflow::runtime::Compiler compiler(
      factories,
      {.dump_dot_on_failure = false, .verbose_tracing = false, .enable_profiling = false});

  const auto output = compiler.compile(linear_graph());

  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->graph.num_vertices(), 3);
  EXPECT_EQ(output->graph.num_edges(), 2);
  EXPECT_EQ(output->resources.tasks.size(), 3);
  EXPECT_FALSE(output->sections.empty());
  EXPECT_EQ(output->resources.tensor_descs.size(), 2);
}

TEST(CompilerTest, OrdersSynchronizingAsyncProducerBeforeOrdinaryProducer) {
  auto factories = registry();
  factories.register_async("ordinary", std::make_unique<AsyncFactory>(false));
  factories.register_async("synchronizing", std::make_unique<AsyncFactory>(true));

  GraphSpec  graph;
  const auto source             = graph.add_vertex(NodeSpec{"source", "source", {}});
  const auto ordinary           = graph.add_vertex(NodeSpec{"ordinary", "ordinary", {}});
  const auto synchronizing      = graph.add_vertex(NodeSpec{"synchronizing", "synchronizing", {}});
  const auto ordinary_sink      = graph.add_vertex(NodeSpec{"ordinary-sink", "sink", {}});
  const auto synchronizing_sink = graph.add_vertex(NodeSpec{"synchronizing-sink", "sink", {}});
  graph.add_edge(source, holoflow::core::EdgeSpec{0, 0}, ordinary);
  graph.add_edge(source, holoflow::core::EdgeSpec{0, 0}, synchronizing);
  graph.add_edge(ordinary, holoflow::core::EdgeSpec{0, 0}, ordinary_sink);
  graph.add_edge(synchronizing, holoflow::core::EdgeSpec{0, 0}, synchronizing_sink);

  holoflow::runtime::Compiler       compiler(factories, {.dump_dot_on_failure = false});
  const auto                        output         = compiler.compile(graph);
  const holoflow::runtime::Section *source_section = nullptr;
  for (const auto &section : output->sections) {
    if (std::find(section.sync_topo.begin(), section.sync_topo.end(), source) !=
        section.sync_topo.end()) {
      source_section = &section;
      break;
    }
  }

  ASSERT_NE(source_section, nullptr);
  ASSERT_EQ(source_section->async_prod.size(), 2);
  EXPECT_TRUE(source_section->has_synchronizing_async_producer);
  EXPECT_EQ(output->graph[source_section->async_prod[0]].spec.name, "synchronizing");
  EXPECT_EQ(output->graph[source_section->async_prod[1]].spec.name, "ordinary");
}

TEST(CompilerTest, RejectsUnknownFactory) {
  auto                        factories = registry();
  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  GraphSpec                   graph;
  graph.add_vertex(
      NodeSpec{.name = "bad", .kind = "missing", .settings = nlohmann::json::object()});

  EXPECT_THROW((void)compiler.compile(graph), std::exception);
}

TEST(CompilerTest, RejectsCycles) {
  auto                        factories = registry();
  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  auto                        graph = linear_graph();
  graph.add_edge(2, holoflow::core::EdgeSpec{.out_idx = 0, .in_idx = 0}, 0);

  EXPECT_THROW((void)compiler.compile(graph), std::exception);
}

TEST(CompilerTest, RejectsOutOfRangeTensorPorts) {
  auto                        factories = registry();
  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  auto                        graph  = linear_graph(5);

  EXPECT_THROW((void)compiler.compile(graph), std::exception);
}

TEST(SchedulerTest, StartsStopsAndPublishesNodeMetrics) {
  auto                         factories = registry();
  holoflow::runtime::Compiler  compiler(factories, {.dump_dot_on_failure = false});
  auto                         output = compiler.compile(linear_graph());
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources,
                                         std::chrono::milliseconds{5});

  scheduler.start();
  EXPECT_TRUE(scheduler.is_running());
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  scheduler.request_stop();
  scheduler.wait();

  EXPECT_FALSE(scheduler.is_running());
  EXPECT_TRUE(scheduler.stop_requested());
  const auto metrics = scheduler.metrics();
  EXPECT_TRUE(metrics.contains("source"));
  EXPECT_TRUE(metrics.contains("unary"));
  EXPECT_TRUE(metrics.contains("sink"));
}
