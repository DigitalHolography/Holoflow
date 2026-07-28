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

#include <boost/graph/adjacency_list.hpp>
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

GraphSpec linear_graph() {
  GraphSpec  graph;
  const auto source = add_vertex(
      NodeSpec{.name = "source", .kind = "source", .settings = nlohmann::json::object()}, graph);
  const auto unary = add_vertex(
      NodeSpec{.name = "unary", .kind = "unary", .settings = nlohmann::json::object()}, graph);
  const auto sink = add_vertex(
      NodeSpec{.name = "sink", .kind = "sink", .settings = nlohmann::json::object()}, graph);
  add_edge(source, unary, holoflow::core::EdgeSpec{.out_idx = 0, .in_idx = 0}, graph);
  add_edge(unary, sink, holoflow::core::EdgeSpec{.out_idx = 0, .in_idx = 0}, graph);
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
  EXPECT_EQ(num_vertices(output->graph), 3);
  EXPECT_EQ(num_edges(output->graph), 2);
  EXPECT_EQ(output->resources.tasks.size(), 3);
  EXPECT_FALSE(output->sections.empty());
  EXPECT_EQ(output->resources.tensor_descs.size(), 2);
}

TEST(CompilerTest, RejectsUnknownFactory) {
  auto                        factories = registry();
  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  GraphSpec                   graph;
  add_vertex(NodeSpec{.name = "bad", .kind = "missing", .settings = nlohmann::json::object()},
             graph);

  EXPECT_THROW((void)compiler.compile(graph), std::exception);
}

TEST(CompilerTest, RejectsCycles) {
  auto                        factories = registry();
  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  auto                        graph = linear_graph();
  add_edge(2, 0, holoflow::core::EdgeSpec{.out_idx = 0, .in_idx = 0}, graph);

  EXPECT_THROW((void)compiler.compile(graph), std::exception);
}

TEST(CompilerTest, RejectsOutOfRangeTensorPorts) {
  auto                        factories = registry();
  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  auto                        graph  = linear_graph();
  graph[*edges(graph).first].out_idx = 5;

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
