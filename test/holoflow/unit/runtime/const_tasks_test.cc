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

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
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

using holoflow::core::ConstInferenceState;
using holoflow::core::DType;
using holoflow::core::GraphSpec;
using holoflow::core::InferResult;
using holoflow::core::MemLoc;
using holoflow::core::NodeSpec;
using holoflow::core::OpResult;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

struct Counter {
  std::atomic<int> calls{0};
};

struct ConstTaskRaceGate {
  std::atomic<bool> first_chain_task_started{false};
  std::atomic<bool> wait_timed_out{false};
};

class CountingTask final : public holoflow::core::ISyncTask {
public:
  explicit CountingTask(std::shared_ptr<Counter> counter,
                        std::function<OpResult(holoflow::core::SyncCtx &)> on_execute = {})
      : counter_(std::move(counter)), on_execute_(std::move(on_execute)) {}

  OpResult execute(holoflow::core::SyncCtx &ctx) override {
    ++counter_->calls;
    if (!ctx.outputs.empty()) {
      auto *output = reinterpret_cast<float *>(ctx.outputs[0].data());
      output[0]     = 42.0F;
    }
    if (on_execute_) {
      return on_execute_(ctx);
    }
    return OpResult::Ok;
  }

private:
  std::shared_ptr<Counter> counter_;
  std::function<OpResult(holoflow::core::SyncCtx &)> on_execute_;
};

class TestFactory final : public holoflow::core::ISyncTaskFactory {
public:
  TestFactory(std::shared_ptr<Counter> counter, ConstInferenceState constness, size_t input_count,
              bool has_output,
              std::function<OpResult(holoflow::core::SyncCtx &)> on_execute = {},
              bool owned_output = false)
      : counter_(std::move(counter)), constness_(constness), input_count_(input_count),
        has_output_(has_output), on_execute_(std::move(on_execute)),
        owned_output_(owned_output) {}

  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    if (inputs.size() != input_count_) {
      throw std::invalid_argument("unexpected test task input count");
    }

    return InferResult{
        .input_descs   = std::vector<TDesc>(inputs.begin(), inputs.end()),
        .output_descs  = has_output_ ? std::vector<TDesc>{TDesc({1}, DType::F32, MemLoc::Host)}
                                     : std::vector<TDesc>{},
        .in_place      = {},
        .owned_inputs  = std::vector<bool>(inputs.size(), false),
        .owned_outputs = has_output_ ? std::vector<bool>{owned_output_} : std::vector<bool>{},
        .kind          = TaskKind::Sync,
        .constness     = constness_,
    };
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &, const holoflow::core::SyncCreateCtx &)
      const override {
    return std::make_unique<CountingTask>(counter_, on_execute_);
  }

private:
  std::shared_ptr<Counter>       counter_;
  ConstInferenceState constness_;
  size_t              input_count_;
  bool                has_output_;
  std::function<OpResult(holoflow::core::SyncCtx &)> on_execute_;
  bool                owned_output_;
};

class NoOpAsyncTask final : public holoflow::core::IAsyncTask {
public:
  OpResult try_push(holoflow::core::AsyncPushCtx &) override { return OpResult::Eof; }
  OpResult try_pop(holoflow::core::AsyncPopCtx &) override { return OpResult::Eof; }
};

class AsyncTestFactory final : public holoflow::core::IAsyncTaskFactory {
public:
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    return InferResult{
        .input_descs   = std::vector<TDesc>(inputs.begin(), inputs.end()),
        .output_descs  = {TDesc({1}, DType::F32, MemLoc::Host)},
        .in_place      = {},
        .owned_inputs  = std::vector<bool>(inputs.size(), false),
        .owned_outputs = {false},
        .kind          = TaskKind::Async,
        .constness     = ConstInferenceState::Constant,
    };
  }

  std::unique_ptr<holoflow::core::IAsyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::AsyncCreateCtx &) const override {
    return std::make_unique<NoOpAsyncTask>();
  }
};

TEST(ConstTasksTest, CompilerPropagatesConstnessButKeepsSinkMutable) {
  auto counter = std::make_shared<Counter>();
  holoflow::core::Registry factories;
  factories.register_sync("constant", std::make_unique<TestFactory>(
                                           counter, ConstInferenceState::Constant, 0, true));
  factories.register_sync("transform", std::make_unique<TestFactory>(
                                           counter, ConstInferenceState::Undefined, 1, true));
  factories.register_sync("sink", std::make_unique<TestFactory>(
                                      counter, ConstInferenceState::Undefined, 1, false));

  GraphSpec graph;
  const auto source = add_vertex(NodeSpec{"source", "constant", {}}, graph);
  const auto transform = add_vertex(NodeSpec{"transform", "transform", {}}, graph);
  const auto sink = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(source, transform, {0, 0}, graph);
  add_edge(transform, sink, {0, 0}, graph);

  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  const auto output = compiler.compile(graph);

  EXPECT_EQ(output->graph[source].infer.constness, ConstInferenceState::Constant);
  EXPECT_EQ(output->graph[transform].infer.constness, ConstInferenceState::Constant);
  EXPECT_EQ(output->graph[sink].infer.constness, ConstInferenceState::Mutable);
}

TEST(ConstTasksTest, UnspecifiedSourceDefaultsMutableAndPreventsPropagation) {
  auto counter = std::make_shared<Counter>();
  holoflow::core::Registry factories;
  factories.register_sync("constant", std::make_unique<TestFactory>(
                                           counter, ConstInferenceState::Constant, 0, true));
  factories.register_sync("unspecified_source", std::make_unique<TestFactory>(
                                                     counter, ConstInferenceState::Undefined, 0,
                                                     true));
  factories.register_sync("combine", std::make_unique<TestFactory>(
                                          counter, ConstInferenceState::Undefined, 2, true));

  GraphSpec graph;
  const auto constant = add_vertex(NodeSpec{"constant", "constant", {}}, graph);
  const auto mutable_source =
      add_vertex(NodeSpec{"mutable_source", "unspecified_source", {}}, graph);
  const auto combine = add_vertex(NodeSpec{"combine", "combine", {}}, graph);
  add_edge(constant, combine, {0, 0}, graph);
  add_edge(mutable_source, combine, {0, 1}, graph);

  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  const auto output = compiler.compile(graph);

  EXPECT_EQ(output->graph[constant].infer.constness, ConstInferenceState::Constant);
  EXPECT_EQ(output->graph[mutable_source].infer.constness, ConstInferenceState::Mutable);
  EXPECT_EQ(output->graph[combine].infer.constness, ConstInferenceState::Mutable);
}

TEST(ConstTasksTest, RejectsConstantTaskWithMutableInput) {
  auto counter = std::make_shared<Counter>();
  holoflow::core::Registry factories;
  factories.register_sync("mutable", std::make_unique<TestFactory>(
                                          counter, ConstInferenceState::Mutable, 0, true));
  factories.register_sync("constant", std::make_unique<TestFactory>(
                                           counter, ConstInferenceState::Constant, 1, true));

  GraphSpec graph;
  const auto source = add_vertex(NodeSpec{"mutable", "mutable", {}}, graph);
  const auto constant = add_vertex(NodeSpec{"constant", "constant", {}}, graph);
  add_edge(source, constant, {0, 0}, graph);

  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  EXPECT_THROW((void)compiler.compile(graph), std::exception);
}

TEST(ConstTasksTest, AsyncTaskIsAlwaysMutable) {
  holoflow::core::Registry factories;
  factories.register_async("async", std::make_unique<AsyncTestFactory>());

  GraphSpec graph;
  const auto async = add_vertex(NodeSpec{"async", "async", {}}, graph);

  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  const auto output = compiler.compile(graph);

  EXPECT_EQ(output->graph[async].infer.constness, ConstInferenceState::Mutable);
}

TEST(ConstTasksTest, OwnedOutputIsNotClassifiedAsCachedConstant) {
  auto counter = std::make_shared<Counter>();
  holoflow::core::Registry factories;
  factories.register_sync("owned_constant", std::make_unique<TestFactory>(
                                                  counter, ConstInferenceState::Constant, 0, true,
                                                  std::function<OpResult(holoflow::core::SyncCtx &)>{},
                                                  true));
  factories.register_sync("transform", std::make_unique<TestFactory>(
                                           counter, ConstInferenceState::Undefined, 1, true));

  GraphSpec graph;
  const auto source = add_vertex(NodeSpec{"source", "owned_constant", {}}, graph);
  const auto transform = add_vertex(NodeSpec{"transform", "transform", {}}, graph);
  add_edge(source, transform, {0, 0}, graph);

  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  const auto output = compiler.compile(graph);

  EXPECT_EQ(output->graph[source].infer.constness, ConstInferenceState::Mutable);
  EXPECT_EQ(output->graph[transform].infer.constness, ConstInferenceState::Mutable);
}

TEST(ConstTasksTest, SchedulerExecutesConstantOnlyGraphOnce) {
  auto counter = std::make_shared<Counter>();
  holoflow::core::Registry factories;
  factories.register_sync("constant", std::make_unique<TestFactory>(
                                           counter, ConstInferenceState::Constant, 0, true));

  GraphSpec graph;
  add_vertex(NodeSpec{"constant", "constant", {}}, graph);

  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  const auto output = compiler.compile(graph);

  ASSERT_EQ(output->sections.size(), 1);
  EXPECT_EQ(output->sections[0].const_sync_topo.size(), 1);
  EXPECT_TRUE(output->sections[0].sync_topo.empty());

  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  EXPECT_EQ(counter->calls.load(), 1);

  // A completed constant result remains cached across restarts of this scheduler.
  scheduler.start();
  scheduler.wait();
  EXPECT_EQ(counter->calls.load(), 1);
}

TEST(ConstTasksTest, FinishesConstantInitializationAcrossSections) {
  constexpr auto wait_timeout = std::chrono::seconds{3};

  auto gate = std::make_shared<ConstTaskRaceGate>();
  auto independent_calls = std::make_shared<Counter>();
  auto chain_source_calls = std::make_shared<Counter>();
  auto chain_first_calls = std::make_shared<Counter>();
  auto chain_last_calls = std::make_shared<Counter>();

  auto wait_for = [gate, wait_timeout](const std::function<bool()> &condition) {
    const auto deadline = std::chrono::steady_clock::now() + wait_timeout;
    while (!condition() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    if (!condition()) {
      gate->wait_timed_out.store(true);
    }
  };

  holoflow::core::Registry factories;
  factories.register_sync(
      "independent_constant",
      std::make_unique<TestFactory>(
          independent_calls, ConstInferenceState::Constant, 0, true,
          [wait_for, gate](holoflow::core::SyncCtx &) {
            // Let the chained section reach its first task before this section finishes.
            wait_for([&] { return gate->first_chain_task_started.load(); });
            return OpResult::Ok;
          }));
  factories.register_sync(
      "chain_source",
      std::make_unique<TestFactory>(chain_source_calls, ConstInferenceState::Constant, 0, true));
  factories.register_sync(
      "chain_first",
      std::make_unique<TestFactory>(
          chain_first_calls, ConstInferenceState::Undefined, 1, true,
          [gate](holoflow::core::SyncCtx &) {
            gate->first_chain_task_started.store(true);
            // Keep this section in initialization while the independent section finishes.
            std::this_thread::sleep_for(std::chrono::milliseconds{50});
            return OpResult::Ok;
          }));
  factories.register_sync(
      "chain_last",
      std::make_unique<TestFactory>(chain_last_calls, ConstInferenceState::Undefined, 1, true));

  GraphSpec graph;
  add_vertex(NodeSpec{"independent", "independent_constant", {}}, graph);
  const auto source = add_vertex(NodeSpec{"chain-source", "chain_source", {}}, graph);
  const auto first = add_vertex(NodeSpec{"chain-first", "chain_first", {}}, graph);
  const auto last = add_vertex(NodeSpec{"chain-last", "chain_last", {}}, graph);
  add_edge(source, first, {0, 0}, graph);
  add_edge(first, last, {0, 0}, graph);

  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  const auto output = compiler.compile(graph);

  ASSERT_EQ(output->sections.size(), 2);
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  EXPECT_FALSE(gate->wait_timed_out.load());
  EXPECT_EQ(independent_calls->calls.load(), 1);
  EXPECT_EQ(chain_source_calls->calls.load(), 1);
  EXPECT_EQ(chain_first_calls->calls.load(), 1);
  // The early stop must not prevent the rest of the other section's constants from initializing.
  EXPECT_EQ(chain_last_calls->calls.load(), 1);
}

TEST(ConstTasksTest, InitializesIndependentConstantSections) {
  auto calls_a = std::make_shared<Counter>();
  auto calls_b = std::make_shared<Counter>();
  auto calls_c = std::make_shared<Counter>();
  holoflow::core::Registry factories;
  factories.register_sync("constant-a", std::make_unique<TestFactory>(
                                             calls_a, ConstInferenceState::Constant, 0, true));
  factories.register_sync("constant-b", std::make_unique<TestFactory>(
                                             calls_b, ConstInferenceState::Constant, 0, true));
  factories.register_sync("constant-c", std::make_unique<TestFactory>(
                                             calls_c, ConstInferenceState::Constant, 0, true));

  GraphSpec graph;
  add_vertex(NodeSpec{"a", "constant-a", {}}, graph);
  add_vertex(NodeSpec{"b", "constant-b", {}}, graph);
  add_vertex(NodeSpec{"c", "constant-c", {}}, graph);

  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  const auto output = compiler.compile(graph);
  ASSERT_EQ(output->sections.size(), 3);

  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  EXPECT_EQ(calls_a->calls.load(), 1);
  EXPECT_EQ(calls_b->calls.load(), 1);
  EXPECT_EQ(calls_c->calls.load(), 1);
}

TEST(ConstTasksTest, ReusesConstantResultWhileMutableWorkRepeats) {
  constexpr int iterations = 3;
  auto constant_calls = std::make_shared<Counter>();
  auto mutable_source_calls = std::make_shared<Counter>();
  auto sink_calls = std::make_shared<Counter>();

  holoflow::core::Registry factories;
  factories.register_sync("constant", std::make_unique<TestFactory>(
                                           constant_calls, ConstInferenceState::Constant, 0, true));
  factories.register_sync("mutable", std::make_unique<TestFactory>(
                                          mutable_source_calls, ConstInferenceState::Mutable, 0, true));
  factories.register_sync(
      "sink", std::make_unique<TestFactory>(
                  sink_calls, ConstInferenceState::Undefined, 1, false,
                  [sink_calls, iterations](holoflow::core::SyncCtx &) {
                    return sink_calls->calls.load() >= iterations ? OpResult::Eof : OpResult::Ok;
                  }));

  GraphSpec graph;
  add_vertex(NodeSpec{"constant", "constant", {}}, graph);
  const auto mutable_source = add_vertex(NodeSpec{"mutable", "mutable", {}}, graph);
  const auto sink = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(mutable_source, sink, {0, 0}, graph);

  holoflow::runtime::Compiler compiler(factories, {.dump_dot_on_failure = false});
  const auto output = compiler.compile(graph);
  ASSERT_EQ(output->sections.size(), 2);
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  EXPECT_EQ(constant_calls->calls.load(), 1);
  EXPECT_EQ(mutable_source_calls->calls.load(), iterations);
  EXPECT_EQ(sink_calls->calls.load(), iterations);
}

} // namespace
