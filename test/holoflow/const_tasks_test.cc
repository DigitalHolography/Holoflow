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
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
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

class CountingTask final : public holoflow::core::ISyncTask {
public:
  explicit CountingTask(std::shared_ptr<Counter> counter) : counter_(std::move(counter)) {}

  OpResult execute(holoflow::core::SyncCtx &ctx) override {
    ++counter_->calls;
    if (!ctx.outputs.empty()) {
      auto *output = reinterpret_cast<float *>(ctx.outputs[0].data());
      output[0]     = 42.0F;
    }
    return OpResult::Ok;
  }

private:
  std::shared_ptr<Counter> counter_;
};

class TestFactory final : public holoflow::core::ISyncTaskFactory {
public:
  TestFactory(std::shared_ptr<Counter> counter, ConstInferenceState constness, size_t input_count,
              bool has_output)
      : counter_(std::move(counter)), constness_(constness), input_count_(input_count),
        has_output_(has_output) {}

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
        .owned_outputs = has_output_ ? std::vector<bool>{false} : std::vector<bool>{},
        .kind          = TaskKind::Sync,
        .constness     = constness_,
    };
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &, const holoflow::core::SyncCreateCtx &)
      const override {
    return std::make_unique<CountingTask>(counter_);
  }

private:
  std::shared_ptr<Counter>       counter_;
  ConstInferenceState constness_;
  size_t              input_count_;
  bool                has_output_;
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
}

} // namespace
