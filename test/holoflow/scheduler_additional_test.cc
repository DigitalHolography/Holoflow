// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <span>

#include "holoflow/runtime/compiler.hh"

namespace {

struct ResultState {
  holoflow::core::OpResult result = holoflow::core::OpResult::Eof;
  std::atomic<int>         calls{0};
};

class ResultTask final : public holoflow::core::ISyncTask {
public:
  explicit ResultTask(std::shared_ptr<ResultState> state) : state_(std::move(state)) {}
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
    ++state_->calls;
    return state_->result;
  }

private:
  std::shared_ptr<ResultState> state_;
};

class ResultFactory final : public holoflow::core::ISyncTaskFactory {
public:
  explicit ResultFactory(std::shared_ptr<ResultState> state) : state_(std::move(state)) {}
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,
                                    const nlohmann::json &) const override {
    return {{}, {}, {}, {}, {}, holoflow::core::TaskKind::Sync};
  }
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<ResultTask>(state_);
  }

private:
  std::shared_ptr<ResultState> state_;
};

std::pair<std::unique_ptr<holoflow::runtime::CompilerOutput>,
          std::unique_ptr<holoflow::core::Registry>>
compile_result_graph(std::shared_ptr<ResultState> state) {
  auto registry = std::make_unique<holoflow::core::Registry>();
  registry->register_sync("result", std::make_unique<ResultFactory>(state));
  holoflow::core::GraphSpec graph;
  add_vertex(holoflow::core::NodeSpec{"node", "result", {}}, graph);
  holoflow::runtime::Compiler compiler(*registry,
                                       {.dump_dot_on_failure = false, .enable_profiling = false});
  return {compiler.compile(graph), std::move(registry)};
}

} // namespace

class SchedulerResultTest : public testing::TestWithParam<holoflow::core::OpResult> {};

TEST_P(SchedulerResultTest, StopsForTerminalOrInvalidSyncResults) {
  auto state              = std::make_shared<ResultState>();
  state->result           = GetParam();
  auto [output, registry] = compile_result_graph(state);
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources,
                                         std::chrono::milliseconds{0});
  scheduler.start();
  scheduler.wait();
  EXPECT_TRUE(scheduler.stop_requested());
  EXPECT_FALSE(scheduler.is_running());
  EXPECT_GE(state->calls, 1);
}

INSTANTIATE_TEST_SUITE_P(TerminalResults, SchedulerResultTest,
                         testing::Values(holoflow::core::OpResult::Cancelled,
                                         holoflow::core::OpResult::Eof,
                                         holoflow::core::OpResult::NotReady));

TEST(SchedulerTest, RepeatedLifecycleCallsAreSafe) {
  auto state              = std::make_shared<ResultState>();
  auto [output, registry] = compile_result_graph(state);
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.request_stop();
  scheduler.start();
  scheduler.start();
  scheduler.request_stop();
  scheduler.request_stop();
  scheduler.wait();
  EXPECT_FALSE(scheduler.is_running());
}
