// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <array>
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

struct OwnershipState {
  std::array<float, 2> input{};
  std::array<float, 2> output{};
  std::atomic<int>     acquire_calls{0};
  std::atomic<int>     release_calls{0};
  std::atomic<bool>    observed_shared_input{false};
  std::atomic<bool>    observed_shared_output{false};
  std::atomic<bool>    rejected_bad_storage_indices{false};
};

class OwnershipSourceTask final : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    auto *values = reinterpret_cast<float *>(ctx.outputs[0].data());
    values[0]    = 4.F;
    values[1]    = 8.F;
    return holoflow::core::OpResult::Ok;
  }
};

class OwnershipSourceFactory final : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,
                                    const nlohmann::json &) const override {
    using namespace holoflow::core;
    return {{}, {TDesc({2}, DType::F32, MemLoc::Host)}, {}, {}, {false}, TaskKind::Sync};
  }
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<OwnershipSourceTask>();
  }
};

class OwnershipRelayTask final : public holoflow::core::ISyncTask {
public:
  explicit OwnershipRelayTask(std::shared_ptr<OwnershipState> state) : state_(std::move(state)) {}

  std::optional<holoflow::core::TView> acquire_input(int index) override {
    if (index != 0)
      throw std::out_of_range("owned input");
    if (state_->acquire_calls.fetch_add(1) == 0)
      return std::nullopt;

    auto &storage = storage_access().owned_input_storage(0);
    storage.ptr   = reinterpret_cast<std::byte *>(state_->input.data());
    return holoflow::core::TView{
        holoflow::core::TDesc({2}, holoflow::core::DType::F32, holoflow::core::MemLoc::Host),
        &storage};
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    bool rejected_input  = false;
    bool rejected_output = false;
    try {
      (void)storage_access().owned_input_storage(1);
    } catch (const std::out_of_range &) {
      rejected_input = true;
    }
    try {
      (void)storage_access().owned_output_storage(1);
    } catch (const std::out_of_range &) {
      rejected_output = true;
    }
    state_->rejected_bad_storage_indices.store(rejected_input && rejected_output);

    state_->observed_shared_input.store(ctx.inputs[0].data() ==
                                        reinterpret_cast<std::byte *>(state_->input.data()));
    const auto *input = reinterpret_cast<const float *>(ctx.inputs[0].data());

    auto &storage = storage_access().owned_output_storage(0);
    storage.ptr   = reinterpret_cast<std::byte *>(state_->output.data());
    state_->observed_shared_output.store(ctx.outputs[0].data() ==
                                         reinterpret_cast<std::byte *>(state_->output.data()));
    auto *output = reinterpret_cast<float *>(ctx.outputs[0].data());
    output[0]    = input[0] * 2.F;
    output[1]    = input[1] * 2.F;
    return holoflow::core::OpResult::Ok;
  }

  void release_output(int index) override {
    if (index != 0)
      throw std::out_of_range("owned output");
    ++state_->release_calls;
    storage_access().owned_output_storage(0).ptr = nullptr;
  }

private:
  std::shared_ptr<OwnershipState> state_;
};

class OwnershipRelayFactory final : public holoflow::core::ISyncTaskFactory {
public:
  explicit OwnershipRelayFactory(std::shared_ptr<OwnershipState> state)
      : state_(std::move(state)) {}

  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> inputs,
                                    const nlohmann::json &) const override {
    using namespace holoflow::core;
    return {{inputs[0]}, {inputs[0]}, {}, {true}, {true}, TaskKind::Sync};
  }
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<OwnershipRelayTask>(state_);
  }

private:
  std::shared_ptr<OwnershipState> state_;
};

class OwnershipSinkTask final : public holoflow::core::ISyncTask {
public:
  explicit OwnershipSinkTask(std::shared_ptr<OwnershipState> state) : state_(std::move(state)) {}
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto *input = reinterpret_cast<const float *>(ctx.inputs[0].data());
    if (input != state_->output.data() || input[0] != 8.F || input[1] != 16.F)
      throw std::runtime_error("owned output was not published through shared Storage");
    return holoflow::core::OpResult::Eof;
  }

private:
  std::shared_ptr<OwnershipState> state_;
};

class OwnershipSinkFactory final : public holoflow::core::ISyncTaskFactory {
public:
  explicit OwnershipSinkFactory(std::shared_ptr<OwnershipState> state) : state_(std::move(state)) {}
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> inputs,
                                    const nlohmann::json &) const override {
    return {{inputs[0]}, {}, {}, {false}, {}, holoflow::core::TaskKind::Sync};
  }
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<OwnershipSinkTask>(state_);
  }

private:
  std::shared_ptr<OwnershipState> state_;
};

class UnproducedOwnedTask final : public holoflow::core::ISyncTask {
public:
  explicit UnproducedOwnedTask(std::shared_ptr<OwnershipState> state) : state_(std::move(state)) {}
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
    return holoflow::core::OpResult::Eof;
  }
  void release_output(int) override { ++state_->release_calls; }

private:
  std::shared_ptr<OwnershipState> state_;
};

class UnproducedOwnedFactory final : public holoflow::core::ISyncTaskFactory {
public:
  explicit UnproducedOwnedFactory(std::shared_ptr<OwnershipState> state)
      : state_(std::move(state)) {}
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,
                                    const nlohmann::json &) const override {
    using namespace holoflow::core;
    return {{}, {TDesc({1}, DType::F32, MemLoc::Host)}, {}, {}, {true}, TaskKind::Sync};
  }
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<UnproducedOwnedTask>(state_);
  }

private:
  std::shared_ptr<OwnershipState> state_;
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

TEST(SchedulerTest, UsesSharedStorageAndReleasesSuccessfulOwnedOutputExactlyOnce) {
  auto                     state = std::make_shared<OwnershipState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<OwnershipSourceFactory>());
  registry.register_sync("relay", std::make_unique<OwnershipRelayFactory>(state));
  registry.register_sync("sink", std::make_unique<OwnershipSinkFactory>(state));

  holoflow::core::GraphSpec graph;
  auto source = add_vertex(holoflow::core::NodeSpec{"source", "source", {}}, graph);
  auto relay  = add_vertex(holoflow::core::NodeSpec{"relay", "relay", {}}, graph);
  auto sink   = add_vertex(holoflow::core::NodeSpec{"sink", "sink", {}}, graph);
  add_edge(source, relay, holoflow::core::EdgeSpec{0, 0}, graph);
  add_edge(relay, sink, holoflow::core::EdgeSpec{0, 0}, graph);

  holoflow::runtime::Compiler  compiler(registry,
                                        {.dump_dot_on_failure = false, .enable_profiling = false});
  auto                         output = compiler.compile(graph);
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  EXPECT_GE(state->acquire_calls.load(), 2);
  EXPECT_TRUE(state->observed_shared_input.load());
  EXPECT_TRUE(state->observed_shared_output.load());
  EXPECT_TRUE(state->rejected_bad_storage_indices.load());
  EXPECT_EQ(state->release_calls.load(), 1);
}

TEST(SchedulerTest, DoesNotReleaseOwnedOutputWhenOperationDoesNotProduceIt) {
  auto                     state = std::make_shared<OwnershipState>();
  holoflow::core::Registry registry;
  registry.register_sync("terminal-owned", std::make_unique<UnproducedOwnedFactory>(state));
  holoflow::core::GraphSpec graph;
  add_vertex(holoflow::core::NodeSpec{"terminal", "terminal-owned", {}}, graph);

  holoflow::runtime::Compiler  compiler(registry,
                                        {.dump_dot_on_failure = false, .enable_profiling = false});
  auto                         output = compiler.compile(graph);
  holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
  scheduler.start();
  scheduler.wait();

  EXPECT_EQ(state->release_calls.load(), 0);
}
