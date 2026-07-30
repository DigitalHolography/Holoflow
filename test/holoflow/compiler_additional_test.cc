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
#include <filesystem>
#include <memory>
#include <span>

#include "holoflow/runtime/compiler.hh"
#include "support/math_tasks.hh"

namespace {

using holoflow::core::EdgeSpec;
using holoflow::core::GraphSpec;
using holoflow::core::InferResult;
using holoflow::core::NodeSpec;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

GraphSpec source_sink_graph() {
  GraphSpec graph;
  auto      source = add_vertex(NodeSpec{"source", "source", {}}, graph);
  auto      sink   = add_vertex(NodeSpec{"sink", "sink", {}}, graph);
  add_edge(source, sink, EdgeSpec{0, 0}, graph);
  return graph;
}

struct TrackingState {
  int create_calls = 0;
  int update_calls = 0;
};

class NoopTask final : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &) override {
    return holoflow::core::OpResult::Eof;
  }
};

class TrackingSourceFactory final : public holoflow::core::ISyncTaskFactory {
public:
  explicit TrackingSourceFactory(std::shared_ptr<TrackingState> state) : state_(std::move(state)) {}

  InferResult infer(std::span<const TDesc>, const nlohmann::json &) const override {
    return {{},      {TDesc({8}, holoflow::core::DType::F32, holoflow::core::MemLoc::Host)},
            {},      {},
            {false}, TaskKind::Sync};
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    ++state_->create_calls;
    return std::make_unique<NoopTask>();
  }

  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task, std::span<const TDesc>,
         const nlohmann::json &, const holoflow::core::SyncCreateCtx &) const override {
    ++state_->update_calls;
    return old_task;
  }

private:
  std::shared_ptr<TrackingState> state_;
};

class SinkFactory final : public holoflow::core::ISyncTaskFactory {
public:
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    return {{inputs.begin(), inputs.end()}, {}, {}, {false}, {}, TaskKind::Sync};
  }
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<NoopTask>();
  }
};

class OwnedSourceFactory final : public holoflow::core::ISyncTaskFactory {
public:
  InferResult infer(std::span<const TDesc>, const nlohmann::json &) const override {
    return {{},     {TDesc({4}, holoflow::core::DType::F32, holoflow::core::MemLoc::Host)},
            {},     {},
            {true}, TaskKind::Sync};
  }
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<NoopTask>();
  }
};

class OwnedSinkFactory final : public holoflow::core::ISyncTaskFactory {
public:
  InferResult infer(std::span<const TDesc> inputs, const nlohmann::json &) const override {
    return {{inputs.begin(), inputs.end()}, {}, {}, {true}, {}, TaskKind::Sync};
  }
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const TDesc>, const nlohmann::json &,
         const holoflow::core::SyncCreateCtx &) const override {
    return std::make_unique<NoopTask>();
  }
};

} // namespace

TEST(CompilerTest, CompilesAnEmptyGraph) {
  holoflow::core::Registry    registry;
  holoflow::runtime::Compiler compiler(registry,
                                       {.dump_dot_on_failure = false, .enable_profiling = false});
  const auto                  output = compiler.compile({});
  EXPECT_EQ(num_vertices(output->graph), 0);
  EXPECT_TRUE(output->sections.empty());
  EXPECT_TRUE(output->resources.tasks.empty());
}

TEST(CompilerTest, RejectsDuplicateNamesAndInputDestinations) {
  auto                     state = std::make_shared<TrackingState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<TrackingSourceFactory>(state));
  registry.register_sync("sink", std::make_unique<SinkFactory>());
  holoflow::runtime::Compiler compiler(registry, {.dump_dot_on_failure = false});

  GraphSpec duplicate_names;
  add_vertex(NodeSpec{"same", "source", {}}, duplicate_names);
  add_vertex(NodeSpec{"same", "source", {}}, duplicate_names);
  EXPECT_THROW((void)compiler.compile(duplicate_names), std::runtime_error);

  GraphSpec duplicate_input;
  auto      a = add_vertex(NodeSpec{"a", "source", {}}, duplicate_input);
  auto      b = add_vertex(NodeSpec{"b", "source", {}}, duplicate_input);
  auto      s = add_vertex(NodeSpec{"sink", "sink", {}}, duplicate_input);
  add_edge(a, s, EdgeSpec{0, 0}, duplicate_input);
  add_edge(b, s, EdgeSpec{0, 0}, duplicate_input);
  EXPECT_THROW((void)compiler.compile(duplicate_input), std::runtime_error);
}

TEST(CompilerTest, ReusesTaskStreamAndExactSizeHostAllocation) {
  auto                     tracking = std::make_shared<TrackingState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<TrackingSourceFactory>(tracking));
  registry.register_sync("sink", std::make_unique<SinkFactory>());
  holoflow::runtime::Compiler compiler(registry,
                                       {.dump_dot_on_failure = false, .enable_profiling = false});

  auto first = compiler.compile(source_sink_graph());
  ASSERT_EQ(first->resources.memory_blocks.size(), 1);
  const auto first_ptr    = first->resources.memory_blocks.begin()->second.get();
  const auto first_stream = first->resources.streams.begin()->second.get();

  auto second = compiler.compile(source_sink_graph(), std::move(first));
  EXPECT_EQ(tracking->create_calls, 1);
  EXPECT_EQ(tracking->update_calls, 1);
  EXPECT_EQ(second->resources.memory_blocks.begin()->second.get(), first_ptr);
  EXPECT_EQ(second->resources.streams.begin()->second.get(), first_stream);
}

TEST(CompilerTest, EmitsLogsTraceAndSuccessGraph) {
  auto                     tracking = std::make_shared<TrackingState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<TrackingSourceFactory>(tracking));
  registry.register_sync("sink", std::make_unique<SinkFactory>());
  const auto unique_suffix = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto directory = std::filesystem::temp_directory_path() /
                         ("holoflow-compiler-observability-test-" + std::to_string(unique_suffix));

  {
    holoflow::runtime::Compiler compiler(registry, {.log_dir             = directory,
                                                    .dump_dot_on_failure = true,
                                                    .verbose_tracing     = false,
                                                    .enable_profiling    = true,
                                                    .trace_filename      = "trace.json"});
    ASSERT_NE(compiler.compile(source_sink_graph()), nullptr);
  }

  EXPECT_TRUE(std::filesystem::exists(directory / "compiler.log"));
  EXPECT_TRUE(std::filesystem::exists(directory / "compilation_success.dot"));
  EXPECT_TRUE(std::filesystem::exists(directory / "trace.json"));
}

TEST(CompilerTest, RejectsMultipleOwnersOfOneTensor) {
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<OwnedSourceFactory>());
  registry.register_sync("sink", std::make_unique<OwnedSinkFactory>());
  holoflow::runtime::Compiler compiler(registry, {.dump_dot_on_failure = false});
  EXPECT_THROW((void)compiler.compile(source_sink_graph()), std::runtime_error);
}
