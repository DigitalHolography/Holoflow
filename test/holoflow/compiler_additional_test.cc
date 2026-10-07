// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <algorithm>

#include "holoflow/runtime/tracing.hh"
#include "support/native_trace.hh"
#include <boost/graph/adjacency_list.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <span>

#include "../../src/holoflow/src/runtime/diagnostics_file_writer.hh"
#include "holoflow/runtime/compiler.hh"
#include "support/math_tasks.hh"

namespace {

using holoflow::core::EdgeSpec;
using holoflow::core::GraphSpec;
using holoflow::core::InferResult;
using holoflow::core::NodeSpec;
using holoflow::core::TaskKind;
using holoflow::core::TDesc;

bool background_slice(const std::string &name) {
  return name == "Format CUDA Graph Diagnostics" || name == "Write CUDA Graph Diagnostics File" ||
         name == "Format Graph Spec JSON" || name == "Format Pipeline Graph DOT" ||
         name == "Write Diagnostic Text File";
}

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

TEST(CompilerTest, EmitsLogsNativeTraceAndSuccessGraph) {
  auto                     tracking = std::make_shared<TrackingState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<TrackingSourceFactory>(tracking));
  registry.register_sync("sink", std::make_unique<SinkFactory>());
  const auto directory = trace_test::temporary_path("holoflow-native-compiler-");
  {
    holoflow::runtime::Compiler compiler(registry, {.log_dir             = directory,
                                                    .dump_dot_on_failure = true,
                                                    .verbose_tracing     = false,
                                                    .trace_filename      = "trace.perfetto-trace"});
    auto                        output = compiler.compile(source_sink_graph());
    ASSERT_NE(output, nullptr);
    output = compiler.compile(source_sink_graph(), std::move(output));
    ASSERT_NE(output, nullptr);
    if (holoflow::runtime::tracing::Session::available()) {
      const auto trace    = trace_test::read(directory / "trace.perfetto-trace");
      auto       position = [&](const std::string &name) {
        size_t index = trace.slices.size();
        size_t count = 0;
        for (size_t i = 0; i < trace.slices.size(); ++i) {
          if (trace.slices[i].name == name) {
            index = i;
            ++count;
          }
          // Background file writes can outlive an automatic compilation capture.
          if (!background_slice(trace.slices[i].name))
            EXPECT_TRUE(trace.slices[i].complete);
        }
        EXPECT_EQ(count, 1) << name;
        return index;
      };
      const auto validation = position("Validate Spec");
      for (const auto *name :
           {"Initialize Compilation", "Dump Graph Spec", "Drain Previous CUDA Streams"})
        EXPECT_LT(position(name), validation);
      const auto carry = position("Carry Compatible Section CUDA Graphs");
      EXPECT_LT(position("Task Binding"), carry);
      EXPECT_LT(carry, position("Inspect Section CUDA Graphs"));
      const auto total = position("Total Compilation");
      EXPECT_LT(total, position("Initialize Compilation"));
      EXPECT_EQ(trace.slices[total].outcome, "success");
    }
  }
  holoflow::runtime::section_diagnostics_file_writer().flush();
  EXPECT_TRUE(std::filesystem::exists(directory / "compiler.log"));
  EXPECT_TRUE(std::filesystem::exists(directory / "compilation_success.dot"));
  EXPECT_EQ(std::filesystem::exists(directory / "trace.perfetto-trace"),
            holoflow::runtime::tracing::Session::available());
  std::ifstream     input(directory / "compiler.log");
  const std::string logs{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  EXPECT_EQ(logs.find("End Pass"), std::string::npos);
  EXPECT_EQ(logs.find("Compilation Passes Summary"), std::string::npos);
}

TEST(CompilerTest, RejectsMultipleOwnersOfOneTensor) {
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<OwnedSourceFactory>());
  registry.register_sync("sink", std::make_unique<OwnedSinkFactory>());
  holoflow::runtime::Compiler compiler(registry, {.dump_dot_on_failure = false});
  EXPECT_THROW((void)compiler.compile(source_sink_graph()), std::runtime_error);
}

TEST(CompilerTest, ProfilingDisabledDoesNotWriteTrace) {
  holoflow::core::Registry registry;
  const auto               directory =
      std::filesystem::temp_directory_path() /
      ("holoflow-no-profiling-test-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  {
    holoflow::runtime::Compiler compiler(
        registry, {.log_dir = directory, .dump_dot_on_failure = false, .enable_profiling = false});
    EXPECT_NE(compiler.compile({}), nullptr);
    EXPECT_FALSE(std::filesystem::exists(directory / "trace_events.perfetto-trace"));
  }
}

TEST(CompilerTest, FailureExportsCompletedPreparationAndValidationSpans) {
  holoflow::core::Registry registry;
  const auto               directory =
      std::filesystem::temp_directory_path() /
      ("holoflow-failed-trace-test-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  holoflow::runtime::Compiler compiler(registry,
                                       {.log_dir = directory, .dump_dot_on_failure = false});
  GraphSpec                   invalid;
  add_vertex(NodeSpec{"missing", "missing", {}}, invalid);
  EXPECT_THROW((void)compiler.compile(invalid), std::runtime_error);
  if (holoflow::runtime::tracing::Session::available()) {
    const auto trace      = trace_test::read(directory / "trace_events.perfetto-trace");
    bool       validation = false;
    bool       total      = false;
    for (const auto &slice : trace.slices) {
      validation |= slice.name == "Validate Spec";
      total |= slice.name == "Total Compilation" && slice.outcome == "failure";
      if (!background_slice(slice.name))
        EXPECT_TRUE(slice.complete);
    }
    EXPECT_TRUE(validation);
    EXPECT_TRUE(total);
  }
}

TEST(CompilerTest, TraceExportFailureDoesNotFailCompilation) {
  holoflow::core::Registry registry;
  const auto               directory =
      std::filesystem::temp_directory_path() /
      ("holoflow-trace-export-failure-test-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  holoflow::runtime::Compiler compiler(
      registry, {.log_dir = directory, .dump_dot_on_failure = false, .trace_filename = "."});
  EXPECT_NE(compiler.compile({}), nullptr);
}

TEST(CompilerTest, ExplicitSessionCapturesCompilationSchedulerStartupStopAndResume) {
  if (!holoflow::runtime::tracing::Session::available())
    GTEST_SKIP() << "SDK disabled";
  auto                     state = std::make_shared<TrackingState>();
  holoflow::core::Registry registry;
  registry.register_sync("source", std::make_unique<TrackingSourceFactory>(state));
  registry.register_sync("sink", std::make_unique<SinkFactory>());
  const auto directory = trace_test::temporary_path("holoflow-explicit-lifecycle-");
  auto       session   = holoflow::runtime::tracing::Session::start();
  ASSERT_NE(session, nullptr);
  {
    holoflow::runtime::tracing::ScopedTrace lifecycle("Test Pipeline Lifecycle");
    holoflow::runtime::Compiler             compiler(
        registry,
        {.log_dir = directory, .dump_dot_on_failure = false, .max_section_cuda_graphs = 0});
    auto output = compiler.compile(source_sink_graph());
    for (int run = 0; run < 2; ++run) {
      holoflow::runtime::Scheduler scheduler(output->graph, output->sections, output->resources);
      scheduler.start();
      scheduler.request_stop();
      scheduler.wait();
    }
    EXPECT_TRUE(holoflow::runtime::tracing::Session::active());
    EXPECT_FALSE(std::filesystem::exists(directory / "trace_events.perfetto-trace"));
  }
  holoflow::runtime::section_diagnostics_file_writer().flush();
  session->stop_and_save(directory / "lifecycle.perfetto-trace");
  const auto trace = trace_test::read(directory / "lifecycle.perfetto-trace");
  for (const auto *name :
       {"Initialize Scheduler", "Drain Startup CUDA Streams", "Prepare Startup CUDA Graphs",
        "Create Scheduler Workers", "Scheduler Request Stop", "Scheduler Wait",
        "Stop Metrics Thread", "Submit Shutdown CUDA Graph Diagnostics"}) {
    size_t count = 0;
    for (const auto &slice : trace.slices)
      count += slice.name == name;
    EXPECT_EQ(count, 2) << name;
  }
  for (const auto *name :
       {"Collect CUDA Graph Storage Owners", "Inspect CUDA Graph Sections",
        "Inspect CUDA Graph Task Eligibility", "Inspect CUDA Graph Storage Domains",
        "Prepare CUDA Graph Sections", "Install CUDA Graph Cache"}) {
    EXPECT_TRUE(std::any_of(trace.slices.begin(), trace.slices.end(), [name](const auto &slice) {
      return slice.name == name && slice.complete;
    })) << name;
  }
  bool joined = false;
  for (const auto &slice : trace.slices) {
    joined |= slice.name.starts_with("Join Worker ");
    EXPECT_TRUE(slice.complete);
  }
  EXPECT_TRUE(joined);
}

TEST(CompilerTest, LoggingAndCompilationDoNotWaitForDiagnosticWrites) {
  using namespace std::chrono_literals;
  auto &writer = holoflow::runtime::section_diagnostics_file_writer();
  writer.flush();
  const auto         directory = trace_test::temporary_path("holoflow-buffered-compiler-");
  std::promise<void> started, release;
  auto               starting = started.get_future();
  auto               released = release.get_future();
  writer.submit_text(directory / "block.txt", [&] {
    started.set_value();
    EXPECT_EQ(released.wait_for(5s), std::future_status::ready);
    return std::string{};
  });
  ASSERT_EQ(starting.wait_for(3s), std::future_status::ready);
  holoflow::core::Registry registry;
  double                   setup_ms = 0;
  for (int i = 0; i < 20; ++i) {
    const auto                  start = std::chrono::steady_clock::now();
    holoflow::runtime::Compiler compiler(
        registry,
        {.log_dir = directory / "output", .dump_dot_on_failure = false, .enable_profiling = false});
    setup_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    EXPECT_NE(compiler.compile({}), nullptr);
  }
  EXPECT_FALSE(std::filesystem::exists(directory / "output"));
  release.set_value();
  writer.flush();
  EXPECT_TRUE(std::filesystem::exists(directory / "output" / "compiler.log"));
  std::cout << "Compiler logging setup: mean_ms=" << setup_ms / 20 << '\n';
}

TEST(CompilerTest, BufferedLogContainsOnlyLatestCompilationAndIncludesFailures) {
  auto                    &writer    = holoflow::runtime::section_diagnostics_file_writer();
  const auto               directory = trace_test::temporary_path("holoflow-latest-compiler-log-");
  holoflow::core::Registry registry;
  holoflow::runtime::Compiler compiler(
      registry, {.log_dir = directory, .dump_dot_on_failure = false, .enable_profiling = false});
  auto read_log = [&] {
    writer.flush();
    std::ifstream file(directory / "compiler.log");
    return std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  };
  GraphSpec invalid;
  auto      node = add_vertex(NodeSpec{"missing", "first_missing", {}}, invalid);
  EXPECT_THROW((void)compiler.compile(invalid), std::runtime_error);
  EXPECT_NE(read_log().find("first_missing"), std::string::npos);
  invalid[node].kind = "second_missing";
  EXPECT_THROW((void)compiler.compile(invalid), std::runtime_error);
  auto log = read_log();
  EXPECT_NE(log.find("Compilation Failed"), std::string::npos);
  EXPECT_NE(log.find("second_missing"), std::string::npos);
  EXPECT_EQ(log.find("first_missing"), std::string::npos);
  EXPECT_NE(compiler.compile({}), nullptr);
  EXPECT_EQ(read_log().find("Compilation Failed"), std::string::npos);
}
