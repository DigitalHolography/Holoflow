// Copyright 2025 Digital Holography Foundation
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
#include <functional>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <boost/graph/graph_traits.hpp>
#include <nlohmann/json.hpp>

#include "factory_maker.hh"
#include "graph_builder.hh"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/graph_display.hh"
#include "holoflow/runtime/graph_exec.hh"

#include <chrono>
#include <thread>

namespace holoflow::runtime {
namespace {

using holoflow::core::TaskKind;
using holoflow::test::AsyncFactorySpec;
using holoflow::test::copy_descs;
using holoflow::test::GraphBuilder;
using holoflow::test::make_desc;
using holoflow::test::make_infer_result;
using holoflow::test::RecordingAsyncFactory;
using holoflow::test::RecordingSyncFactory;
using holoflow::test::StubAsyncTask;
using holoflow::test::StubSyncTask;
using holoflow::test::SyncFactorySpec;

TEST(Scheduler, BasicLifecycle) {
  core::Registry registry;

  SyncFactorySpec source_spec;
  source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)});
  };
  registry.register_sync("source", std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

  SyncFactorySpec sink_spec;
  sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    auto input_descs = copy_descs(inputs);
    return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
  };
  registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

  GraphBuilder builder;
  builder.add_node("src", "src", "source");
  builder.add_node("snk", "snk", "sink");
  builder.add_edge("src", "snk", 0, 0);
  auto graph = builder.finish();

  Compiler compiler(registry);

  auto output = compiler.compile(graph);

  Scheduler scheduler(output->graph, output->sections, output->resources);

  EXPECT_FALSE(scheduler.is_running());
  EXPECT_FALSE(scheduler.stop_requested());

  scheduler.start();
  EXPECT_TRUE(scheduler.is_running());
  EXPECT_FALSE(scheduler.stop_requested());

  scheduler.request_stop();
  EXPECT_TRUE(scheduler.stop_requested());

  scheduler.wait();
  EXPECT_FALSE(scheduler.is_running());
  EXPECT_TRUE(scheduler.stop_requested());
}

TEST(Scheduler, SectionExecution) {
  core::Registry registry;

  SyncFactorySpec source_spec;
  source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)});
  };
  auto *source_factory = new RecordingSyncFactory(std::move(source_spec));
  registry.register_sync("source", std::unique_ptr<RecordingSyncFactory>(source_factory));

  AsyncFactorySpec process_spec;
  process_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Async, copy_descs(inputs), copy_descs(inputs));
  };
  auto *process_factory = new RecordingAsyncFactory(std::move(process_spec));
  registry.register_async("process", std::unique_ptr<RecordingAsyncFactory>(process_factory));

  SyncFactorySpec sink_spec;
  sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, copy_descs(inputs), {});
  };
  auto *sink_factory = new RecordingSyncFactory(std::move(sink_spec));
  registry.register_sync("sink", std::unique_ptr<RecordingSyncFactory>(sink_factory));

  GraphBuilder builder;
  builder.add_node("src", "src", "source");
  builder.add_node("proc", "proc", "process");
  builder.add_node("snk", "snk", "sink");
  builder.add_edge("src", "proc", 0, 0);
  builder.add_edge("proc", "snk", 0, 0);
  auto graph = builder.finish();

  Compiler compiler(registry);
  auto     output = compiler.compile(graph);

  Scheduler scheduler(output->graph, output->sections, output->resources);

  scheduler.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  scheduler.request_stop();
  scheduler.wait();

  // Verify execution by checking if the factories were called
  EXPECT_GT(source_factory->create_calls().size(), 0);
  EXPECT_GT(process_factory->create_calls().size(), 0);
  EXPECT_GT(sink_factory->create_calls().size(), 0);

  // Start scheduler and let it run briefly
  scheduler.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  scheduler.request_stop();
  scheduler.wait();

  // Verify execution
  EXPECT_GT(source_factory->create_calls().size(), 0);
  EXPECT_GT(process_factory->create_calls().size(), 0);
  EXPECT_GT(sink_factory->create_calls().size(), 0);
}

auto make_timed_sync_creator(std::vector<std::chrono::steady_clock::time_point> &exec_times,
                             std::mutex                                         &timing_mutex)
    -> std::function<std::unique_ptr<StubSyncTask>(
        std::span<const core::TDesc>, const nlohmann::json &, const core::SyncCreateCtx &)> {
  return [&exec_times,
          &timing_mutex](std::span<const core::TDesc>, const nlohmann::json &,
                         const core::SyncCreateCtx &ctx) -> std::unique_ptr<StubSyncTask> {
    class TimedSyncTask : public StubSyncTask {
    public:
      TimedSyncTask(core::SyncCreateCtx                                 ctx,
                    std::vector<std::chrono::steady_clock::time_point> &times, std::mutex &mutex)
          : StubSyncTask(ctx), exec_times_(times), timing_mutex_(mutex) {}

      core::OpResult execute(core::SyncCtx &) override {
        {
          std::lock_guard<std::mutex> lock(timing_mutex_);
          exec_times_.push_back(std::chrono::steady_clock::now());
        }
        return core::OpResult::Ok;
      }

    private:
      std::vector<std::chrono::steady_clock::time_point> &exec_times_;
      std::mutex                                         &timing_mutex_;
    };

    return std::make_unique<TimedSyncTask>(ctx, exec_times, timing_mutex);
  };
}

auto make_timed_async_creator(std::vector<std::chrono::steady_clock::time_point> &push_times,
                              std::vector<std::chrono::steady_clock::time_point> &pop_times,
                              std::mutex &timing_mutex, std::atomic<bool> &has_data)
    -> std::function<std::unique_ptr<StubAsyncTask>(
        std::span<const core::TDesc>, const nlohmann::json &, const core::AsyncCreateCtx &)> {
  return [&push_times, &pop_times, &timing_mutex,
          &has_data](std::span<const core::TDesc>, const nlohmann::json &,
                     const core::AsyncCreateCtx &ctx) -> std::unique_ptr<StubAsyncTask> {
    class TimedProcessTask : public StubAsyncTask {
    public:
      TimedProcessTask(core::AsyncCreateCtx                                ctx,
                       std::vector<std::chrono::steady_clock::time_point> &push_times,
                       std::vector<std::chrono::steady_clock::time_point> &pop_times,
                       std::mutex &mutex, std::atomic<bool> &has_data)
          : StubAsyncTask(ctx), push_times_(push_times), pop_times_(pop_times),
            timing_mutex_(mutex), has_data_(has_data) {}

      core::OpResult try_push(core::AsyncPushCtx &) override {
        {
          std::lock_guard<std::mutex> lock(timing_mutex_);
          push_times_.push_back(std::chrono::steady_clock::now());
        }
        has_data_ = true;
        return core::OpResult::Ok;
      }

      core::OpResult try_pop(core::AsyncPopCtx &) override {
        if (!has_data_) {
          return core::OpResult::NotReady;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        {
          std::lock_guard<std::mutex> lock(timing_mutex_);
          pop_times_.push_back(std::chrono::steady_clock::now());
        }
        has_data_ = false;
        return core::OpResult::Ok;
      }

    private:
      std::vector<std::chrono::steady_clock::time_point> &push_times_;
      std::vector<std::chrono::steady_clock::time_point> &pop_times_;
      std::mutex                                         &timing_mutex_;
      std::atomic<bool>                                  &has_data_;
    };

    return std::make_unique<TimedProcessTask>(ctx, push_times, pop_times, timing_mutex, has_data);
  };
}

TEST(Scheduler, StreamSynchronization) {

  core::Registry registry;

  std::vector<std::chrono::steady_clock::time_point> source_exec_times;
  std::vector<std::chrono::steady_clock::time_point> process_push_times;
  std::vector<std::chrono::steady_clock::time_point> process_pop_times;
  std::vector<std::chrono::steady_clock::time_point> sink_exec_times;
  std::mutex                                         timing_mutex;

  std::atomic<bool> process_has_data{false};

  // Source - sync node
  SyncFactorySpec source_spec;
  source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)});
  };
  source_spec.create   = make_timed_sync_creator(source_exec_times, timing_mutex);
  auto *source_factory = new RecordingSyncFactory(std::move(source_spec));
  registry.register_sync("source", std::unique_ptr<RecordingSyncFactory>(source_factory));

  // Process - async node
  AsyncFactorySpec process_spec;
  process_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Async, copy_descs(inputs), copy_descs(inputs));
  };
  process_spec.create   = make_timed_async_creator(process_push_times, process_pop_times,
                                                   timing_mutex, process_has_data);
  auto *process_factory = new RecordingAsyncFactory(std::move(process_spec));
  registry.register_async("process", std::unique_ptr<RecordingAsyncFactory>(process_factory));

  // Sink - sync node
  SyncFactorySpec sink_spec;
  sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, copy_descs(inputs), {});
  };
  sink_spec.create   = make_timed_sync_creator(sink_exec_times, timing_mutex);
  auto *sink_factory = new RecordingSyncFactory(std::move(sink_spec));
  registry.register_sync("sink", std::unique_ptr<RecordingSyncFactory>(sink_factory));

  {
    // Create graph with sync->async->sync pattern
    GraphBuilder builder;
    builder.add_node("src", "src", "source");
    builder.add_node("proc", "proc", "process");
    builder.add_node("snk", "snk", "sink");
    builder.add_edge("src", "proc", 0, 0);
    builder.add_edge("proc", "snk", 0, 0);
    auto graph = builder.finish();

    Compiler compiler(registry);
    auto     output = compiler.compile(graph);

    Scheduler scheduler(output->graph, output->sections, output->resources);

    scheduler.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    scheduler.request_stop();
    scheduler.wait();

    // Basic execution verification
    ASSERT_GT(source_exec_times.size(), 0) << "Source task never executed";
    ASSERT_GT(process_push_times.size(), 0) << "Process push never executed";
    ASSERT_GT(process_pop_times.size(), 0) << "Process pop never executed";
    ASSERT_GT(sink_exec_times.size(), 0) << "Sink task never executed";

    // Verify execution order for a sample of iterations
    size_t num_samples = std::min({source_exec_times.size(), process_push_times.size(),
                                   process_pop_times.size(), sink_exec_times.size()});

    ASSERT_GT(num_samples, 0) << "No complete iterations found";

    for (size_t i = 0; i < num_samples; ++i) {
      EXPECT_LT(source_exec_times[i], process_push_times[i])
          << "Source must execute before Process push for iteration " << i;
      EXPECT_LT(process_push_times[i], process_pop_times[i])
          << "Process push must complete before pop for iteration " << i;
      EXPECT_LT(process_pop_times[i], sink_exec_times[i])
          << "Process pop must complete before Sink for iteration " << i;

      auto push_delay = std::chrono::duration_cast<std::chrono::nanoseconds>(process_push_times[i] -
                                                                             source_exec_times[i])
                            .count();
      EXPECT_GT(push_delay, 0) << "Expected delay between Source and Process push in iteration "
                               << i;

      auto pop_delay = std::chrono::duration_cast<std::chrono::milliseconds>(process_pop_times[i] -
                                                                             process_push_times[i])
                           .count();
      EXPECT_GE(pop_delay, 10) << "Expected minimum 10ms delay between push and pop in iteration "
                               << i;

      auto sink_delay = std::chrono::duration_cast<std::chrono::nanoseconds>(sink_exec_times[i] -
                                                                             process_pop_times[i])
                            .count();
      EXPECT_GT(sink_delay, 0) << "Expected delay between Process pop and Sink in iteration " << i;
    }
  }
}

class SourceTask : public StubSyncTask {
public:
  SourceTask(const core::SyncCreateCtx &ctx) : StubSyncTask(ctx) {}
};

class OtherSourceTask : public StubSyncTask {
public:
  OtherSourceTask(const core::SyncCreateCtx &ctx) : StubSyncTask(ctx) {}
};

TEST(Scheduler, WrongUpdate) {
  core::Registry registry;

  SyncFactorySpec source_spec;
  source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)});
  };

  source_spec.create = [](std::span<const core::TDesc>, const nlohmann::json &,
                          const core::SyncCreateCtx &ctx) {
    return std::make_unique<SourceTask>(ctx);
  };
  source_spec.update = [](std::unique_ptr<core::ISyncTask> old_task,
                          std::span<const core::TDesc> input_descs, const nlohmann::json &jsettings,
                          const core::SyncCreateCtx &ctx) {
    (void)input_descs;
    (void)jsettings;
    EXPECT_NE(dynamic_cast<SourceTask *>(old_task.get()), nullptr);
    return std::make_unique<SourceTask>(ctx);
  };
  registry.register_sync("source", std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

  SyncFactorySpec sink_spec;
  sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    auto input_descs = copy_descs(inputs);
    return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
  };
  registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));
  
    SyncFactorySpec other_source_spec;
    other_source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
      return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)});
    };
    other_source_spec.create = [](std::span<const core::TDesc>, const nlohmann::json &,
                                  const core::SyncCreateCtx &ctx) {
      return std::make_unique<OtherSourceTask>(ctx);
    };
    other_source_spec.update = [](std::unique_ptr<core::ISyncTask> old_task,
                                  std::span<const core::TDesc>     input_descs,
                                  const nlohmann::json &jsettings, const core::SyncCreateCtx &ctx) {
      (void)input_descs;
      (void)jsettings;
      EXPECT_NE(dynamic_cast<OtherSourceTask *>(old_task.get()), nullptr);
      return std::make_unique<OtherSourceTask>(ctx);
    };
    registry.register_sync("other_source",
                           std::make_unique<RecordingSyncFactory>(std::move(other_source_spec)));

  GraphBuilder builder;
  builder.add_node("src", "src", "source");
  builder.add_node("snk", "snk", "sink");
  builder.add_edge("src", "snk", 0, 0);
  auto graph = builder.finish();

  Compiler compiler(registry);

  auto output = compiler.compile(graph);

  Scheduler scheduler(output->graph, output->sections, output->resources);

  EXPECT_FALSE(scheduler.is_running());
  EXPECT_FALSE(scheduler.stop_requested());

  scheduler.start();
  EXPECT_TRUE(scheduler.is_running());
  EXPECT_FALSE(scheduler.stop_requested());

  scheduler.request_stop();
  scheduler.wait();
  EXPECT_FALSE(scheduler.is_running());

  builder.change_node("src", "other_source", {});
  graph = builder.finish();

  Compiler compiler2(registry);
  output = compiler2.compile(graph, std::move(output));

  Scheduler scheduler2(output->graph, output->sections, output->resources);

  scheduler2.start();
  EXPECT_TRUE(scheduler2.is_running());
}

TEST(Scheduler, OwnedInputHandling) {
  core::Registry registry;

  SyncFactorySpec source_spec;
  source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)});
  };
  auto *source_factory = new RecordingSyncFactory(std::move(source_spec));
  registry.register_sync("source", std::unique_ptr<RecordingSyncFactory>(source_factory));

  AsyncFactorySpec process_spec;
  process_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Async, copy_descs(inputs), copy_descs(inputs), {true}, {true});
  };
  auto *process_factory = new RecordingAsyncFactory(std::move(process_spec));
  registry.register_async("process", std::unique_ptr<RecordingAsyncFactory>(process_factory));

  SyncFactorySpec sink_spec;
  sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, copy_descs(inputs), {}, {false}, {});
  };
  auto *sink_factory = new RecordingSyncFactory(std::move(sink_spec));
  registry.register_sync("sink", std::unique_ptr<RecordingSyncFactory>(sink_factory));

  GraphBuilder builder;
  builder.add_node("src", "src", "source");
  builder.add_node("proc", "proc", "process");
  builder.add_node("snk", "snk", "sink");
  builder.add_edge("src", "proc", 0, 0);
  builder.add_edge("proc", "snk", 0, 0);
  auto graph = builder.finish();

  Compiler compiler(registry);
  auto output = compiler.compile(graph);
  Scheduler scheduler(output->graph, output->sections, output->resources);

  scheduler.start();
  EXPECT_TRUE(scheduler.is_running());
  EXPECT_FALSE(scheduler.stop_requested());

  scheduler.request_stop();
  scheduler.wait();
  EXPECT_FALSE(scheduler.is_running());

  EXPECT_EQ(source_factory->create_calls().size(), 1);
  EXPECT_EQ(process_factory->create_calls().size(), 1);
  EXPECT_EQ(sink_factory->create_calls().size(), 1);
    

  scheduler.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  scheduler.request_stop();
  scheduler.wait();

  EXPECT_EQ(source_factory->create_calls().size(), 1);
  EXPECT_EQ(process_factory->create_calls().size(), 1);
  EXPECT_EQ(sink_factory->create_calls().size(), 1);
}

TEST(Scheduler, OwnedInputWrongUpdate) {
  core::Registry registry;

  SyncFactorySpec source_spec;
  source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)}, {}, {true});
  };
  source_spec.create = [](std::span<const core::TDesc>, const nlohmann::json &,
                          const core::SyncCreateCtx &ctx) {
    return std::make_unique<SourceTask>(ctx);
  };
  source_spec.update = [](std::unique_ptr<core::ISyncTask> old_task,
                          std::span<const core::TDesc>, const nlohmann::json &,
                          const core::SyncCreateCtx &ctx) {
    EXPECT_NE(dynamic_cast<SourceTask *>(old_task.get()), nullptr);
    return std::make_unique<SourceTask>(ctx);
  };
  registry.register_sync("source", std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

  SyncFactorySpec sink_spec;
  sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, copy_descs(inputs), {}, {false}, {});
  };
  registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

  SyncFactorySpec other_source_spec;
  other_source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)});
  };
  other_source_spec.create = [](std::span<const core::TDesc>, const nlohmann::json &,
                                const core::SyncCreateCtx &ctx) {
    return std::make_unique<OtherSourceTask>(ctx);
  };
  other_source_spec.update = [](std::unique_ptr<core::ISyncTask> old_task,
                                std::span<const core::TDesc>, const nlohmann::json &,
                                const core::SyncCreateCtx &ctx) {
    EXPECT_NE(dynamic_cast<OtherSourceTask *>(old_task.get()), nullptr);
    return std::make_unique<OtherSourceTask>(ctx);
  };
  registry.register_sync("other_source", 
                        std::make_unique<RecordingSyncFactory>(std::move(other_source_spec)));

  GraphBuilder builder;
  builder.add_node("src", "src", "source");
  builder.add_node("snk", "snk", "sink");
  builder.add_edge("src", "snk", 0, 0);
  auto graph = builder.finish();

  Compiler compiler(registry);
  auto output = compiler.compile(graph);
  Scheduler scheduler(output->graph, output->sections, output->resources);

  scheduler.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  scheduler.request_stop();
  scheduler.wait();

  builder.change_node("src", "other_source", {});
  graph = builder.finish();

  Compiler compiler2(registry);
  output = compiler2.compile(graph, std::move(output));

  Scheduler scheduler2(output->graph, output->sections, output->resources);
  scheduler2.start();
  EXPECT_TRUE(scheduler2.is_running());
  
  scheduler2.request_stop();
  scheduler2.wait();
  EXPECT_FALSE(scheduler2.is_running());
}

TEST(Scheduler, UpdateMetrics) {
  core::Registry registry;

  SyncFactorySpec source_spec;
  source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)});
  };
  auto *source_factory = new RecordingSyncFactory(std::move(source_spec));
  registry.register_sync("source", std::unique_ptr<RecordingSyncFactory>(source_factory));

  SyncFactorySpec sink_spec;
  sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, copy_descs(inputs), {}, {false}, {});
  };
  auto *sink_factory = new RecordingSyncFactory(std::move(sink_spec));
  registry.register_sync("sink", std::unique_ptr<RecordingSyncFactory>(sink_factory));

  GraphBuilder builder;
  builder.add_node("src", "src", "source");
  builder.add_node("snk", "snk", "sink");
  builder.add_edge("src", "snk", 0, 0);
  auto graph = builder.finish();

  Compiler compiler(registry);
  auto output = compiler.compile(graph);
  Scheduler scheduler(output->graph, output->sections, output->resources);

  scheduler.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  scheduler.request_stop();
  scheduler.wait();
  auto metrics = scheduler.metrics();
  EXPECT_EQ(metrics.size(), 2);
  EXPECT_GT(metrics["src"].average_duration_ms, 0);
  EXPECT_GT(metrics["snk"].average_duration_ms, 0);

  scheduler.start();
  metrics = scheduler.metrics();
  EXPECT_EQ(metrics["src"].average_duration_ms, 0);
  EXPECT_EQ(metrics["snk"].average_duration_ms, 0);
}

TEST(Scheduler, StopRequest) {
  core::Registry registry;

  SyncFactorySpec source_spec;
  source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)});
  };
  auto *source_factory = new RecordingSyncFactory(std::move(source_spec));
  registry.register_sync("source", std::unique_ptr<RecordingSyncFactory>(source_factory));

  SyncFactorySpec sink_spec;
  sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, copy_descs(inputs), {}, {false}, {});
  };
  auto *sink_factory = new RecordingSyncFactory(std::move(sink_spec));
  registry.register_sync("sink", std::unique_ptr<RecordingSyncFactory>(sink_factory));

  GraphBuilder builder;
  builder.add_node("src", "src", "source");
  builder.add_node("snk", "snk", "sink");
  builder.add_edge("src", "snk", 0, 0);
  auto graph = builder.finish();

  Compiler compiler(registry);
  auto     output = compiler.compile(graph);

  Scheduler scheduler(output->graph, output->sections, output->resources);

  EXPECT_FALSE(scheduler.is_running());
  EXPECT_FALSE(scheduler.stop_requested());

  scheduler.start();
  EXPECT_TRUE(scheduler.is_running());
  EXPECT_FALSE(scheduler.stop_requested());

  scheduler.request_stop();
  EXPECT_TRUE(scheduler.stop_requested());

  scheduler.wait();
  EXPECT_FALSE(scheduler.is_running());
  EXPECT_TRUE(scheduler.stop_requested());
}

TEST(Scheduler, BigGraphAsyncNode) {
  core::Registry registry;

  SyncFactorySpec source_spec;
  source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)}, {}, {true});
  };
  auto *source_factory = new RecordingSyncFactory(std::move(source_spec));
  registry.register_sync("source", std::unique_ptr<RecordingSyncFactory>(source_factory));

  AsyncFactorySpec process_spec;
  process_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Async, copy_descs(inputs), copy_descs(inputs), {false}, {false});
  };
  auto *process_factory = new RecordingAsyncFactory(std::move(process_spec));
  registry.register_async("process", std::unique_ptr<RecordingAsyncFactory>(process_factory));

  SyncFactorySpec sink_spec;
  sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, copy_descs(inputs), {}, {false}, {});
  };
  auto *sink_factory = new RecordingSyncFactory(std::move(sink_spec));
  registry.register_sync("sink", std::unique_ptr<RecordingSyncFactory>(sink_factory));

  GraphBuilder builder;
  builder.add_node("src", "src", "source");

  const int num_nodes = 100;  
  for (int i = 1; i < num_nodes; ++i) {
    builder.add_node("node" + std::to_string(i), "proc" + std::to_string(i), "process");
    builder.add_node("snk" + std::to_string(i), "snk" + std::to_string(i), "sink");
    builder.add_edge("src", "node" + std::to_string(i), 0, 0);
    builder.add_edge("node" + std::to_string(i), "snk" + std::to_string(i), 0, 0);
  }

  auto graph = builder.finish();

  Compiler compiler(registry);
  auto     output = compiler.compile(graph);

  Scheduler scheduler(output->graph, output->sections, output->resources);

  scheduler.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  scheduler.request_stop();
  scheduler.wait();

  EXPECT_EQ(process_factory->create_calls().size(), num_nodes - 1);
}

} // namespace
} // namespace holoflow::runtime
