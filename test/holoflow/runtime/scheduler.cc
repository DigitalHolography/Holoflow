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
using holoflow::test::make_desc;
using holoflow::test::make_infer_result;
using holoflow::test::RecordingAsyncFactory;
using holoflow::test::RecordingSyncFactory;
using holoflow::test::StubAsyncTask;
using holoflow::test::StubSyncTask;
using holoflow::test::SyncFactorySpec;

std::vector<core::TDesc> copy_descs(std::span<const core::TDesc> descs) {
  return std::vector<core::TDesc>(descs.begin(), descs.end());
}

class GraphBuilder {
public:
  GraphBuilder &add_node(const std::string &id, const std::string &name, const std::string &kind,
                         const nlohmann::json &settings = {}) {
    core::NodeSpec node;
    node.name     = name;
    node.kind     = kind;
    node.settings = settings;
    auto v        = boost::add_vertex(node, graph_);
    nodes_.emplace(id, v);
    return *this;
  }

  GraphBuilder &add_node(const std::string &name, const std::string &kind) {
    return add_node(name, name, kind);
  }

  GraphBuilder &add_edge(const std::string &src_id, const std::string &dst_id, int out_idx = 0,
                         int in_idx = 0) {
    core::EdgeSpec edge{out_idx, in_idx};
    boost::add_edge(nodes_.at(src_id), nodes_.at(dst_id), edge, graph_);
    return *this;
  }

  core::GraphSpec finish() { return std::move(graph_); }

private:
  core::GraphSpec                                                        graph_;
  std::map<std::string, core::GraphSpec::vertex_descriptor, std::less<>> nodes_;
};

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

TEST(Scheduler, StreamSynchronization) {
  core::Registry registry;

  std::vector<std::chrono::steady_clock::time_point> source_exec_times;
  std::vector<std::chrono::steady_clock::time_point> process_push_times;
  std::vector<std::chrono::steady_clock::time_point> process_pop_times;
  std::vector<std::chrono::steady_clock::time_point> sink_exec_times;
  std::mutex timing_mutex;
  
  std::atomic<bool> push_ready{true};
  std::atomic<bool> pop_ready{false};

  SyncFactorySpec source_spec;
  source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4}, core::DType::F32)});
  };
  source_spec.create = [&source_exec_times, &timing_mutex, &push_ready](
      std::span<const core::TDesc>, const nlohmann::json &, const core::SyncCreateCtx &ctx) {
    class TimedSourceTask : public StubSyncTask {
    public:
      TimedSourceTask(core::SyncCreateCtx ctx, 
                   std::vector<std::chrono::steady_clock::time_point>& times,
                   std::mutex& mutex,
                   std::atomic<bool>& push_flag)
        : StubSyncTask(ctx), exec_times_(times), timing_mutex_(mutex), push_ready_(push_flag) {}
        
      core::OpResult execute(core::SyncCtx&) override {
        if (!push_ready_) {
          return core::OpResult::Ok;
        }
        {
          std::lock_guard<std::mutex> lock(timing_mutex_);
          exec_times_.push_back(std::chrono::steady_clock::now());
        }
        return core::OpResult::Ok;
      }
      
    private:
      std::vector<std::chrono::steady_clock::time_point>& exec_times_;
      std::mutex& timing_mutex_;
      std::atomic<bool>& push_ready_;
    };
    
    return std::make_unique<TimedSourceTask>(ctx, source_exec_times, timing_mutex, push_ready);
  };
  auto *source_factory = new RecordingSyncFactory(std::move(source_spec));
  registry.register_sync("source", std::unique_ptr<RecordingSyncFactory>(source_factory));

  AsyncFactorySpec process_spec;
  process_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Async, copy_descs(inputs), copy_descs(inputs));
  };
  process_spec.create = [&process_push_times, &process_pop_times, &timing_mutex, &push_ready, &pop_ready](
      std::span<const core::TDesc>, const nlohmann::json &, const core::AsyncCreateCtx &ctx) {
    class TimedProcessTask : public StubAsyncTask {
    public:
      TimedProcessTask(core::AsyncCreateCtx ctx, 
                    std::vector<std::chrono::steady_clock::time_point>& push_times,
                    std::vector<std::chrono::steady_clock::time_point>& pop_times,
                    std::mutex& mutex,
                    std::atomic<bool>& push_flag,
                    std::atomic<bool>& pop_flag
                    ) 
        : StubAsyncTask(ctx), push_times_(push_times), pop_times_(pop_times), timing_mutex_(mutex),
          push_ready_(push_flag), pop_ready_(pop_flag),
          push_start_(), pop_start_(), last_push_time_() {}
        
      core::OpResult try_push(core::AsyncPushCtx&) override {
        auto now = std::chrono::steady_clock::now();

        if (!push_ready_) {
          return core::OpResult::NotReady;
        }
        
        if (push_start_.time_since_epoch().count() == 0) {
          // Start push operation
          push_start_ = now;
          return core::OpResult::NotReady;
        }
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - push_start_).count();
        if (elapsed < 10) {
          // Still processing
          return core::OpResult::NotReady;  
        }
        
        // Push complete
        {
          std::lock_guard<std::mutex> lock(timing_mutex_);
          push_times_.push_back(now);
        }
        push_ready_ = false;
        pop_ready_ = true; // Signal that data is ready for pop
        last_push_time_ = now;
        push_start_ = std::chrono::steady_clock::time_point(); // Reset timer
        return core::OpResult::Ok;
      }
      
      core::OpResult try_pop(core::AsyncPopCtx&) override {
        auto now = std::chrono::steady_clock::now();

        if (!pop_ready_) {
          return core::OpResult::NotReady;
        }

        // Enforce minimum delay since push completed
        if (last_push_time_) {
          auto since_push = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - *last_push_time_).count();
          if (since_push < 10) {
            return core::OpResult::NotReady;
          }
        }
        
        if (pop_start_.time_since_epoch().count() == 0) {
          // Start pop operation
          pop_start_ = now;
          return core::OpResult::NotReady;
        }
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - pop_start_).count();
        if (elapsed < 10) {
          // Still processing
          return core::OpResult::NotReady;
        }
        
        // Pop complete
        {
          std::lock_guard<std::mutex> lock(timing_mutex_);
          pop_times_.push_back(now);
        }
        pop_ready_ = false; // Reset pop flag
        pop_start_ = std::chrono::steady_clock::time_point(); // Reset timer
        push_ready_ = true; // Ready for next push
        last_push_time_.reset();
        return core::OpResult::Ok;
      }

    private:
      std::vector<std::chrono::steady_clock::time_point>& push_times_;
      std::vector<std::chrono::steady_clock::time_point>& pop_times_;
      std::mutex& timing_mutex_;
      std::atomic<bool>& push_ready_;
      std::atomic<bool>& pop_ready_;
      std::chrono::steady_clock::time_point push_start_;
      std::chrono::steady_clock::time_point pop_start_;
      std::optional<std::chrono::steady_clock::time_point> last_push_time_;
    };
    
    return std::make_unique<TimedProcessTask>(ctx, process_push_times, process_pop_times, timing_mutex,
                                          push_ready, pop_ready);
  };
  auto *process_factory = new RecordingAsyncFactory(std::move(process_spec));
  registry.register_async("process", std::unique_ptr<RecordingAsyncFactory>(process_factory));

  SyncFactorySpec sink_spec;
  sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
    return make_infer_result(TaskKind::Sync, copy_descs(inputs), {});
  };
  sink_spec.create = [&sink_exec_times, &timing_mutex, &pop_ready](
      std::span<const core::TDesc>, const nlohmann::json &, const core::SyncCreateCtx &ctx) {
    class TimedSinkTask : public StubSyncTask {
    public:
      TimedSinkTask(core::SyncCreateCtx ctx, 
                   std::vector<std::chrono::steady_clock::time_point>& times,
                   std::mutex& mutex,
                   std::atomic<bool>& pop_flag)
        : StubSyncTask(ctx), exec_times_(times), timing_mutex_(mutex), pop_ready_(pop_flag) {}
        
      core::OpResult execute(core::SyncCtx&) override {
        if (!pop_ready_) {
          return core::OpResult::Ok;
        }
        {
          std::lock_guard<std::mutex> lock(timing_mutex_);
          exec_times_.push_back(std::chrono::steady_clock::now());
        }
        pop_ready_ = false;
        return core::OpResult::Ok;
      }
      
    private:
      std::vector<std::chrono::steady_clock::time_point>& exec_times_;
      std::mutex& timing_mutex_;
      std::atomic<bool>& pop_ready_;
    };
    
    return std::make_unique<TimedSinkTask>(ctx, sink_exec_times, timing_mutex, pop_ready);
  };
  auto *sink_factory = new RecordingSyncFactory(std::move(sink_spec));
  registry.register_sync("sink", std::unique_ptr<RecordingSyncFactory>(sink_factory));

  // Create graph with sync->async->sync pattern
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
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  scheduler.request_stop();
  scheduler.wait();

  // Basic execution verification
  ASSERT_GT(source_exec_times.size(), 0) << "Source task never executed";
  ASSERT_GT(process_push_times.size(), 0) << "Process push never executed";
  ASSERT_GT(process_pop_times.size(), 0) << "Process pop never executed";
  ASSERT_GT(sink_exec_times.size(), 0) << "Sink task never executed";
  
  // Verify sections were created
  ASSERT_EQ(output->sections.size(), 2) << "Expected source, process, and sink sections";

  // Verify execution order for a sample of iterations
  size_t num_samples = std::min({
    source_exec_times.size(), 
    process_push_times.size(),
    process_pop_times.size(),
    sink_exec_times.size()
  });

  ASSERT_GT(num_samples, 0) << "No complete iterations found";
  
  for (size_t i = 0; i < num_samples; ++i) {
    // Verify correct execution order: source -> process_push -> process_pop -> sink
    EXPECT_LT(source_exec_times[i], process_push_times[i]) 
      << "Source must execute before Process push for iteration " << i;
    EXPECT_LT(process_push_times[i], process_pop_times[i])
      << "Process push must complete before pop for iteration " << i;
    EXPECT_LT(process_pop_times[i], sink_exec_times[i])
      << "Process pop must complete before Sink for iteration " << i;

    auto push_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
      process_push_times[i] - source_exec_times[i]).count();
    EXPECT_GT(push_delay, 0) 
      << "Expected delay between Source and Process push in iteration " << i;

    auto pop_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
      process_pop_times[i] - process_push_times[i]).count();
    EXPECT_GE(pop_delay, 10)
      << "Expected minimum 10ms delay between push and pop in iteration " << i;

    auto sink_delay = std::chrono::duration_cast<std::chrono::milliseconds>(
      sink_exec_times[i] - process_pop_times[i]).count();
    EXPECT_GT(sink_delay, 0)
      << "Expected delay between Process pop and Sink in iteration " << i;
  }
}

TEST(Scheduler, ErrorHandling) {}


TEST(Scheduler, OwnedTensorManagement) {}

} // namespace
} // namespace holoflow::runtime
