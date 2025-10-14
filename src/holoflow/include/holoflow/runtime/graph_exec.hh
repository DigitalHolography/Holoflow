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

#pragma once

#include <atomic>
#include <boost/graph/adjacency_list.hpp>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "curaii/cuda.hh"
#include "driver_types.h"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow_event/router.hh"

namespace holoflow::runtime {

struct NodePlan {
  core::NodeSpec    spec;     ///< Node specification.
  core::InferResult infer;    ///< Inference metadata.
  std::vector<int>  in_tids;  ///< Input tensor IDs.
  std::vector<int>  out_tids; ///< Output tensor IDs.
};

struct EdgePlan {
  core::EdgeSpec spec; ///< Edge specification.
  core::TDesc    desc; ///< Tensor descriptor.
  int            tid;  ///< Tensor ID.
};

using GraphPlan = boost::adjacency_list<boost::vecS,           // OutEdgeList
                                        boost::vecS,           // VertexList
                                        boost::bidirectionalS, // Directed graph
                                        NodePlan,              // Vertex properties
                                        EdgePlan               // Edge properties
                                        >;

struct ExecResouces {
  std::map<int, curaii::CudaStream>                   streams; ///< CUDA streams by ID.
  std::map<std::string, std::unique_ptr<core::ITask>> tasks;   ///< Task instances by ID.
  std::map<int, core::Tensor>                         tensors; ///< Allocated tensors by ID.
};

struct Section {
  int                                       id;         ///< Section ID.
  std::string                               name;       ///< Section name (for logging).
  cudaStream_t                              stream;     ///< CUDA stream for this section.
  std::vector<GraphPlan::vertex_descriptor> sync_topo;  ///< Synchronous nodes in topological order.
  std::vector<GraphPlan::vertex_descriptor> async_cons; ///< Asynchronous consumer nodes.
  std::vector<GraphPlan::vertex_descriptor> async_prod; ///< Asynchronous producer nodes
};

struct SyncRt {
  core::ISyncTask         *task = nullptr;
  std::vector<core::TView> in_views;
  std::vector<core::TView> out_views;
  core::SyncCtx            ctx{};
};

struct AsyncRt {
  core::IAsyncTask        *task = nullptr;
  std::vector<core::TView> in_views;
  std::vector<core::TView> out_views;
  core::AsyncPushCtx       pctx{};
  core::AsyncPopCtx        xctx{};
};

using NodeRt = std::variant<SyncRt, AsyncRt>;

struct NodeMetrics {
  double   average_duration_ms                = 0.0;
  double   runs_per_second                    = 0.0;
  double   host_throughput_bytes_per_second   = 0.0;
  double   device_throughput_bytes_per_second = 0.0;
  uint64_t sample_count                       = 0;
};

class Scheduler {
public:
  Scheduler(const GraphPlan &graph, const std::vector<Section> &sections, ExecResouces &res,
            std::chrono::milliseconds metrics_interval = std::chrono::milliseconds{1000});

  ~Scheduler();

  void set_metrics_interval(std::chrono::milliseconds interval);
  [[nodiscard]] std::map<std::string, NodeMetrics> metrics() const;

  void start();
  void request_stop();
  void wait();

  bool is_running() const;
  bool stop_requested() const;

  [[nodiscard]] bool ui_try_send(const std::string &node_id, nlohmann::json &&data) noexcept;

  [[nodiscard]] std::optional<holoflow_event::Event> ui_try_receive() noexcept;

private:
  void build_event_handles();
  void build_nodes_rts();
  void reset_metrics_state();
  void start_metrics_thread();
  void stop_metrics_thread();
  void metrics_loop();
  void aggregate_metrics(double interval_seconds);
  void record_node_sample(std::size_t idx, uint64_t duration_ns, uint64_t host_bytes,
                          uint64_t device_bytes);
  static std::pair<uint64_t, uint64_t> sum_bytes(std::span<const core::TView> views);

  void run_router();

  void run_section(int section_id);

  /// This function acquires all owned inputs for the given node,
  /// and updates the corresponding TViews in tviews_.
  /// This function blocks until all owned inputs are acquired.
  /// @warning If stop_ is set while waiting, the function returns early,
  /// and some owned inputs may not be acquired.
  /// @warning This function must be called on a synchronous or asynchronous
  /// producer node only.
  void acquire_owned_inputs(GraphPlan::vertex_descriptor v);

  /// This function releases all owned outputs for the given node,
  /// and clears the corresponding TViews in tviews_.
  /// This function does not block.
  /// @warning Cleared twiews will be set to {nullptr, {}}, but their
  /// indices in tviews_ remain accessible.
  void release_owned_outputs(GraphPlan::vertex_descriptor v);

  /// Refreshes both input and output TViews in the `SyncRt` of the given node
  /// based on the current `tviews_`.
  /// @warning This function must be called on a synchronous node only.
  void refresh_views_sync(GraphPlan::vertex_descriptor v);

  /// Refreshes output TViews in the `AsyncRt` of the given node
  /// based on the current `tviews_`.
  /// @warning This function must be called on an asynchronous consumer node only.
  void refresh_views_async_cons(GraphPlan::vertex_descriptor v);

  /// Refreshes input TViews in the `AsyncRt` of the given node
  /// based on the current `tviews_`.
  /// @warning This function must be called on an asynchronous producer node only.
  void refresh_views_async_prod(GraphPlan::vertex_descriptor v);

  /// Executes a synchronous node.
  /// The input and output TViews in the `SyncRt` must be up-to-date
  /// before calling this function.
  /// @warning This function must be called on a synchronous node only.
  void run_sync(GraphPlan::vertex_descriptor v);

  /// Executes an asynchronous consumer node.
  /// The output TViews in the `AsyncRt` must be up-to-date before calling this function.
  /// @warning This function must be called on an asynchronous consumer node only.
  void run_async_cons(GraphPlan::vertex_descriptor v);

  /// Executes an asynchronous producer node.
  /// The input TViews in the `AsyncRt` must be up-to-date before calling
  /// this function.
  /// @warning This function must be called on an asynchronous producer node only.
  void run_async_prod(GraphPlan::vertex_descriptor v);

  /// This function updates the TViews in tviews_ based on the outputs
  /// of the given synchronous node.
  /// @warning This function must be called on a synchronous node only.
  void refresh_outputs_sync(GraphPlan::vertex_descriptor v);

  /// This function updates the TViews in tviews_ based on the outputs
  /// of the given asynchronous consumer node.
  /// @warning This function must be called on an asynchronous consumer node only.
  void refresh_outputs_async_cons(GraphPlan::vertex_descriptor v);

private:
  std::atomic<bool>           running_{false}; ///< True if the scheduler is running.
  std::atomic<bool>           stop_{false};    ///< True if a stop has been requested.
  const GraphPlan            &graph_;          ///< The computational graph to execute.
  const std::vector<Section> &sections_;       ///< Execution sections.
  ExecResouces               &res_;            ///< Execution resources (streams, tasks, tensors).

  /// Current TViews for all tensors by their IDs.
  /// If a TView is {nullptr, {}}, it means the tensor is not currently
  /// available (owned tensor not aqcuired).
  /// They are exclusive to sections, i.e. no sharing between sections.
  std::vector<core::TView> tviews_;

  std::vector<NodeRt>      node_rts_; ///< Runtime data for each node.
  std::vector<std::string> node_names_;

  struct NodeMetricAccumulator {
    std::atomic<uint64_t> duration_ns{0};
    std::atomic<uint64_t> run_count{0};
    std::atomic<uint64_t> host_bytes{0};
    std::atomic<uint64_t> device_bytes{0};

    NodeMetricAccumulator() = default;

    NodeMetricAccumulator(const NodeMetricAccumulator &other) {
      duration_ns.store(other.duration_ns.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
      run_count.store(other.run_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
      host_bytes.store(other.host_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
      device_bytes.store(other.device_bytes.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    }

    NodeMetricAccumulator &operator=(const NodeMetricAccumulator &other) {
      if (this == &other) {
        return *this;
      }
      duration_ns.store(other.duration_ns.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
      run_count.store(other.run_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
      host_bytes.store(other.host_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
      device_bytes.store(other.device_bytes.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
      return *this;
    }

    NodeMetricAccumulator(NodeMetricAccumulator &&other) noexcept : NodeMetricAccumulator(other) {}

    NodeMetricAccumulator &operator=(NodeMetricAccumulator &&other) noexcept {
      return (*this = other);
    }
  };

  std::vector<NodeMetricAccumulator> metric_accumulators_;

  mutable std::mutex                 metrics_mutex_;
  std::map<std::string, NodeMetrics> latest_metrics_;
  std::chrono::milliseconds          metrics_interval_;
  std::atomic<bool>                  metrics_running_{false};
  std::thread                        metrics_thread_;
  std::condition_variable            metrics_cv_;
  mutable std::mutex                 metrics_thread_mutex_;

  std::vector<std::thread> threads_; ///< Threads for each section.

  holoflow_event::Router                                     router_;
  std::map<std::string, holoflow_event::Router::NodeHandles> event_handles_;
};

} // namespace holoflow::runtime
