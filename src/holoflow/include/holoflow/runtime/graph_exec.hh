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
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <thread>
#include <variant>
#include <vector>

#include "curaii/cuda.hh"
#include "driver_types.h"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"

namespace holoflow::runtime {

struct NodePlan {
  int               id;       ///< Node ID.
  core::NodeSpec    spec;     ///< Node specification.
  core::InferResult infer;    ///< Inference metadata.
  std::vector<int>  in_tids;  ///< Input tensor IDs.
  std::vector<int>  out_tids; ///< Output tensor IDs.
  int               section;  ///< Execution section ID.
};

struct EdgePlan {
  core::EdgeSpec spec; ///< Edge specification.
  core::TDesc    desc; ///< Tensor descriptor.
  int            tid;  ///< Tensor ID.
};

using GraphPlan = boost::adjacency_list<boost::vecS,      // OutEdgeList
                                        boost::vecS,      // VertexList
                                        boost::directedS, // Directed graph
                                        NodePlan,         // Vertex properties
                                        EdgePlan          // Edge properties
                                        >;

struct ExecResouces {
  std::map<int, curaii::CudaStream>           streams; ///< CUDA streams for each section ID.
  std::map<int, std::unique_ptr<core::ITask>> tasks;   ///< Task instances for each node ID.
  std::map<int, core::Tensor>                 tensors; ///< Allocated tensors by ID.
};

struct Section {
  int                                       id;         ///< Section ID.
  std::string                               name;       ///< Section name (for logging).
  cudaStream_t                              stream;     ///< CUDA stream for this section.
  std::vector<GraphPlan::vertex_descriptor> sync_topo;  ///< Synchronous nodes in topological order.
  std::vector<GraphPlan::vertex_descriptor> async_cons; ///< Asynchronous consumer nodes.
  std::vector<GraphPlan::vertex_descriptor> async_prod; ///< Asynchronous producer nodes
};

struct SyncMetrics {
  std::size_t num_execs = 0;
  double      total_ms  = 0.0;
};

struct AsyncMetrics {
  std::size_t num_pushes    = 0;
  std::size_t num_pops      = 0;
  double      total_push_ms = 0.0;
  double      total_pop_ms  = 0.0;
};

struct SyncRt {
  core::ISyncTask         *task = nullptr;
  std::vector<core::TView> in_views;
  std::vector<core::TView> out_views;
  core::SyncCtx            ctx{};
  SyncMetrics              m{};
};

struct AsyncRt {
  core::IAsyncTask        *task = nullptr;
  std::vector<core::TView> in_views;
  std::vector<core::TView> out_views;
  core::AsyncPushCtx       pctx{};
  core::AsyncPopCtx        xctx{};
  AsyncMetrics             m{};
};

using NodeRt = std::variant<SyncRt, AsyncRt>;

class Scheduler {
public:
  Scheduler(const GraphPlan &graph, const std::vector<Section> &sections, ExecResouces &res);

  void start();
  void request_stop();
  void wait();

  bool is_running() const;
  bool stop_requested() const;

private:
  void build_nodes_rts();

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
  std::vector<std::thread> threads_;  ///< Threads for each section.
};

} // namespace holoflow::runtime