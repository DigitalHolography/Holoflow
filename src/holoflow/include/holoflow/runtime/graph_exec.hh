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
#include <functional>
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
  std::vector<curaii::CudaStream>           streams; ///< CUDA streams for each section.
  std::vector<std::unique_ptr<core::ITask>> tasks;   ///< Task instances for each node.
  std::map<int, core::Tensor>               tensors; ///< Allocated tensors by ID.
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
  Scheduler(const GraphPlan &graph, ExecResouces &res);

  void start();
  void request_stop();
  void wait();

  bool is_running() const;
  bool stop_requested() const;

private:
  void build_sections();
  void build_nodes_rts();

  void run_section(int section_id);

  void acquire_owned_inputs(GraphPlan::vertex_descriptor v);
  void release_owned_outputs(GraphPlan::vertex_descriptor v);

  void refresh_views_sync(GraphPlan::vertex_descriptor v);
  void refresh_views_async_cons(GraphPlan::vertex_descriptor v);
  void refresh_views_async_prod(GraphPlan::vertex_descriptor v);

  void run_sync(GraphPlan::vertex_descriptor v);
  void run_async_cons(GraphPlan::vertex_descriptor v);
  void run_async_prod(GraphPlan::vertex_descriptor v);

  void refresh_outputs_sync(GraphPlan::vertex_descriptor v);
  void refresh_outputs_async_cons(GraphPlan::vertex_descriptor v);
  void refresh_outputs_async_prod(GraphPlan::vertex_descriptor v);

private:
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_{false};
  const GraphPlan  &graph_;
  ExecResouces     &res_;

  std::vector<core::TView> tviews_;
  std::vector<Section>     sections_;
  std::vector<NodeRt>      node_rts_;
  std::vector<std::thread> threads_;
};

} // namespace holoflow::runtime