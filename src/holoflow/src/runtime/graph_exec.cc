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

#include "holoflow/runtime/graph_exec.hh"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <chrono>
#include <windows.h>
#include <winnt.h>

#include "boost/graph/properties.hpp"
#include "bug.hh"
#include "cuda_runtime_api.h"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "logger.hh"

namespace holoflow::runtime {

Scheduler::Scheduler(const GraphPlan &graph, ExecResouces &resources)
    : graph_(graph), res_(resources) {
  build_sections();
  build_nodes_rts();
}

void Scheduler::start() {
  logger()->info("Starting scheduler");
  if (running_.exchange(true)) {
    logger()->warn("Scheduler is already running");
    return;
  }

  stop_.store(false);
  threads_.reserve(sections_.size());
  for (size_t i = 0; i < sections_.size(); i++) {
    threads_.emplace_back(&Scheduler::run_section, this, static_cast<int>(i));
  }
}

void Scheduler::request_stop() {
  logger()->info("Stop requested");
  if (!running_.load()) {
    logger()->warn("Scheduler is not running");
    return;
  }
  if (stop_.exchange(true)) {
    logger()->warn("Stop already requested");
    return;
  }
}

void Scheduler::wait() {
  logger()->info("Waiting for scheduler to stop");
  for (auto &t : threads_) {
    auto tid = GetThreadId(static_cast<HANDLE>(t.native_handle()));
    logger()->info("Joining thread {}", tid);
    t.join();
  }

  threads_.clear();
  running_.store(false);
  // TODO: Is this really the best place to reset running_?
}

void Scheduler::run_section(int section_id) {
  const auto &sec    = sections_.at(section_id);
  auto        stream = sec.stream;

  // Nodes in sections are topologically sorted, so we can execute them in order.
  // However, owned inputs are used as outputs for former nodes, so we need to
  // acquire them first.
  // Owned outputs are used as inputs for later nodes, so we need to release
  // them last.

  // TODO: How to handle end of stream (Eof)? Do we need to propagate it?
  // Do we need to stop the scheduler when we reach Eof for every node?
  // Do we need to notify nodes when we reach Eof for their inputs?

  // TODO: How to handle stream synchronization? Should asynchronous tasks
  // be responsible for synchronizing push-stream before enabling to pop data?
  // Or should the scheduler be the sole responsible for synchronizing streams?

  // TODO: How to properly collect metrics on given tasks run on cuda streams?
  // Should we use cuda events?

  while (!stop_.load()) {
    logger()->trace("Running section {}", sec.name);

    // Acquire owned inputs.
    //
    // - We do not acquire owned-inputs of async consumers here, as they
    //   used in the former section only.
    //
    // - It is mandatory to check stop_ after acquiring inputs, as
    //   the scheduler may have been requested to stop while waiting
    //   for owned inputs to become available. This leads to undefined
    //   behavior if we proceed to execute nodes after stop_ was set.
    for (auto v : sec.sync_topo) {
      acquire_owned_inputs(v);
    }
    for (auto v : sec.async_prod) {
      acquire_owned_inputs(v);
    }
    if (stop_.load()) {
      break;
    }

    // Execute nodes.
    //
    // - We know the nodes are topologically sorted within the section, so we
    //   can execute them in order. The topological order also takes into account
    //   in-place operations, so inputs or siblings is not changed before they are
    //   executed.
    //
    // - It is mandatory to syncronize the stream before running async producers,
    //   as async consumers from the next section may depend on work done on this stream,
    //   and we have no guarantee that the async producer at the end of this section
    //   will synchronize the stream before pushing data (it may not be cuda-related).
    for (auto v : sec.async_cons) {
      refresh_views_async_cons(v);
      run_async_cons(v);
      refresh_outputs_async_cons(v);
    }

    for (auto v : sec.sync_topo) {
      refresh_views_sync(v);
      run_sync(v);
      refresh_outputs_sync(v);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (auto v : sec.async_prod) {
      refresh_views_async_prod(v);
      run_async_prod(v);
      refresh_outputs_async_prod(v);
    }

    // Release owned outputs.
    //
    // - We do not release owned-outputs of async producers here, as they
    // used only in the next section.
    for (auto v : sec.sync_topo) {
      release_owned_outputs(v);
    }
    for (auto v : sec.async_cons) {
      release_owned_outputs(v);
    }
  }
}

void Scheduler::acquire_owned_inputs(GraphPlan::vertex_descriptor v) {
  const auto  idx        = boost::get(boost::vertex_index, graph_, v);
  const auto &np         = graph_[v];
  auto       &nrt        = node_rts_.at(idx);
  const auto &owned_mask = np.infer.owned_inputs;
  auto       *task       = std::holds_alternative<SyncRt>(nrt)
                               ? static_cast<core::ITask *>(std::get<SyncRt>(nrt).task)
                               : static_cast<core::ITask *>(std::get<AsyncRt>(nrt).task);

  for (size_t i = 0; i < owned_mask.size(); i++) {
    if (!owned_mask.at(i))
      continue;

    std::optional<core::TView> tview = task->acquire_input(static_cast<int>(i));
    while (!tview.has_value()) {
      if (stop_.load())
        return;
      tview = task->acquire_input(static_cast<int>(i));
    }

    tviews_.at(np.in_tids.at(i)) = tview.value();
  }
}

void Scheduler::release_owned_outputs(GraphPlan::vertex_descriptor v) {
  const auto  idx        = boost::get(boost::vertex_index, graph_, v);
  const auto &np         = graph_[v];
  auto       &nrt        = node_rts_.at(idx);
  const auto &owned_mask = np.infer.owned_outputs;
  auto       *task       = std::holds_alternative<SyncRt>(nrt)
                               ? static_cast<core::ITask *>(std::get<SyncRt>(nrt).task)
                               : static_cast<core::ITask *>(std::get<AsyncRt>(nrt).task);

  for (size_t i = 0; i < owned_mask.size(); i++) {
    if (!owned_mask.at(i))
      continue;

    task->release_output(static_cast<int>(i));
    tviews_.at(np.out_tids.at(i)) = core::TView{nullptr, core::TDesc{}};
  }
}

void Scheduler::refresh_views_sync(GraphPlan::vertex_descriptor v) {
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);

  HOLOFLOW_CHECK(std::holds_alternative<SyncRt>(nrt),
                 "refresh_views_sync called on an asynchronous node");

  auto &srt = std::get<SyncRt>(nrt);
  for (size_t i = 0; i < np.in_tids.size(); i++) {
    srt.in_views.at(i) = tviews_.at(np.in_tids.at(i));
  }
  for (size_t i = 0; i < np.out_tids.size(); i++) {
    srt.out_views.at(i) = tviews_.at(np.out_tids.at(i));
  }
}

void Scheduler::refresh_views_async_cons(GraphPlan::vertex_descriptor v) {
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);

  HOLOFLOW_CHECK(std::holds_alternative<AsyncRt>(nrt),
                 "refresh_views_async_cons called on a synchronous node");

  auto &art = std::get<AsyncRt>(nrt);
  for (size_t i = 0; i < np.out_tids.size(); i++) {
    art.out_views.at(i) = tviews_.at(np.out_tids.at(i));
  }
}

void Scheduler::refresh_views_async_prod(GraphPlan::vertex_descriptor v) {
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);

  HOLOFLOW_CHECK(std::holds_alternative<AsyncRt>(nrt),
                 "refresh_views_async_prod called on a synchronous node");

  auto &art = std::get<AsyncRt>(nrt);
  for (size_t i = 0; i < np.in_tids.size(); i++) {
    art.in_views.at(i) = tviews_.at(np.in_tids.at(i));
  }
}

void Scheduler::run_sync(GraphPlan::vertex_descriptor v) {
  using clock     = std::chrono::high_resolution_clock;
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);

  HOLOFLOW_CHECK(std::holds_alternative<SyncRt>(nrt), "run_sync called on an asynchronous node");
  auto &srt = std::get<SyncRt>(nrt);

  logger()->trace("Executing node '{}'", np.spec.name);
  auto t0 = clock::now();
  auto r  = srt.task->execute(srt.ctx);
  auto t1 = clock::now();

  switch (r) {
  case core::OpResult::Cancelled:
    logger()->debug("Node '{}' execution cancelled", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::Eof:
    logger()->debug("Node '{}' reached end of stream", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::NotReady:
    logger()->error("The synchronous task '{}' returned NotReady, which is not allowed",
                    np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::Ok:
    // All good.
    break;
  }

  using msf = std::chrono::duration<double, std::milli>;
  srt.m.total_ms += std::chrono::duration_cast<msf>(t1 - t0).count();
  srt.m.num_execs++;
}

void Scheduler::run_async_cons(GraphPlan::vertex_descriptor v) { HOLOFLOW_UNIMPLEMENTED(); }

void Scheduler::run_async_prod(GraphPlan::vertex_descriptor v) { HOLOFLOW_UNIMPLEMENTED(); }

void Scheduler::refresh_outputs_sync(GraphPlan::vertex_descriptor v) { HOLOFLOW_UNIMPLEMENTED(); }

void Scheduler::refresh_outputs_async_cons(GraphPlan::vertex_descriptor v) {
  HOLOFLOW_UNIMPLEMENTED();
}

void Scheduler::refresh_outputs_async_prod(GraphPlan::vertex_descriptor v) {
  HOLOFLOW_UNIMPLEMENTED();
}
} // namespace holoflow::runtime