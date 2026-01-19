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

#define NOMINMAX

#include "holoflow/runtime/graph_exec.hh"

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <chrono>
#include <map>
#include <mutex>
#include <nvtx3/nvtx3.hpp>
#include <vector>
#include <windows.h>

#include "boost/graph/properties.hpp"
#include "boost/range/iterator_range_core.hpp"
#include "bug.hh"
#include "cuda_runtime_api.h"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "logger.hh"

namespace holoflow::runtime {
namespace {

inline std::string node_context_msg(const holoflow::runtime::GraphPlan             &g,
                                    holoflow::runtime::GraphPlan::vertex_descriptor v,
                                    std::string_view phase, int section_id,
                                    std::string_view section_name) {
  const auto &np       = g[v];
  const auto  vertex_i = static_cast<std::size_t>(boost::get(boost::vertex_index, g, v));
  const auto  tid      = ::GetCurrentThreadId();

  return std::format("Exception in node '{}'\n"
                     "  phase: {}\n"
                     "  section: {} (id={})\n"
                     "  vertex_idx: {}\n"
                     "  thread_id: {}\n",
                     np.spec.name, phase, section_name, section_id, vertex_i, tid);
}

[[noreturn]] inline void
rethrow_with_node_context(const holoflow::runtime::GraphPlan             &g,
                          holoflow::runtime::GraphPlan::vertex_descriptor v, std::string_view phase,
                          int section_id, std::string_view section_name) {
  try {
    throw; // rethrow current exception
  } catch (const std::exception &e) {
    throw std::runtime_error(node_context_msg(g, v, phase, section_id, section_name) +
                             std::string{"  what: "} + e.what());
  } catch (...) {
    throw std::runtime_error(node_context_msg(g, v, phase, section_id, section_name) +
                             "  what: <non-std exception>");
  }
}

} // namespace

Scheduler::Scheduler(const GraphPlan &graph, const std::vector<Section> &sections,
                     ExecResouces &resources, std::chrono::milliseconds metrics_interval)
    : graph_(graph), sections_(sections), res_(resources), metrics_interval_(metrics_interval) {
  // Count distinct tids
  int nb_tids = 0;
  for (const auto &v : boost::make_iterator_range(boost::vertices(graph))) {
    const auto &np   = graph[v];
    const auto  tids = std::array{std::span{np.in_tids}, std::span{np.out_tids}} | std::views::join;

    if (!tids.empty()) {
      int max = std::ranges::max(tids);
      nb_tids = std::max(nb_tids, max + 1);
    }
  }

  tviews_.resize(nb_tids);
  for (int tid = 0; tid < nb_tids; tid++) {
    if (resources.tensors.contains(tid)) {
      tviews_.at(tid) = resources.tensors.at(tid).view();
    }
  }

  if (metrics_interval_.count() <= 0) {
    metrics_interval_ = std::chrono::milliseconds{1};
  }

  build_event_handles();
  build_nodes_rts();
}

Scheduler::~Scheduler() {
  if (is_running()) {
    request_stop();
    wait();
  } else {
    stop_metrics_thread();
  }
}

void Scheduler::set_metrics_interval(std::chrono::milliseconds interval) {
  if (interval.count() <= 0) {
    interval = std::chrono::milliseconds{1};
  }
  {
    std::lock_guard<std::mutex> lock(metrics_thread_mutex_);
    metrics_interval_ = interval;
  }
  metrics_cv_.notify_all();
}

std::map<std::string, NodeMetrics> Scheduler::metrics() const {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  return latest_metrics_;
}

void Scheduler::start() {
  logger()->info("[Scheduler::start] Starting scheduler");
  if (running_.exchange(true)) {
    logger()->warn("[Scheduler::start] Scheduler is already running");
    return;
  }

  stop_.store(false);
  reset_metrics_state();
  start_metrics_thread();
  threads_.reserve(sections_.size());
  for (size_t i = 0; i < sections_.size(); i++) {
    threads_.emplace_back(&Scheduler::run_section, this, static_cast<int>(i));
  }

  threads_.emplace_back(&Scheduler::run_router, this);
}

void Scheduler::request_stop() {
  logger()->info("[Scheduler::request_stop] Requesting scheduler to stop");
  if (!running_.load()) {
    logger()->warn("[Scheduler::request_stop] Scheduler is not running");
    return;
  }
  if (stop_.exchange(true)) {
    logger()->warn("[Scheduler::request_stop] Stop already requested");
    return;
  }
}

void Scheduler::wait() {
  logger()->info("[Scheduler::wait] Waiting for scheduler to stop");
  for (auto &t : threads_) {
    auto tid = GetThreadId(static_cast<HANDLE>(t.native_handle()));
    logger()->debug("[Scheduler::wait] Joining thread {}...", tid);
    t.join();
    logger()->debug("[Scheduler::wait] Thread {} joined", tid);
  }

  threads_.clear();
  logger()->info("[Scheduler::wait] Scheduler stopped");
  running_.store(false);
  stop_metrics_thread();
  // TODO: Is this really the best place to reset running_?
}

bool Scheduler::is_running() const { return running_.load(); }

bool Scheduler::stop_requested() const { return stop_.load(); }

bool Scheduler::ui_try_send(const std::string &node_id, nlohmann::json &&data) noexcept {
  return router_.ui_try_send(node_id, std::move(data));
}

std::optional<holoflow_event::Event> Scheduler::ui_try_receive() noexcept {
  return router_.ui_try_receive();
}

void Scheduler::build_event_handles() {
  event_handles_.clear();
  for (auto v : boost::make_iterator_range(boost::vertices(graph_))) {
    const auto &np = graph_[v];
    event_handles_.emplace(np.spec.name, router_.bind_node(np.spec.name));
  }
}

void Scheduler::build_nodes_rts() {
  const auto num_vertices = boost::num_vertices(graph_);
  node_rts_.resize(num_vertices);
  node_names_.resize(num_vertices);
  metric_accumulators_.resize(num_vertices);

  for (auto v : boost::make_iterator_range(boost::vertices(graph_))) {
    const auto idx      = boost::get(boost::vertex_index, graph_, v);
    const auto np       = graph_[v];
    node_names_.at(idx) = np.spec.name;
    auto *task          = res_.tasks.at(np.spec.name).get();
    HOLOFLOW_CHECK(task != nullptr, "Task for node {} is null", np.spec.name);

    if (auto *st = dynamic_cast<core::ISyncTask *>(task)) {
      SyncRt srt;
      srt.task             = st;
      srt.in_views         = std::vector<core::TView>();
      srt.out_views        = std::vector<core::TView>();
      srt.ctx.inputs       = srt.in_views;
      srt.ctx.outputs      = srt.out_views;
      srt.ctx.cancelled    = &stop_;
      srt.ctx.event_writer = &event_handles_.at(np.spec.name).out;
      srt.ctx.event_reader = &event_handles_.at(np.spec.name).in;
      node_rts_.at(idx)    = std::move(srt);
    } else if (auto *at = dynamic_cast<core::IAsyncTask *>(task)) {
      AsyncRt art;
      art.task           = at;
      art.in_views       = std::vector<core::TView>();
      art.out_views      = std::vector<core::TView>();
      art.pctx.inputs    = art.in_views;
      art.pctx.cancelled = &stop_;
      art.xctx.outputs   = art.out_views;
      art.xctx.cancelled = &stop_;
      node_rts_.at(idx)  = std::move(art);
    } else {
      HOLOFLOW_BUG("Task for node {} is neither sync nor async", np.spec.name);
    }
  }
}

void Scheduler::run_router() {
  logger()->info("[Scheduler::run_router] Starting event router");
  while (!stop_.load()) {
    router_.tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  logger()->info("[Scheduler::run_router] Event router stopped");
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
    logger()->trace("[Scheduler::run_section] Running section {}", sec.name);
    nvtxRangePush(sec.name.c_str());

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
    if (stop_.load()) {
      break;
    }

    for (auto v : sec.sync_topo) {
      refresh_views_sync(v);
      try {
        run_sync(v);
      } catch (...) {
        stop_.store(true);
        rethrow_with_node_context(graph_, v, "sync/execute", section_id, sec.name);
      }
      refresh_outputs_sync(v);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (auto v : sec.async_prod) {
      refresh_views_async_prod(v);
      run_async_prod(v);
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
    nvtxRangePop();
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
    tviews_.at(np.out_tids.at(i)) = std::nullopt;
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
    HOLOFLOW_CHECK(tviews_.at(np.in_tids.at(i)).has_value(),
                   "Input tensor view {} for node {} is not available", i, np.spec.name);
    srt.in_views.at(i) = tviews_.at(np.in_tids.at(i)).value();
  }
  for (size_t i = 0; i < np.out_tids.size(); i++) {
    HOLOFLOW_CHECK(tviews_.at(np.out_tids.at(i)).has_value(),
                   "Output tensor view {} for node {} is not available", i, np.spec.name);
    srt.out_views.at(i) = tviews_.at(np.out_tids.at(i)).value();
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
    HOLOFLOW_CHECK(tviews_.at(np.out_tids.at(i)).has_value(),
                   "Output tensor view {} for node {} is not available", i, np.spec.name);
    art.out_views.at(i) = tviews_.at(np.out_tids.at(i)).value();
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
    HOLOFLOW_CHECK(tviews_.at(np.in_tids.at(i)).has_value(),
                   "Input tensor view {} for node {} is not available", i, np.spec.name);
    art.in_views.at(i) = tviews_.at(np.in_tids.at(i)).value();
  }
}

void Scheduler::run_sync(GraphPlan::vertex_descriptor v) {
  using clock     = std::chrono::high_resolution_clock;
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);

  auto is_sync = std::holds_alternative<SyncRt>(nrt);
  HOLOFLOW_CHECK(is_sync, "run_sync called on an asynchronous node");
  auto &srt = std::get<SyncRt>(nrt);

  logger()->trace("[Scheduler::run_sync] Executing node '{}'", np.spec.name);
  auto t0 = clock::now();
  auto r  = srt.task->execute(srt.ctx);
  auto t1 = clock::now();

  switch (r) {
  case core::OpResult::Cancelled:
    logger()->debug("[Scheduler::run_sync] Node '{}' execution cancelled", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::Eof:
    logger()->debug("[Scheduler::run_sync] Node '{}' reached end of stream", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::NotReady:
    logger()->error(
        "[Scheduler::run_sync] The synchronous task '{}' returned NotReady, which is not allowed",
        np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::Ok:
    // All good.
    break;
  }

  if (r == core::OpResult::Ok) {
    const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto [host_in, device_in] =
        sum_bytes(std::span<const core::TView>(srt.in_views.data(), srt.in_views.size()));
    auto [host_out, device_out] =
        sum_bytes(std::span<const core::TView>(srt.out_views.data(), srt.out_views.size()));
    record_node_sample(idx, static_cast<uint64_t>(duration_ns), host_in + host_out,
                       device_in + device_out);
  }
}

void Scheduler::run_async_cons(GraphPlan::vertex_descriptor v) {
  using clock     = std::chrono::high_resolution_clock;
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);

  auto is_async = std::holds_alternative<AsyncRt>(nrt);
  HOLOFLOW_CHECK(is_async, "run_async_cons called on a synchronous node");
  auto &art = std::get<AsyncRt>(nrt);

  logger()->trace("[Scheduler::run_async_cons] Executing node '{}'", np.spec.name);
  auto t0 = clock::now();
  auto r  = art.task->try_pop(art.xctx);
  while (r == core::OpResult::NotReady) {
    if (stop_.load())
      return;
    r = art.task->try_pop(art.xctx);
  }
  auto t1 = clock::now();

  switch (r) {
  case core::OpResult::Cancelled:
    logger()->debug("[Scheduler::run_async_cons] Node '{}' execution cancelled", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::Eof:
    logger()->debug("[Scheduler::run_async_cons] Node '{}' reached end of stream", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::NotReady:
    HOLOFLOW_UNREACHABLE();
    break;
  case core::OpResult::Ok:
    // All good.
    break;
  }

  if (r == core::OpResult::Ok) {
    const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto [host_out, device_out] =
        sum_bytes(std::span<const core::TView>(art.out_views.data(), art.out_views.size()));
    record_node_sample(idx, static_cast<uint64_t>(duration_ns), host_out, device_out);
  }
}

void Scheduler::run_async_prod(GraphPlan::vertex_descriptor v) {
  using clock     = std::chrono::high_resolution_clock;
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);

  auto is_async = std::holds_alternative<AsyncRt>(nrt);
  HOLOFLOW_CHECK(is_async, "run_async_prod called on a synchronous node");
  auto &art = std::get<AsyncRt>(nrt);

  logger()->trace("[Scheduler::run_async_prod] Executing node '{}'", np.spec.name);
  auto t0 = clock::now();
  auto r  = art.task->try_push(art.pctx);
  while (r == core::OpResult::NotReady) {
    if (stop_.load())
      return;
    r = art.task->try_push(art.pctx);
  }
  auto t1 = clock::now();

  switch (r) {
  case core::OpResult::Cancelled:
    logger()->debug("[Scheduler::run_async_prod] Node '{}' execution cancelled", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::Eof:
    logger()->debug("[Scheduler::run_async_prod] Node '{}' reached end of stream", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::NotReady:
    HOLOFLOW_UNREACHABLE();
    break;
  case core::OpResult::Ok:
    // All good.
    break;
  }

  if (r == core::OpResult::Ok) {
    const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto [host_in, device_in] =
        sum_bytes(std::span<const core::TView>(art.in_views.data(), art.in_views.size()));
    record_node_sample(idx, static_cast<uint64_t>(duration_ns), host_in, device_in);
  }
}

void Scheduler::refresh_outputs_sync(GraphPlan::vertex_descriptor v) {
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);

  auto is_sync = std::holds_alternative<SyncRt>(nrt);
  HOLOFLOW_CHECK(is_sync, "refresh_outputs_sync called on an asynchronous node");
  auto &srt = std::get<SyncRt>(nrt);

  for (size_t i = 0; i < np.out_tids.size(); i++) {
    if (np.infer.owned_outputs.at(i)) {
      tviews_.at(np.out_tids.at(i)) = srt.out_views.at(i);
    }
  }
}

void Scheduler::refresh_outputs_async_cons(GraphPlan::vertex_descriptor v) {
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);

  auto is_async = std::holds_alternative<AsyncRt>(nrt);
  HOLOFLOW_CHECK(is_async, "refresh_outputs_async_cons called on a synchronous node");
  auto &art = std::get<AsyncRt>(nrt);

  for (size_t i = 0; i < np.out_tids.size(); i++) {
    if (np.infer.owned_outputs.at(i)) {
      tviews_.at(np.out_tids.at(i)) = art.out_views.at(i);
    }
  }
}

void Scheduler::reset_metrics_state() {
  auto num_vertices = boost::num_vertices(graph_);
  metric_accumulators_.clear();
  metric_accumulators_.resize(num_vertices);

  std::lock_guard<std::mutex> lock(metrics_mutex_);
  latest_metrics_.clear();
  for (const auto &name : node_names_) {
    latest_metrics_.emplace(name, NodeMetrics{});
  }
}

void Scheduler::start_metrics_thread() {
  if (metrics_interval_.count() <= 0) {
    return;
  }
  bool expected = false;
  if (!metrics_running_.compare_exchange_strong(expected, true)) {
    return;
  }
  metrics_thread_ = std::thread(&Scheduler::metrics_loop, this);
}

void Scheduler::stop_metrics_thread() {
  bool expected = true;
  if (!metrics_running_.compare_exchange_strong(expected, false)) {
    return;
  }
  metrics_cv_.notify_all();
  if (metrics_thread_.joinable()) {
    metrics_thread_.join();
  }
  metrics_thread_ = std::thread();
}

void Scheduler::metrics_loop() {
  auto                         last = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(metrics_thread_mutex_);
  while (metrics_running_.load()) {
    auto interval = metrics_interval_;
    if (metrics_cv_.wait_for(lock, interval, [this] { return !metrics_running_.load(); })) {
      break;
    }
    auto now     = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last).count();
    last         = now;
    aggregate_metrics(elapsed);
  }
  auto now     = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration<double>(now - last).count();
  aggregate_metrics(elapsed);
}

void Scheduler::aggregate_metrics(double interval_seconds) {
  if (interval_seconds <= 0.0) {
    interval_seconds = static_cast<double>(metrics_interval_.count()) / 1000.0;
    if (interval_seconds <= 0.0) {
      interval_seconds = 1.0;
    }
  }

  std::map<std::string, NodeMetrics> snapshot;

  for (std::size_t idx = 0; idx < metric_accumulators_.size(); ++idx) {
    auto      &acc          = metric_accumulators_.at(idx);
    const auto duration_ns  = acc.duration_ns.exchange(0, std::memory_order_relaxed);
    const auto runs         = acc.run_count.exchange(0, std::memory_order_relaxed);
    const auto host_bytes   = acc.host_bytes.exchange(0, std::memory_order_relaxed);
    const auto device_bytes = acc.device_bytes.exchange(0, std::memory_order_relaxed);

    NodeMetrics metrics;
    metrics.sample_count = runs;
    if (runs > 0) {
      metrics.average_duration_ms =
          static_cast<double>(duration_ns) / static_cast<double>(runs) / 1'000'000.0;
    }
    if (interval_seconds > 0.0) {
      metrics.runs_per_second                  = static_cast<double>(runs) / interval_seconds;
      metrics.host_throughput_bytes_per_second = static_cast<double>(host_bytes) / interval_seconds;
      metrics.device_throughput_bytes_per_second =
          static_cast<double>(device_bytes) / interval_seconds;
    }
    snapshot.emplace(node_names_.at(idx), metrics);
  }

  std::lock_guard<std::mutex> lock(metrics_mutex_);
  latest_metrics_ = std::move(snapshot);
}

void Scheduler::record_node_sample(std::size_t idx, uint64_t duration_ns, uint64_t host_bytes,
                                   uint64_t device_bytes) {
  if (idx >= metric_accumulators_.size()) {
    return;
  }
  auto &acc = metric_accumulators_.at(idx);
  acc.duration_ns.fetch_add(duration_ns, std::memory_order_relaxed);
  acc.run_count.fetch_add(1, std::memory_order_relaxed);
  acc.host_bytes.fetch_add(host_bytes, std::memory_order_relaxed);
  acc.device_bytes.fetch_add(device_bytes, std::memory_order_relaxed);
}

std::pair<uint64_t, uint64_t> Scheduler::sum_bytes(std::span<const core::TView> views) {
  uint64_t host_total   = 0;
  uint64_t device_total = 0;
  for (const auto &view : views) {
    if (view.ptr == nullptr) {
      continue;
    }
    const auto bytes = static_cast<uint64_t>(view.desc.num_bytes());
    if (view.desc.mem_loc == core::MemLoc::Device) {
      device_total += bytes;
    } else {
      host_total += bytes;
    }
  }
  return {host_total, device_total};
}

} // namespace holoflow::runtime
