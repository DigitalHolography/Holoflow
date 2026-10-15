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
#include "holoflow/runtime/tracing.hh"

#include <algorithm>
#include <array>
#include <atomic>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <format>
#include <map>
#include <mutex>
#include <nvtx3/nvtx3.hpp>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
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
#include "section_cuda_graph.hh"

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

size_t count_distinct_tids(const GraphPlan &graph) {
  std::set<int> tids;
  for (const auto &v : boost::make_iterator_range(boost::vertices(graph))) {
    const auto &np   = graph[v];
    const auto tids_ = std::array{std::span{np.in_tids}, std::span{np.out_tids}} | std::views::join;

    for (const auto tid : tids_) {
      tids.insert(tid);
    }
  }
  return tids.size();
}

std::string task_range_name(std::string_view operation, const core::NodeSpec &spec) {
  return std::format("{}: {} ({})", operation, spec.name, spec.kind);
}

// Dispatch steps do not correspond to section iterations. Keep these thread-bound scopes alive
// across dispatches, but never across a pause: resumed checkpoints run on a new worker thread.
class SectionNvtxRanges {
public:
  void begin_iteration(const char *name) {
    if (!iteration_)
      iteration_.emplace(name);
  }

  void phase(const char *name) {
    if (phase_name_ == name)
      return;
    end_phase();
    phase_.emplace(name);
    phase_name_ = name;
  }

  void end_phase() {
    phase_.reset();
    phase_name_ = {};
  }

  void end_iteration() {
    end_phase();
    iteration_.reset();
  }

private:
  // Reverse destruction order closes the phase before its parent, including during unwinding.
  std::optional<nvtx3::scoped_range> iteration_;
  std::optional<nvtx3::scoped_range> phase_;
  std::string_view                   phase_name_;
};

[[noreturn]] inline void
log_and_abort_current_exception(std::string_view thread_name, std::string_view node_name,
                                const Scheduler::FailureCallback &failure_callback,
                                std::atomic_flag                 &failure_dumped) {
  std::string error_message;
  try {
    throw;
  } catch (const std::exception &e) {
    error_message = e.what();
    logger()->critical("[{}] Fatal runtime exception:\n{}", thread_name, e.what());
  } catch (...) {
    error_message = "<non-std exception>";
    logger()->critical("[{}] Fatal runtime exception: <non-std exception>", thread_name);
  }

  if (failure_callback && !failure_dumped.test_and_set(std::memory_order_relaxed)) {
    try {
      failure_callback(thread_name, node_name, error_message);
    } catch (const std::exception &e) {
      logger()->error("Failed to write runtime failure graph: {}", e.what());
    } catch (...) {
      logger()->error("Failed to write runtime failure graph: unknown error");
    }
  }

  logger()->flush();
  std::abort();
}

} // namespace

void *MemoryBlock::get() {
  void *ptr = nullptr;

  switch (mem_loc) {
  case core::MemLoc::Host:
    ptr = h_data.get();
    break;
  case core::MemLoc::Device:
    ptr = d_data.get();
    break;
  }

  HOLOFLOW_CHECK(ptr != nullptr, "MemoryBlock pointer is null");
  return ptr;
}

class Scheduler::Impl {
public:
  Impl(const GraphPlan &graph, const std::vector<Section> &sections, ExecResouces &res,
       std::chrono::milliseconds metrics_interval, Scheduler::FailureCallback failure_callback);

  ~Impl();

  void set_metrics_interval(std::chrono::milliseconds interval);
  [[nodiscard]] std::map<std::string, NodeMetrics> metrics() const;
  /// Host submission timing/counts for section graph launches, not GPU execution duration.
  [[nodiscard]] std::map<std::string, NodeMetrics> section_graph_metrics() const;
  [[nodiscard]] nlohmann::json                     section_graph_diagnostics() const;

  void start();
  /// Suspend with acquired buffers and completed operations retained for start().
  void request_pause();
  void request_stop();
  void wait();

  bool                           is_running() const;
  bool                           stop_requested() const;
  [[nodiscard]] std::string_view current_node_name() const noexcept;

  [[nodiscard]] bool ui_try_send(const std::string &node_id, nlohmann::json &&data) noexcept;

  [[nodiscard]] std::optional<holoflow_event::Event> ui_try_receive() noexcept;

private:
  struct SyncRt {
    core::ISyncTask         *task = nullptr;
    std::string              range_name;
    std::vector<core::TView> in_views;
    std::vector<core::TView> out_views;
    core::SyncCtx            ctx{};
  };

  struct AsyncRt {
    core::IAsyncTask        *task = nullptr;
    std::string              pop_range_name;
    std::string              push_range_name;
    std::vector<core::TView> in_views;
    std::vector<core::TView> out_views;
    core::AsyncPushCtx       pctx{};
    core::AsyncPopCtx        xctx{};
  };

  using NodeRt = std::variant<SyncRt, AsyncRt>;

  void init_tviews();
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
  bool run_prequeued_section(int section_id);
  void retire_prequeued(int section_id);

  /// This function acquires all owned inputs for the given node.
  /// Owning tasks publish memory by updating their compiler-provided Storage;
  /// all scheduler TViews retain pointers to that stable Storage object.
  /// Waits for each owned input to become available, unless interrupted by a stop request.
  /// @param[in,out] port Next input port to process; starts at 0 and skips non-owned ports.
  /// Preserved on interruption to avoid duplicate acquisition. Equals owned_inputs.size() on
  /// completion; reset to 0 for the next node.
  /// @warning This function must be called on a synchronous or asynchronous
  /// producer node only.
  void acquire_owned_inputs(GraphPlan::vertex_descriptor v, size_t &port);

  /// This function releases all owned outputs for the given node.
  /// Pointer cleanup remains the owning task's responsibility.
  /// This function does not block; a pause request may interrupt it between releases.
  /// @param[in,out] port Next output port to process; starts at 0 and skips non-owned ports.
  /// Preserved on pause to avoid duplicate release. Equals owned_outputs.size() on completion;
  /// reset to 0 for the next node.
  void release_owned_outputs(GraphPlan::vertex_descriptor v, size_t &port);

  /// Executes a synchronous node.
  /// @warning This function must be called on a synchronous node only.
  [[nodiscard]] core::OpResult run_sync(GraphPlan::vertex_descriptor v);

  /// Executes an asynchronous consumer node.
  /// @warning This function must be called on an asynchronous consumer node only.
  [[nodiscard]] core::OpResult run_async_cons(GraphPlan::vertex_descriptor v);

  /// Executes an asynchronous producer node.
  /// @warning This function must be called on an asynchronous producer node only.
  [[nodiscard]] core::OpResult run_async_prod(GraphPlan::vertex_descriptor v);

private:
  std::atomic<bool> running_{false}; ///< True if the scheduler is running.
  std::atomic<bool> stop_{false};    ///< True if a stop has been requested.
  std::atomic<bool> paused_{false};
  std::atomic<bool> aborted_{false};
  struct SectionCheckpoint {
    enum class Stage { AcquireSync, AcquireProducer, Pop, Compute, Barrier, Push, Release };
    Stage                                     stage               = Stage::AcquireSync;
    size_t                                    node                = 0;
    size_t                                    port                = 0;
    bool                                      computation_started = false;
    bool                                      graph_submitted     = false;
    std::vector<GraphPlan::vertex_descriptor> produced;
  };
  std::vector<SectionCheckpoint> checkpoints_;
  struct Reservation {
    GraphPlan::vertex_descriptor node;
    core::AsyncTicket            ticket;
    uint64_t                     duration_ns = 0;
  };
  struct InFlight {
    InFlight() { CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming)); }
    ~InFlight() { CUDA_CHECK_NT(cudaEventDestroy(done)); }
    InFlight(const InFlight &)                           = delete;
    InFlight                &operator=(const InFlight &) = delete;
    cudaEvent_t              done                        = nullptr;
    std::vector<Reservation> pushes, pops;
    // Storage is shared by task contexts. Save addresses, rather than copying mutable TViews.
    std::vector<std::pair<core::Storage *, std::byte *>> bindings;
  };
  struct PrequeueState {
    std::array<InFlight, 2> slots;
    size_t                  head = 0, size = 0;
  };
  std::vector<std::unique_ptr<PrequeueState>> prequeues_;
  const GraphPlan                            &graph_;    ///< The computational graph to execute.
  const std::vector<Section>                 &sections_; ///< Execution sections.
  ExecResouces &res_; ///< Execution resources (streams, tasks, tensors).

  /// Stable TViews for all tensors by their IDs. Copies across node contexts
  /// observe ownership changes through their shared Storage pointers.
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
  std::vector<NodeMetricAccumulator> graph_metric_accumulators_;

  mutable std::mutex                 metrics_mutex_;
  std::map<std::string, NodeMetrics> latest_metrics_;
  std::map<std::string, NodeMetrics> latest_section_graph_metrics_;
  std::chrono::milliseconds          metrics_interval_;
  std::atomic<bool>                  metrics_running_{false};
  std::thread                        metrics_thread_;
  std::condition_variable            metrics_cv_;
  mutable std::mutex                 metrics_thread_mutex_;

  std::vector<std::thread> threads_; ///< Threads for each section.

  holoflow_event::Router                                     router_;
  std::map<std::string, holoflow_event::Router::NodeHandles> event_handles_;
  std::atomic_flag                                           failure_dumped_ = ATOMIC_FLAG_INIT;
  Scheduler::FailureCallback                                 failure_callback_;
};

Scheduler::Impl::Impl(const GraphPlan &graph, const std::vector<Section> &sections,
                      ExecResouces &resources, std::chrono::milliseconds metrics_interval,
                      Scheduler::FailureCallback failure_callback)
    : graph_(graph), sections_(sections), res_(resources), metrics_interval_(metrics_interval),
      failure_callback_(std::move(failure_callback)) {
  tracing::ScopedTrace trace("Initialize Scheduler", "scheduler");

  if (failure_callback_) {
    detail::set_bug_callback([this](std::string_view file, std::size_t line) {
      if (!failure_dumped_.test_and_set(std::memory_order_relaxed)) {
        failure_callback_("HOLOFLOW_BUG", detail::current_node_name(),
                          std::format("Fatal bug at {}:{}", file, line));
      }
    });
  } else {
    detail::set_bug_callback({});
  }

  if (metrics_interval_.count() <= 0) {
    metrics_interval_ = std::chrono::milliseconds{1};
  }

  // init_tensor_tables();
  // bind_resource_tensors();
  {
    tracing::ScopedTrace scope("Initialize Tensor Views", "scheduler");
    init_tviews();
  }
  {
    tracing::ScopedTrace scope("Build Event Handles", "scheduler");
    build_event_handles();
  }
  {
    tracing::ScopedTrace scope("Build Node Runtime State", "scheduler");
    build_nodes_rts();
  }
  prequeues_.resize(sections_.size());
}

Scheduler::Impl::~Impl() {
  if (is_running() || paused_.load()) {
    request_stop();
    wait();
  } else {
    stop_metrics_thread();
  }
  detail::set_bug_callback({});
}

void Scheduler::Impl::set_metrics_interval(std::chrono::milliseconds interval) {
  if (interval.count() <= 0) {
    interval = std::chrono::milliseconds{1};
  }
  {
    std::lock_guard<std::mutex> lock(metrics_thread_mutex_);
    metrics_interval_ = interval;
  }
  metrics_cv_.notify_all();
}

std::map<std::string, NodeMetrics> Scheduler::Impl::metrics() const {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  return latest_metrics_;
}

std::map<std::string, NodeMetrics> Scheduler::Impl::section_graph_metrics() const {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  return latest_section_graph_metrics_;
}

nlohmann::json Scheduler::Impl::section_graph_diagnostics() const {
  auto result = nlohmann::json::array();
  for (const auto &[id, graphs] : res_.section_cuda_graphs)
    result.push_back(graphs->snapshot());
  return result;
}

void Scheduler::Impl::start() {
  tracing::ScopedTrace trace("Scheduler Start", "scheduler");
  logger()->info("[Scheduler::start] Starting scheduler");
  if (running_.exchange(true)) {
    logger()->warn("[Scheduler::start] Scheduler is already running");
    return;
  }

  const bool resume = paused_.exchange(false) && !aborted_.load();
  aborted_.store(false);
  stop_.store(false);
  if (!resume) {
    checkpoints_.assign(sections_.size(), SectionCheckpoint{});
  }
  try {
    {
      tracing::ScopedTrace drain("Drain Startup CUDA Streams", "scheduler");
      for (const auto &[id, stream] : res_.streams) {
        tracing::ScopedTrace stream_trace(std::format("Synchronize Startup Stream {}", id),
                                          "detail");
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
      }
    }
    if (!resume) {
      tracing::ScopedTrace graphs("Prepare Startup CUDA Graphs", "scheduler");
      refresh_section_cuda_graphs(graph_, sections_, res_, true);
    }
    for (const auto &section : sections_) {
      auto it = res_.section_cuda_graphs.find(section.id);
      if (it == res_.section_cuda_graphs.end())
        continue;
      auto      &graphs   = *it->second;
      const bool eligible = section.supports_prequeue && graphs.enabled;
      graphs.prequeue_depth.store(eligible ? 2 : 1, std::memory_order_relaxed);
      std::lock_guard lock(graphs.diagnostic_mutex);
      graphs.diagnostics["prequeue_fallback_reason"] = eligible ? ""
                                                       : section.supports_prequeue
                                                           ? "CUDA graphs unavailable"
                                                           : section.prequeue_fallback_reason;
    }
    // Graph preparation writes before eligibility is selected. Publish the actual scheduling
    // decision as well, including on resume, before any worker can update runtime counters.
    write_section_cuda_graph_diagnostics(res_);
  } catch (...) {
    running_.store(false);
    throw;
  }
  {
    tracing::ScopedTrace scope("Reset Scheduler Metrics", "scheduler");
    reset_metrics_state();
  }
  {
    tracing::ScopedTrace scope("Start Metrics Thread", "scheduler");
    start_metrics_thread();
  }
  tracing::ScopedTrace workers("Create Scheduler Workers", "scheduler");
  threads_.reserve(sections_.size());
  for (size_t i = 0; i < sections_.size(); i++) {
    threads_.emplace_back(&Scheduler::Impl::run_section, this, static_cast<int>(i));
  }

  threads_.emplace_back(&Scheduler::Impl::run_router, this);
}

void Scheduler::Impl::request_pause() {
  if (!running_.load() || stop_.load())
    return;
  paused_.store(true);
  stop_.store(true);
}

void Scheduler::Impl::request_stop() {
  tracing::ScopedTrace trace("Scheduler Request Stop", "scheduler");
  logger()->info("[Scheduler::request_stop] Requesting scheduler to stop");
  aborted_.store(true);
  const bool was_paused = paused_.exchange(false);
  if (!running_.load()) {
    if (!was_paused)
      logger()->warn("[Scheduler::request_stop] Scheduler is not running");
    return;
  }
  if (stop_.exchange(true)) {
    logger()->warn("[Scheduler::request_stop] Stop already requested");
    return;
  }
}

void Scheduler::Impl::wait() {
  tracing::ScopedTrace trace("Scheduler Wait", "scheduler");
  logger()->info("[Scheduler::wait] Waiting for scheduler to stop");
  for (auto &t : threads_) {
    auto tid = GetThreadId(static_cast<HANDLE>(t.native_handle()));
    logger()->debug("[Scheduler::wait] Joining thread {}...", tid);
    {
      tracing::ScopedTrace join(std::format("Join Worker {}", tid), "detail");
      t.join();
    }
    logger()->debug("[Scheduler::wait] Thread {} joined", tid);
  }

  threads_.clear();
  {
    tracing::ScopedTrace drain("Drain Stopped CUDA Streams", "scheduler");
    for (const auto &[id, stream] : res_.streams)
      CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  }
  // A pause may already have joined its workers. Abort/destruction must still return held
  // outputs (including camera DMA buffers) before compatible tasks migrate to a new compilation.
  if (aborted_.load()) {
    for (size_t i = 0; i < prequeues_.size(); ++i)
      while (prequeues_[i] && prequeues_[i]->size)
        retire_prequeued(static_cast<int>(i));
    for (auto &checkpoint : checkpoints_) {
      const bool   releasing = checkpoint.stage == SectionCheckpoint::Stage::Release;
      const size_t first     = releasing ? checkpoint.node : 0;
      for (size_t i = first; i < checkpoint.produced.size(); ++i) {
        size_t port = releasing && i == first ? checkpoint.port : 0;
        release_owned_outputs(checkpoint.produced[i], port);
      }
      checkpoint = SectionCheckpoint{};
    }
  }
  logger()->info("[Scheduler::wait] Scheduler stopped");
  running_.store(false);
  {
    tracing::ScopedTrace metrics("Stop Metrics Thread", "scheduler");
    stop_metrics_thread();
  }
  tracing::ScopedTrace diagnostics("Submit Shutdown CUDA Graph Diagnostics", "scheduler");
  write_section_cuda_graph_diagnostics(res_);
  for (const auto &[id, graphs] : res_.section_cuda_graphs) {
    const auto report = graphs->snapshot();
    logger()->info("[CUDA graphs] {} shutdown: {} launches, {} ordinary iterations, {} pointer "
                   "misses, {} tuple misses",
                   report["section"].get<std::string>(), report["launches"].dump(),
                   report["ordinary_iterations"].dump(), report["pointer_misses"].dump(),
                   report["tuple_misses"].dump());
  }
  // TODO: Is this really the best place to reset running_?
}

bool Scheduler::Impl::is_running() const { return running_.load(); }

bool Scheduler::Impl::stop_requested() const { return stop_.load(); }

std::string_view Scheduler::Impl::current_node_name() const noexcept {
  return detail::current_node_name();
}

bool Scheduler::Impl::ui_try_send(const std::string &node_id, nlohmann::json &&data) noexcept {
  return router_.ui_try_send(node_id, std::move(data));
}

std::optional<holoflow_event::Event> Scheduler::Impl::ui_try_receive() noexcept {
  return router_.ui_try_receive();
}

void Scheduler::Impl::init_tviews() {
  auto nb_tids = count_distinct_tids(graph_);
  tviews_.assign(nb_tids, core::TView{});

  for (auto &[tid, tdesc] : res_.tensor_descs) {
    auto sid        = res_.tid_to_sid.at(tid);
    auto storage    = res_.storages.at(sid).get();
    tviews_.at(tid) = core::TView{tdesc, storage};
  }
}

// void Scheduler::init_tensor_tables() {
//   auto nb_tids = count_distinct_tids(graph_);
//   storages_.assign(nb_tids, core::Storage{});
//   tviews_.assign(nb_tids, core::TView{});
// }

// void Scheduler::bind_resource_tensors() {
// for (auto &[tid, tensor] : res_.tensors) {
//   auto v = tensor.view();

//   storages_.at(tid) = *v.storage;
//   tviews_.at(tid)   = core::TView{v.desc, &storages_.at(tid)};
// }
// }

void Scheduler::Impl::build_event_handles() {
  event_handles_.clear();
  for (auto v : boost::make_iterator_range(boost::vertices(graph_))) {
    const auto &np = graph_[v];
    event_handles_.emplace(np.spec.name, router_.bind_node(np.spec.name));
  }
}

void Scheduler::Impl::build_nodes_rts() {
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

    // Build in_views and out_views
    std::vector<core::TView> in_views(np.in_tids.size());
    std::vector<core::TView> out_views(np.out_tids.size());

    for (size_t i = 0; i < np.in_tids.size(); ++i) {
      int tid     = np.in_tids[i];
      in_views[i] = tviews_.at(tid);
    }

    for (size_t i = 0; i < np.out_tids.size(); ++i) {
      int tid      = np.out_tids[i];
      out_views[i] = tviews_.at(tid);
    }

    if (auto *st = dynamic_cast<core::ISyncTask *>(task)) {
      SyncRt srt;
      srt.range_name       = task_range_name("sync execute", np.spec);
      srt.task             = st;
      srt.in_views         = in_views;
      srt.out_views        = out_views;
      srt.ctx.inputs       = srt.in_views;
      srt.ctx.outputs      = srt.out_views;
      srt.ctx.cancelled    = &stop_;
      srt.ctx.event_writer = &event_handles_.at(np.spec.name).out;
      srt.ctx.event_reader = &event_handles_.at(np.spec.name).in;
      node_rts_.at(idx)    = std::move(srt);
    } else if (auto *at = dynamic_cast<core::IAsyncTask *>(task)) {
      AsyncRt art;
      art.pop_range_name  = task_range_name("async pop", np.spec);
      art.push_range_name = task_range_name("async push", np.spec);
      art.task            = at;
      art.in_views        = in_views;
      art.out_views       = out_views;
      art.pctx.inputs     = art.in_views;
      art.pctx.cancelled  = &stop_;
      art.xctx.outputs    = art.out_views;
      art.xctx.cancelled  = &stop_;
      node_rts_.at(idx)   = std::move(art);
    } else {
      HOLOFLOW_BUG("Task for node {} is neither sync nor async", np.spec.name);
    }
  }
}

void Scheduler::Impl::run_router() {
  tracing::set_thread_name("Scheduler Router");
  try {
    logger()->info("[Scheduler::run_router] Starting event router");
    while (!stop_.load()) {
      router_.tick();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    logger()->info("[Scheduler::run_router] Event router stopped");
  } catch (...) {
    stop_.store(true);
    log_and_abort_current_exception("Scheduler::run_router", {}, failure_callback_,
                                    failure_dumped_);
  }
}

// -------------------------------------------------------------------------------------------------
// Double-buffered section execution
// -------------------------------------------------------------------------------------------------

void Scheduler::Impl::retire_prequeued(int section_id) {
  auto &state = *prequeues_.at(section_id);
  auto &slot  = state.slots[state.head];
  {
    nvtx3::scoped_range wait_range{"Wait iteration event"};
    const auto          status = cudaEventQuery(slot.done);
    if (status == cudaErrorNotReady)
      CUDA_CHECK(cudaEventSynchronize(slot.done));
    else
      CUDA_CHECK(status);
  }
  for (const auto &reservation : slot.pushes) {
    detail::current_node_name() = graph_[reservation.node].spec.name;
    std::get<AsyncRt>(node_rts_.at(reservation.node)).task->commit_push(reservation.ticket);
  }
  for (const auto &reservation : slot.pops) {
    detail::current_node_name() = graph_[reservation.node].spec.name;
    std::get<AsyncRt>(node_rts_.at(reservation.node)).task->release_pop(reservation.ticket);
  }
  slot.pushes.clear();
  slot.pops.clear();
  slot.bindings.clear();
  state.head = (state.head + 1) % state.slots.size();
  --state.size;
  res_.section_cuda_graphs.at(section_id)
      ->prequeue_retirements.fetch_add(1, std::memory_order_relaxed);
  detail::current_node_name() = {};
}

bool Scheduler::Impl::run_prequeued_section(int section_id) {
  const auto &section  = sections_.at(section_id);
  auto        graph_it = res_.section_cuda_graphs.find(section.id);
  if (!section.supports_prequeue || graph_it == res_.section_cuda_graphs.end() ||
      !graph_it->second->enabled)
    return false;
  auto &graphs = *graph_it->second;
  auto &owner  = prequeues_.at(section_id);
  if (!owner) {
    owner = std::make_unique<PrequeueState>();
    for (auto &slot : owner->slots) {
      slot.pushes.reserve(section.async_prod.size());
      slot.pops.reserve(section.async_cons.size());
      slot.bindings.reserve(section.async_prod.size() + section.async_cons.size());
    }
  }
  auto      &state    = *owner;
  const auto rollback = [&](InFlight &slot) {
    for (auto it = slot.pops.rbegin(); it != slot.pops.rend(); ++it)
      std::get<AsyncRt>(node_rts_.at(it->node)).task->cancel_pop(it->ticket);
    for (auto it = slot.pushes.rbegin(); it != slot.pushes.rend(); ++it)
      std::get<AsyncRt>(node_rts_.at(it->node)).task->cancel_push(it->ticket);
    slot.pushes.clear();
    slot.pops.clear();
    slot.bindings.clear();
  };
  const auto control = [&](core::OpResult result) {
    if (result == core::OpResult::Eof) {
      aborted_.store(true);
      paused_.store(false);
      stop_.store(true);
    } else if (result == core::OpResult::Cancelled) {
      if (!paused_.load())
        aborted_.store(true);
      stop_.store(true);
    }
    return result == core::OpResult::Ok;
  };
  while (!stop_.load()) {
    if (state.size == state.slots.size()) {
      retire_prequeued(section_id);
      continue;
    }
    auto &slot  = state.slots[(state.head + state.size) % state.slots.size()];
    bool  ready = true;
    {
      nvtx3::scoped_range reserve_range{"Reserve iteration"};
      for (auto v : section.async_prod) {
        detail::current_node_name() = graph_[v].spec.name;
        auto             &runtime   = std::get<AsyncRt>(node_rts_.at(v));
        core::AsyncTicket ticket;
        if (!control(runtime.task->reserve_push(runtime.pctx, ticket))) {
          ready = false;
          break;
        }
        slot.pushes.push_back({v, ticket});
        for (size_t port = 0; port < runtime.in_views.size(); ++port)
          if (graph_[v].infer.owned_inputs[port]) {
            auto *storage = runtime.in_views[port].storage;
            slot.bindings.emplace_back(storage, storage->ptr);
          }
      }
      if (ready)
        for (auto v : section.async_cons) {
          detail::current_node_name() = graph_[v].spec.name;
          auto             &runtime   = std::get<AsyncRt>(node_rts_.at(v));
          core::AsyncTicket ticket;
          const auto        pop_started = std::chrono::steady_clock::now();
          if (!control(runtime.task->reserve_pop(runtime.xctx, ticket))) {
            ready = false;
            break;
          }
          slot.pops.push_back(
              {v, ticket,
               static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - pop_started)
                                         .count())});
          for (size_t port = 0; port < runtime.out_views.size(); ++port)
            if (graph_[v].infer.owned_outputs[port]) {
              auto *storage = runtime.out_views[port].storage;
              slot.bindings.emplace_back(storage, storage->ptr);
            }
        }
    }
    if (!ready || stop_.load()) {
      rollback(slot);
      if (!stop_.load() && state.size)
        retire_prequeued(section_id);
      else if (!stop_.load())
        std::this_thread::yield();
      continue;
    }
    detail::current_node_name() = {};
    for (const auto &[storage, pointer] : slot.bindings)
      storage->ptr = pointer;
    const auto variant = graphs.variant(res_.storages);
    if (!variant) {
      graphs.report_miss(res_.storages);
      rollback(slot);
      while (state.size)
        retire_prequeued(section_id);
      graphs.prequeue_depth.store(1, std::memory_order_relaxed);
      {
        std::lock_guard lock(graphs.diagnostic_mutex);
        graphs.diagnostics["prequeue_fallback_reason"] = "Runtime graph binding miss";
      }
      return false;
    }
    const auto started = std::chrono::steady_clock::now();
    {
      nvtx3::scoped_range launch_range{"CUDA graph launch"};
      CUDA_CHECK(cudaGraphLaunch(graphs.executables.at(*variant), section.stream));
    }
    auto &acc = graph_metric_accumulators_.at(section_id);
    acc.duration_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - started)
                                  .count()),
        std::memory_order_relaxed);
    acc.run_count.fetch_add(1, std::memory_order_relaxed);
    graphs.launches.fetch_add(1, std::memory_order_relaxed);
    for (const auto &reservation : slot.pushes) {
      detail::current_node_name() = graph_[reservation.node].spec.name;
      auto      &runtime          = std::get<AsyncRt>(node_rts_.at(reservation.node));
      const auto push_started     = std::chrono::steady_clock::now();
      runtime.task->enqueue_push(reservation.ticket);
      const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - push_started)
                                .count();
      auto [host, device] = sum_bytes(runtime.in_views);
      record_node_sample(reservation.node, static_cast<uint64_t>(duration), host, device);
    }
    CUDA_CHECK(cudaEventRecord(slot.done, section.stream));
    for (const auto &reservation : slot.pops) {
      auto &runtime       = std::get<AsyncRt>(node_rts_.at(reservation.node));
      auto [host, device] = sum_bytes(runtime.out_views);
      record_node_sample(reservation.node, reservation.duration_ns, host, device);
    }
    ++state.size;
    if (state.size > graphs.prequeue_peak.load(std::memory_order_relaxed))
      graphs.prequeue_peak.store(state.size, std::memory_order_relaxed);
    graphs.prequeue_submissions.fetch_add(1, std::memory_order_relaxed);
    detail::current_node_name() = {};
  }
  // Pause retains both tickets and events. Abort/EOF cleanup runs in wait(), after all workers
  // have joined and all streams are drained, including when stopping an already paused scheduler.
  return true;
}

void Scheduler::Impl::run_section(int section_id) {
  const auto &sec = sections_.at(section_id);
  tracing::set_thread_name(std::format("Section {}: {}", section_id, sec.name));
  auto &checkpoint = checkpoints_.at(section_id);
  using Stage      = SectionCheckpoint::Stage;
  std::string_view current_node_name;
  detail::current_node_name() = {};
  try {
    if (run_prequeued_section(section_id))
      return;
    SectionNvtxRanges ranges;
    const auto        advance = [&](Stage next) {
      if (!(checkpoint.stage == Stage::AcquireSync && next == Stage::AcquireProducer))
        ranges.end_phase();
      checkpoint.stage = next;
      checkpoint.node  = 0;
      checkpoint.port  = 0;
    };
    while (!stop_.load()) {
      current_node_name           = {};
      detail::current_node_name() = {};
      ranges.begin_iteration(sec.name.c_str());
      switch (checkpoint.stage) {
      case Stage::AcquireSync:
      case Stage::AcquireProducer: {
        const auto &nodes = checkpoint.stage == Stage::AcquireSync ? sec.sync_topo : sec.async_prod;
        if (checkpoint.node == nodes.size()) {
          advance(checkpoint.stage == Stage::AcquireSync ? Stage::AcquireProducer : Stage::Pop);
          break;
        }
        const auto v                = nodes[checkpoint.node];
        current_node_name           = graph_[v].spec.name;
        detail::current_node_name() = current_node_name;
        try {
          if (std::ranges::any_of(graph_[v].infer.owned_inputs, [](bool owned) { return owned; }))
            ranges.phase("Acquire inputs");
          acquire_owned_inputs(v, checkpoint.port);
        } catch (...) {
          rethrow_with_node_context(graph_, v, "acquire_owned_inputs", section_id, sec.name);
        }
        if (checkpoint.port == graph_[v].infer.owned_inputs.size()) {
          ++checkpoint.node;
          checkpoint.port = 0;
        }
        break;
      }
      case Stage::Pop:
        if (checkpoint.node == sec.async_cons.size()) {
          advance(Stage::Compute);
          break;
        }
        {
          const auto v                = sec.async_cons[checkpoint.node];
          current_node_name           = graph_[v].spec.name;
          detail::current_node_name() = current_node_name;
          try {
            ranges.phase("Async pop");
            if (run_async_cons(v) == core::OpResult::Ok) {
              checkpoint.produced.push_back(v);
              ++checkpoint.node;
            }
          } catch (...) {
            rethrow_with_node_context(graph_, v, "execute_async_consumer", section_id, sec.name);
          }
        }
        break;
      case Stage::Compute: {
        if (!checkpoint.graph_submitted && checkpoint.node < sec.sync_topo.size())
          ranges.phase("Compute");
        auto graph_it               = res_.section_cuda_graphs.find(sec.id);
        current_node_name           = {};
        detail::current_node_name() = {};
        if (!checkpoint.computation_started) {
          checkpoint.computation_started = true;
          if (graph_it != res_.section_cuda_graphs.end() && graph_it->second->enabled) {
            auto      &graphs  = *graph_it->second;
            const auto variant = graphs.variant(res_.storages);
            if (variant) {
              const auto started = std::chrono::steady_clock::now();
              {
                nvtx3::scoped_range launch_range{"CUDA graph launch"};
                CUDA_CHECK(cudaGraphLaunch(graphs.executables.at(*variant), sec.stream));
              }
              checkpoint.graph_submitted = true;
              graphs.launches.fetch_add(1, std::memory_order_relaxed);
              auto &acc = graph_metric_accumulators_.at(section_id);
              acc.duration_ns.fetch_add(
                  static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - started)
                                            .count()),
                  std::memory_order_relaxed);
              acc.run_count.fetch_add(1, std::memory_order_relaxed);
            } else {
              graphs.report_miss(res_.storages);
            }
          }
          if (!checkpoint.graph_submitted && graph_it != res_.section_cuda_graphs.end())
            graph_it->second->ordinary_iterations.fetch_add(1, std::memory_order_relaxed);
        }
        if (checkpoint.graph_submitted || checkpoint.node == sec.sync_topo.size()) {
          advance(Stage::Barrier);
          break;
        }
        const auto v                = sec.sync_topo[checkpoint.node];
        current_node_name           = graph_[v].spec.name;
        detail::current_node_name() = current_node_name;
        try {
          if (run_sync(v) == core::OpResult::Ok) {
            checkpoint.produced.push_back(v);
            ++checkpoint.node;
          }
        } catch (...) {
          rethrow_with_node_context(graph_, v, "execute_sync", section_id, sec.name);
        }
        break;
      }
      case Stage::Barrier:
        current_node_name           = {};
        detail::current_node_name() = {};
        if (!sec.has_synchronizing_async_producer) {
          ranges.phase("Wait");
          CUDA_CHECK(cudaStreamSynchronize(sec.stream));
        }
        advance(Stage::Push);
        break;
      case Stage::Push:
        if (checkpoint.node == sec.async_prod.size()) {
          advance(Stage::Release);
          break;
        }
        {
          const auto v                = sec.async_prod[checkpoint.node];
          current_node_name           = graph_[v].spec.name;
          detail::current_node_name() = current_node_name;
          try {
            ranges.phase("Async push");
            if (run_async_prod(v) == core::OpResult::Ok)
              ++checkpoint.node;
          } catch (...) {
            rethrow_with_node_context(graph_, v, "execute_async_producer", section_id, sec.name);
          }
        }
        break;
      case Stage::Release:
        if (checkpoint.node == checkpoint.produced.size()) {
          ranges.end_iteration();
          checkpoint = SectionCheckpoint{};
          break;
        }
        {
          const auto v                = checkpoint.produced[checkpoint.node];
          current_node_name           = graph_[v].spec.name;
          detail::current_node_name() = current_node_name;
          try {
            if (std::ranges::any_of(graph_[v].infer.owned_outputs,
                                    [](bool owned) { return owned; }))
              ranges.phase("Release outputs");
            release_owned_outputs(v, checkpoint.port);
          } catch (...) {
            rethrow_with_node_context(graph_, v, "release_owned_outputs", section_id, sec.name);
          }
          if (checkpoint.port == graph_[v].infer.owned_outputs.size()) {
            ++checkpoint.node;
            checkpoint.port = 0;
          }
        }
        break;
      }
    }
    ranges.end_iteration();
    // Abort/EOF retain the old cleanup semantics. Pause keeps every pending port and operation.
    if (!paused_.load() || aborted_.load()) {
      nvtx3::scoped_range cleanup_range{"Section cleanup"};
      current_node_name           = {};
      detail::current_node_name() = {};
      CUDA_CHECK(cudaStreamSynchronize(sec.stream));
      const size_t first = checkpoint.stage == Stage::Release ? checkpoint.node : 0;
      for (size_t i = first; i < checkpoint.produced.size(); ++i) {
        current_node_name           = graph_[checkpoint.produced[i]].spec.name;
        detail::current_node_name() = current_node_name;
        size_t port = checkpoint.stage == Stage::Release && i == first ? checkpoint.port : 0;
        release_owned_outputs(checkpoint.produced[i], port);
      }
      checkpoint = SectionCheckpoint{};
    }
  } catch (...) {
    if (current_node_name.empty())
      current_node_name = detail::current_node_name();
    aborted_.store(true);
    paused_.store(false);
    stop_.store(true);
    log_and_abort_current_exception(
        std::format("Scheduler::run_section section={} id={}", sec.name, section_id),
        current_node_name, failure_callback_, failure_dumped_);
  }
}

void Scheduler::Impl::acquire_owned_inputs(GraphPlan::vertex_descriptor v, size_t &port) {
  const auto  idx        = boost::get(boost::vertex_index, graph_, v);
  const auto &np         = graph_[v];
  auto       &nrt        = node_rts_.at(idx);
  const auto &owned_mask = np.infer.owned_inputs;
  auto       *task       = std::holds_alternative<SyncRt>(nrt)
                               ? static_cast<core::ITask *>(std::get<SyncRt>(nrt).task)
                               : static_cast<core::ITask *>(std::get<AsyncRt>(nrt).task);

  for (; port < owned_mask.size(); ++port) {
    const auto i = port;
    if (!owned_mask.at(i))
      continue;

    if (stop_.load())
      return;
    std::optional<core::TView> tview = task->acquire_input(static_cast<int>(i));
    while (!tview.has_value()) {
      if (stop_.load())
        return;
      tview = task->acquire_input(static_cast<int>(i));
    }
  }
}

void Scheduler::Impl::release_owned_outputs(GraphPlan::vertex_descriptor v, size_t &port) {
  const auto  idx        = boost::get(boost::vertex_index, graph_, v);
  const auto &np         = graph_[v];
  auto       &nrt        = node_rts_.at(idx);
  const auto &owned_mask = np.infer.owned_outputs;
  auto       *task       = std::holds_alternative<SyncRt>(nrt)
                               ? static_cast<core::ITask *>(std::get<SyncRt>(nrt).task)
                               : static_cast<core::ITask *>(std::get<AsyncRt>(nrt).task);

  for (; port < owned_mask.size(); ++port) {
    const auto i = port;
    if (!owned_mask.at(i))
      continue;

    if (stop_.load() && paused_.load() && !aborted_.load())
      return;
    task->release_output(static_cast<int>(i));
  }
}

core::OpResult Scheduler::Impl::run_sync(GraphPlan::vertex_descriptor v) {
  using clock     = std::chrono::high_resolution_clock;
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);
  HOLOFLOW_CHECK(std::holds_alternative<SyncRt>(nrt));
  auto &srt = std::get<SyncRt>(nrt);

  logger()->trace("[Scheduler::run_sync] Executing node '{}'", np.spec.name);
  auto t0 = clock::now();
  auto r  = core::OpResult::Cancelled;
  {
    nvtx3::scoped_range task_range{srt.range_name.c_str()};
    r = srt.task->execute(srt.ctx);
  }
  auto t1 = clock::now();

  switch (r) {
  case core::OpResult::Cancelled:
    if (!paused_.load())
      aborted_.store(true);
    logger()->debug("[Scheduler::run_sync] Node '{}' execution cancelled", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::Eof:
    aborted_.store(true);
    paused_.store(false);
    logger()->debug("[Scheduler::run_sync] Node '{}' reached end of stream", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::NotReady:
    aborted_.store(true);
    paused_.store(false);
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
  return r;
}

core::OpResult Scheduler::Impl::run_async_cons(GraphPlan::vertex_descriptor v) {
  using clock     = std::chrono::high_resolution_clock;
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);
  HOLOFLOW_CHECK(std::holds_alternative<AsyncRt>(nrt));
  auto &art = std::get<AsyncRt>(nrt);

  logger()->trace("[Scheduler::run_async_cons] Executing node '{}'", np.spec.name);
  auto t0 = clock::now();
  auto r  = core::OpResult::Cancelled;
  {
    nvtx3::scoped_range task_range{art.pop_range_name.c_str()};
    r = art.task->try_pop(art.xctx);
    while (r == core::OpResult::NotReady) {
      if (stop_.load())
        return core::OpResult::Cancelled;
      r = art.task->try_pop(art.xctx);
    }
  }
  auto t1 = clock::now();

  switch (r) {
  case core::OpResult::Cancelled:
    if (!paused_.load())
      aborted_.store(true);
    logger()->debug("[Scheduler::run_async_cons] Node '{}' execution cancelled", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::Eof:
    aborted_.store(true);
    paused_.store(false);
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
  return r;
}

core::OpResult Scheduler::Impl::run_async_prod(GraphPlan::vertex_descriptor v) {
  using clock     = std::chrono::high_resolution_clock;
  const auto  idx = boost::get(boost::vertex_index, graph_, v);
  const auto &np  = graph_[v];
  auto       &nrt = node_rts_.at(idx);
  HOLOFLOW_CHECK(std::holds_alternative<AsyncRt>(nrt));
  auto &art = std::get<AsyncRt>(nrt);

  logger()->trace("[Scheduler::run_async_prod] Executing node '{}'", np.spec.name);
  auto t0 = clock::now();
  auto r  = core::OpResult::Cancelled;
  {
    nvtx3::scoped_range task_range{art.push_range_name.c_str()};
    r = art.task->try_push(art.pctx);
    while (r == core::OpResult::NotReady) {
      if (stop_.load())
        return core::OpResult::Cancelled;
      r = art.task->try_push(art.pctx);
    }
  }
  auto t1 = clock::now();

  switch (r) {
  case core::OpResult::Cancelled:
    if (!paused_.load())
      aborted_.store(true);
    logger()->debug("[Scheduler::run_async_prod] Node '{}' execution cancelled", np.spec.name);
    stop_.store(true);
    break;
  case core::OpResult::Eof:
    aborted_.store(true);
    paused_.store(false);
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
  return r;
}

void Scheduler::Impl::reset_metrics_state() {
  auto num_vertices = boost::num_vertices(graph_);
  metric_accumulators_.clear();
  metric_accumulators_.resize(num_vertices);
  graph_metric_accumulators_.clear();
  graph_metric_accumulators_.resize(sections_.size());

  std::lock_guard<std::mutex> lock(metrics_mutex_);
  latest_metrics_.clear();
  latest_section_graph_metrics_.clear();
  for (const auto &name : node_names_) {
    latest_metrics_.emplace(name, NodeMetrics{});
  }
}

void Scheduler::Impl::start_metrics_thread() {
  if (metrics_interval_.count() <= 0) {
    return;
  }
  bool expected = false;
  if (!metrics_running_.compare_exchange_strong(expected, true)) {
    return;
  }
  metrics_thread_ = std::thread(&Scheduler::Impl::metrics_loop, this);
}

void Scheduler::Impl::stop_metrics_thread() {
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

void Scheduler::Impl::metrics_loop() {
  tracing::set_thread_name("Scheduler Metrics");
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

void Scheduler::Impl::aggregate_metrics(double interval_seconds) {
  if (interval_seconds <= 0.0) {
    interval_seconds = static_cast<double>(metrics_interval_.count()) / 1000.0;
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

  std::map<std::string, NodeMetrics> section_snapshot;
  for (size_t i = 0; i < sections_.size(); ++i) {
    auto       &acc      = graph_metric_accumulators_[i];
    const auto  runs     = acc.run_count.exchange(0, std::memory_order_relaxed);
    const auto  duration = acc.duration_ns.exchange(0, std::memory_order_relaxed);
    NodeMetrics section_metric;
    section_metric.sample_count    = runs;
    section_metric.runs_per_second = runs / interval_seconds;
    if (runs)
      section_metric.average_duration_ms = double(duration) / runs / 1e6;
    section_snapshot.emplace(sections_[i].name, section_metric);
    if (!runs)
      continue;
    for (auto v : sections_[i].sync_topo) {
      const auto &np = graph_[v];
      auto       &m  = snapshot.at(np.spec.name);
      m.sample_count += runs;
      m.runs_per_second += runs / interval_seconds;
      m.individual_timing_available = false;
      m.average_duration_ms         = 0;
      auto bytes                    = [&](const auto &descs) {
        for (const auto &desc : descs) {
          auto &rate = desc.mem_loc == core::MemLoc::Device ? m.device_throughput_bytes_per_second
                                                            : m.host_throughput_bytes_per_second;
          rate += double(desc.num_bytes()) * runs / interval_seconds;
        }
      };
      bytes(np.infer.input_descs);
      bytes(np.infer.output_descs);
    }
  }
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  latest_metrics_               = std::move(snapshot);
  latest_section_graph_metrics_ = std::move(section_snapshot);
}

void Scheduler::Impl::record_node_sample(std::size_t idx, uint64_t duration_ns, uint64_t host_bytes,
                                         uint64_t device_bytes) {
  auto &acc = metric_accumulators_.at(idx);
  acc.duration_ns.fetch_add(duration_ns, std::memory_order_relaxed);
  acc.run_count.fetch_add(1, std::memory_order_relaxed);
  acc.host_bytes.fetch_add(host_bytes, std::memory_order_relaxed);
  acc.device_bytes.fetch_add(device_bytes, std::memory_order_relaxed);
}

std::pair<uint64_t, uint64_t> Scheduler::Impl::sum_bytes(std::span<const core::TView> views) {
  uint64_t host_total   = 0;
  uint64_t device_total = 0;
  for (const auto &view : views) {
    const auto bytes = static_cast<uint64_t>(view.desc.num_bytes());
    if (view.desc.mem_loc == core::MemLoc::Device) {
      device_total += bytes;
    } else {
      host_total += bytes;
    }
  }
  return {host_total, device_total};
}

// -------------------------------------------------------------------------------------------------
// Public API PIMPL forwarding
// -------------------------------------------------------------------------------------------------

Scheduler::Scheduler(const GraphPlan &graph, const std::vector<Section> &sections,
                     ExecResouces &resources, std::chrono::milliseconds metrics_interval)
    : Scheduler(graph, sections, resources, metrics_interval, {}) {}

Scheduler::Scheduler(const GraphPlan &graph, const std::vector<Section> &sections,
                     ExecResouces &resources, std::chrono::milliseconds metrics_interval,
                     FailureCallback failure_callback)
    : impl_(std::make_unique<Impl>(graph, sections, resources, metrics_interval,
                                   std::move(failure_callback))) {}

Scheduler::~Scheduler() = default;

void Scheduler::set_metrics_interval(std::chrono::milliseconds interval) {
  impl_->set_metrics_interval(interval);
}

std::map<std::string, NodeMetrics> Scheduler::metrics() const { return impl_->metrics(); }

std::map<std::string, NodeMetrics> Scheduler::section_graph_metrics() const {
  return impl_->section_graph_metrics();
}

nlohmann::json Scheduler::section_graph_diagnostics() const {
  return impl_->section_graph_diagnostics();
}

void Scheduler::start() { impl_->start(); }

void Scheduler::request_pause() { impl_->request_pause(); }

void Scheduler::request_stop() { impl_->request_stop(); }

void Scheduler::wait() { impl_->wait(); }

bool Scheduler::is_running() const { return impl_->is_running(); }

bool Scheduler::stop_requested() const { return impl_->stop_requested(); }

std::string_view Scheduler::current_node_name() const noexcept {
  return impl_->current_node_name();
}

bool Scheduler::ui_try_send(const std::string &node_id, nlohmann::json &&data) noexcept {
  return impl_->ui_try_send(node_id, std::move(data));
}

std::optional<holoflow_event::Event> Scheduler::ui_try_receive() noexcept {
  return impl_->ui_try_receive();
}

} // namespace holoflow::runtime
