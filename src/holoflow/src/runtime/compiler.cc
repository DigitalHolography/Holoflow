// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "holoflow/runtime/compiler.hh"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/topological_sort.hpp>
#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <queue>
#include <ranges>
#include <set>
#include <stack>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "curaii/cuda.hh"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow/runtime/graph_exec.hh"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace holoflow::runtime {

// -------------------------------------------------------------------------------------------------
// Internal Types & Error Handling
// -------------------------------------------------------------------------------------------------

class CompilerException : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// -------------------------------------------------------------------------------------------------
// Profiling Data Structures
// -------------------------------------------------------------------------------------------------

struct TraceEvent {
  std::string name;
  std::string category;
  long long   start_us;
  long long   dur_us;
  uint32_t    tid;
};

class CompilationProfiler {
public:
  void add_event(std::string name, std::string category, long long start_us, long long dur_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t tid = static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    events_.push_back({std::move(name), std::move(category), start_us, dur_us, tid});
  }

  void log_summary(std::shared_ptr<spdlog::logger> &logger) const {
    if (!logger || events_.empty())
      return;

    double total_time_ms = 0.0;
    for (const auto &ev : events_) {
      if (ev.name == "Total Compilation") {
        total_time_ms = ev.dur_us / 1000.0;
        break;
      }
    }

    logger->info("{:=^60}", " Compilation Passes Summary ");
    logger->info("{:<30} | {:>12} | {:>10}", "Pass Name", "Time (ms)", "% Total");
    logger->info("{:-^60}", "");

    for (const auto &ev : events_) {
      if (ev.category != "pass" && ev.name != "Total Compilation")
        continue;

      double dur_ms  = ev.dur_us / 1000.0;
      double percent = (total_time_ms > 0) ? (dur_ms / total_time_ms) * 100.0 : 0.0;

      if (ev.name == "Total Compilation") {
        logger->info("{:-^60}", "");
      }
      logger->info("{:<30} | {:>12.3f} | {:>9.2f}%", ev.name, dur_ms, percent);
    }
    logger->info("{:=^60}", "");
  }

  void dump_chrome_tracing(const std::filesystem::path &filepath) const {
    std::ofstream out(filepath);
    if (!out.is_open())
      return;

    out << "[\n";
    for (size_t i = 0; i < events_.size(); ++i) {
      const auto &ev = events_[i];
      out << "  {"
          << "\"name\": \"" << ev.name << "\", "
          << "\"cat\": \"" << ev.category << "\", "
          << "\"ph\": \"X\", "
          << "\"ts\": " << ev.start_us << ", "
          << "\"dur\": " << ev.dur_us << ", "
          << "\"pid\": 1, "
          << "\"tid\": " << ev.tid << "}";
      if (i < events_.size() - 1)
        out << ",";
      out << "\n";
    }
    out << "]\n";
  }

private:
  std::vector<TraceEvent> events_;
  std::mutex              mutex_;
};

// -------------------------------------------------------------------------------------------------
// Observability & Scoped Tracer
// -------------------------------------------------------------------------------------------------

class ScopedTrace {
public:
  using Clock       = std::chrono::steady_clock;
  using SystemClock = std::chrono::system_clock;

  ScopedTrace(std::string name, std::string category, std::shared_ptr<spdlog::logger> logger,
              CompilationProfiler *profiler)
      : name_(std::move(name)), category_(std::move(category)), logger_(std::move(logger)),
        profiler_(profiler) {

    start_time_ = Clock::now();
    start_us_   = std::chrono::time_point_cast<std::chrono::microseconds>(SystemClock::now())
                    .time_since_epoch()
                    .count();

    if (logger_ && category_ == "pass") {
      logger_->trace(">> Begin Pass: {}", name_);
    }

    nvtxRangePush(name_.c_str());
  }

  ~ScopedTrace() {
    auto end_time = Clock::now();
    auto dur_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();

    if (logger_ && category_ == "pass") {
      logger_->info("<< End Pass:   {} ({:.3f} ms)", name_, dur_us / 1000.0);
    }

    if (profiler_) {
      profiler_->add_event(name_, category_, start_us_, dur_us);
    }

    nvtxRangePop();
  }

private:
  std::string                     name_;
  std::string                     category_;
  std::shared_ptr<spdlog::logger> logger_;
  CompilationProfiler            *profiler_;
  Clock::time_point               start_time_;
  long long                       start_us_;
};

// -------------------------------------------------------------------------------------------------
// Storage Adapter for owning tasks
// -------------------------------------------------------------------------------------------------
class TaskStorageAdapter : public core::IOStorageAccess {
public:
  TaskStorageAdapter(std::vector<int> in_tids, std::vector<int> out_tids, ExecResouces &resources);
  [[nodiscard]] core::Storage &owned_input_storage(size_t index) override;
  [[nodiscard]] core::Storage &owned_output_storage(size_t index) override;

private:
  std::vector<int> in_tids_;
  std::vector<int> out_tids_;
  ExecResouces    &res_;
};

TaskStorageAdapter::TaskStorageAdapter(std::vector<int> in_tids, std::vector<int> out_tids,
                                       ExecResouces &resources)
    : in_tids_(std::move(in_tids)), out_tids_(std::move(out_tids)), res_(resources) {}

core::Storage &TaskStorageAdapter::owned_input_storage(size_t index) {
  if (index >= in_tids_.size()) {
    throw std::out_of_range("Input index out of range in TaskStorageAdapter");
  }
  size_t tid = in_tids_[index];
  size_t sid = res_.tid_to_sid.at(tid);
  return *res_.storages.at(sid);
}

core::Storage &TaskStorageAdapter::owned_output_storage(size_t index) {
  if (index >= out_tids_.size()) {
    throw std::out_of_range("Output index out of range in TaskStorageAdapter");
  }
  size_t tid = out_tids_[index];
  size_t sid = res_.tid_to_sid.at(tid);
  return *res_.storages.at(sid);
}

// -------------------------------------------------------------------------------------------------
// Compiler Declaration (PIMPL)
// -------------------------------------------------------------------------------------------------

class Compiler::Impl {
public:
  Impl(core::Registry &registry, Compiler::Config config);

  std::unique_ptr<CompilerOutput> run(const core::GraphSpec          &gspec,
                                      std::unique_ptr<CompilerOutput> prev);

private:
  // --- State ---
  core::Registry                 &registry_;
  Compiler::Config                config_;
  std::shared_ptr<spdlog::logger> logger_;
  CompilationProfiler             profiler_;

  const core::GraphSpec          *gspec_ = nullptr;
  std::unique_ptr<CompilerOutput> prev_;
  std::unique_ptr<CompilerOutput> out_;

  // Auxiliary Map: Node Name -> Section ID
  std::unordered_map<std::string, size_t> node_to_section_map_;

  // --- Helpers ---
  void        setup_logging();
  ScopedTrace trace_scope(std::string name, std::string category = "pass");
  void        dump_graphviz(const std::string &filename);
  template <class TaskInterface, class Factory, class Ctx>
  std::unique_ptr<core::ITask> create_or_update_task(Factory &factory, const NodePlan &np,
                                                     const Ctx &ctx);

  // --- Pass Declarations ---
  void validate_spec();
  void build_graph_structure();
  void run_type_inference();
  void assign_tensor_ids();
  void assign_storage_ids();
  void verify_buffer_consistency();
  void allocate_buffers();
  void create_storage_adapters();
  void partition_sections();
  void assign_streams();
  void instantiate_tasks();
  void bind_tasks();

  // Generic Pass Runner
  template <typename Func> void run_pass(const char *name, Func &&fn) {
    auto scope = trace_scope(name, "pass");
    fn();
  }
};

// -------------------------------------------------------------------------------------------------
// Compiler Implementation (PIMPL)
// -------------------------------------------------------------------------------------------------

Compiler::Impl::Impl(core::Registry &registry, Compiler::Config config)
    : registry_(registry), config_(std::move(config)) {
  setup_logging();
}

std::unique_ptr<CompilerOutput> Compiler::Impl::run(const core::GraphSpec          &gspec,
                                                    std::unique_ptr<CompilerOutput> prev) {
  gspec_ = &gspec;
  prev_  = std::move(prev);
  out_   = std::make_unique<CompilerOutput>();

  // Use optional to control exactly when the trace ends without double-destruction
  std::optional<ScopedTrace> total_trace;
  total_trace.emplace(trace_scope("Total Compilation", "lifecycle"));

  try {
    run_pass("Validate Spec", [&] { validate_spec(); });
    run_pass("Build Graph Plan", [&] { build_graph_structure(); });
    run_pass("Type Inference", [&] { run_type_inference(); });

    run_pass("Tensor IDs", [&] { assign_tensor_ids(); });
    run_pass("Storage Mapping", [&] { assign_storage_ids(); });
    run_pass("Buffer Allocation", [&] { allocate_buffers(); });

    run_pass("Storage Adapters", [&] { create_storage_adapters(); });

    run_pass("Section Partitioning", [&] { partition_sections(); });
    run_pass("Stream Assignment", [&] { assign_streams(); });
    run_pass("Task Instantiation", [&] { instantiate_tasks(); });
    run_pass("Task Binding", [&] { bind_tasks(); });

    if (config_.dump_dot_on_failure) {
      run_pass("Dump Graphviz", [&] { dump_graphviz("compilation_success.dot"); });
    }
  } catch (const std::exception &e) {
    logger_->error("Compilation Failed: {}", e.what());
    if (config_.dump_dot_on_failure) {
      run_pass("Dump Graphviz", [&] { dump_graphviz("compilation_failure.dot"); });
    }

    total_trace.reset(); // Stop timer before throwing

    try {
      CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception &cuda_e) {
      logger_->error("CUDA error during cleanup: {}", cuda_e.what());
    }

    try {
      CUDA_CHECK(cudaGetLastError());
    } catch (const std::exception &cuda_e) {
      logger_->error("CUDA error during cleanup: {}", cuda_e.what());
    }

    logger_->flush();
    throw;
  }

  // Stop the total compilation timer safely
  total_trace.reset();

  if (config_.enable_profiling) {
    // profiler_.log_summary(logger_);
    run_pass("Dump log summary", [&] { profiler_.log_summary(logger_); });
    if (!config_.log_dir.empty()) {
      // profiler_.dump_chrome_tracing(config_.log_dir / config_.trace_filename);
      run_pass("Dump Chrome Tracing",
               [&] { profiler_.dump_chrome_tracing(config_.log_dir / config_.trace_filename); });
    }
  }

  return std::move(out_);
}

void Compiler::Impl::setup_logging() {
  if (spdlog::get("compiler")) {
    spdlog::drop("compiler");
  }

  if (!config_.log_dir.empty()) {
    std::filesystem::create_directories(config_.log_dir);
    auto path = config_.log_dir / "compiler.log";
    logger_   = spdlog::basic_logger_mt("compiler", path.string(), true);
  } else {
    logger_ = spdlog::stdout_color_mt("compiler");
  }
  logger_->set_level(config_.verbose_tracing ? spdlog::level::trace : spdlog::level::info);
}

ScopedTrace Compiler::Impl::trace_scope(std::string name, std::string category) {
  return ScopedTrace(std::move(name), std::move(category), logger_,
                     config_.enable_profiling ? &profiler_ : nullptr);
}

// -------------------------------------------------------------------------------------------------
// Pass: Validate Spec
// -------------------------------------------------------------------------------------------------
void Compiler::Impl::validate_spec() {
  std::unordered_set<std::string> names;
  std::unordered_set<std::string> edge_dsts;

  auto vertices = boost::make_iterator_range(boost::vertices(*gspec_));
  for (const auto &v : vertices) {
    const auto &ns = (*gspec_)[v];
    if (!names.insert(ns.name).second) {
      throw CompilerException(std::format("Duplicate node name: '{}'", ns.name));
    }
    if (!registry_.is_registered(ns.kind)) {
      throw CompilerException(std::format("Unknown node kind '{}'", ns.kind));
    }
  }

  auto edges = boost::make_iterator_range(boost::edges(*gspec_));
  for (const auto &e : edges) {
    const auto &es    = (*gspec_)[e];
    const auto  dst   = (*gspec_)[boost::target(e, *gspec_)];
    std::string label = std::format("{}:{}", dst.name, es.in_idx);

    if (!edge_dsts.insert(label).second) {
      throw CompilerException(std::format("Multiple edges targeting: {}", label));
    }
  }
}

// -------------------------------------------------------------------------------------------------
// Pass: Build Graph Structure
// -------------------------------------------------------------------------------------------------
void Compiler::Impl::build_graph_structure() {
  using VSpec = core::GraphSpec::vertex_descriptor;
  using VPlan = GraphPlan::vertex_descriptor;
  std::map<VSpec, VPlan> v_map;
  auto                  &g = out_->graph;

  for (auto v : boost::make_iterator_range(boost::vertices(*gspec_))) {
    NodePlan np;
    np.spec  = (*gspec_)[v];
    v_map[v] = boost::add_vertex(np, g);
  }

  for (auto e : boost::make_iterator_range(boost::edges(*gspec_))) {
    const auto &es  = (*gspec_)[e];
    const auto  src = v_map.at(boost::source(e, *gspec_));
    const auto  dst = v_map.at(boost::target(e, *gspec_));
    EdgePlan    ep;
    ep.spec = es;
    boost::add_edge(src, dst, ep, g);
  }
}

// -------------------------------------------------------------------------------------------------
// Pass: Type Inference
// -------------------------------------------------------------------------------------------------
void Compiler::Impl::run_type_inference() {
  auto                                     &g = out_->graph;
  std::vector<GraphPlan::vertex_descriptor> topo_order;

  try {
    boost::topological_sort(g, std::back_inserter(topo_order));
  } catch (const boost::not_a_dag &) {
    throw CompilerException("Graph contains a cycle (loop), which is not allowed.");
  }

  for (auto v : std::views::reverse(topo_order)) {
    auto &node       = g[v];
    auto  node_trace = trace_scope(std::format("Infer: {}", node.spec.name), "detail");

    auto                     in_degree = boost::in_degree(v, g);
    std::vector<core::TDesc> input_descs(in_degree);

    for (auto e : boost::make_iterator_range(boost::in_edges(v, g))) {
      const auto &edge_plan = g[e];
      if (edge_plan.spec.in_idx >= input_descs.size()) {
        throw CompilerException("Input index out of bounds");
      }
      input_descs[edge_plan.spec.in_idx] = edge_plan.desc;
    }

    const auto &factory = registry_.get(node.spec.kind);
    node.infer          = factory.infer(input_descs, node.spec.settings);

    for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
      auto &edge_plan = g[e];
      if (edge_plan.spec.out_idx >= node.infer.output_descs.size()) {
        throw CompilerException("Output index out of bounds");
      }
      edge_plan.desc = node.infer.output_descs[edge_plan.spec.out_idx];
    }
  }
}

// -------------------------------------------------------------------------------------------------
// Pass: Assign Tensor IDs
// -------------------------------------------------------------------------------------------------
void Compiler::Impl::assign_tensor_ids() {
  auto &g        = out_->graph;
  auto &res      = out_->resources;
  int   next_tid = 0;

  std::vector<GraphPlan::vertex_descriptor> topo;
  boost::topological_sort(g, std::back_inserter(topo));

  for (auto v : std::views::reverse(topo)) {
    auto &node = g[v];

    node.in_tids.resize(node.infer.input_descs.size());
    for (auto e : boost::make_iterator_range(boost::in_edges(v, g))) {
      const auto &ep               = g[e];
      node.in_tids[ep.spec.in_idx] = ep.tid;
      res.tensor_descs[ep.tid]     = ep.desc;
    }

    node.out_tids.resize(node.infer.output_descs.size());
    auto out_edges = boost::out_edges(v, g);

    for (size_t i = 0; i < node.out_tids.size(); ++i) {
      int tid               = next_tid++;
      node.out_tids[i]      = tid;
      res.tensor_descs[tid] = node.infer.output_descs[i];

      for (auto e : boost::make_iterator_range(out_edges)) {
        if (g[e].spec.out_idx == static_cast<int>(i)) {
          g[e].tid = tid;
        }
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------
// Pass: Assign Storage IDs
// -------------------------------------------------------------------------------------------------
void Compiler::Impl::assign_storage_ids() {
  auto &g   = out_->graph;
  auto &res = out_->resources;

  res.tid_to_sid.clear();
  int next_sid = 0;

  std::vector<GraphPlan::vertex_descriptor> topo;
  boost::topological_sort(g, std::back_inserter(topo));

  for (auto v : std::views::reverse(topo)) {
    auto &node = g[v];

    for (size_t out_idx = 0; out_idx < node.out_tids.size(); ++out_idx) {
      int out_tid = node.out_tids[out_idx];
      int sid     = -1;

      for (const auto &ip : node.infer.in_place) {
        if (ip.out_idx == static_cast<int>(out_idx)) {
          int in_tid = node.in_tids[ip.in_idx];

          if (res.tid_to_sid.contains(in_tid)) {
            sid = (int)res.tid_to_sid.at(in_tid);
          } else {
            throw CompilerException(
                std::format("Node '{}': In-place input TID {} has no Storage ID assigned.",
                            node.spec.name, in_tid));
          }
          break;
        }
      }

      if (sid == -1) {
        sid = next_sid++;
      }
      res.tid_to_sid[out_tid] = sid;
    }
  }
}

void Compiler::Impl::verify_buffer_consistency() {
  std::map<int, std::vector<std::string>> owners;
  auto                                   &g = out_->graph;

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &node = g[v];
    for (size_t i = 0; i < node.infer.owned_inputs.size(); ++i) {
      if (node.infer.owned_inputs[i]) {
        owners[node.in_tids[i]].push_back(node.spec.name + ":in");
      }
    }
    for (size_t i = 0; i < node.infer.owned_outputs.size(); ++i) {
      if (node.infer.owned_outputs[i]) {
        owners[node.out_tids[i]].push_back(node.spec.name + ":out");
      }
    }
  }

  for (const auto &[tid, nodeList] : owners) {
    if (nodeList.size() > 1) {
      throw CompilerException(std::format("TID {} has multiple owners", tid));
    }
  }
}

// void Compiler::Impl::allocate_buffers() {
//   auto &g   = out_->graph;
//   auto &res = out_->resources;

//   res.memory_blocks.clear();
//   res.storages.clear();

//   std::unordered_set<size_t> user_managed_sids;
//   for (auto v : boost::make_iterator_range(boost::vertices(g))) {
//     const auto &node = g[v];
//     for (size_t i = 0; i < node.out_tids.size(); ++i) {
//       if (node.infer.owned_outputs[i]) {
//         int    tid = node.out_tids[i];
//         size_t sid = res.tid_to_sid.at(tid);
//         user_managed_sids.insert(sid);
//       }
//     }
//     for (size_t i = 0; i < node.in_tids.size(); ++i) {
//       if (node.infer.owned_inputs[i]) {
//         int    tid = node.in_tids[i];
//         size_t sid = res.tid_to_sid.at(tid);
//         user_managed_sids.insert(sid);
//       }
//     }
//   }

//   std::map<size_t, size_t> sid_to_rep_tid;
//   for (const auto &[tid, sid] : res.tid_to_sid) {
//     if (!sid_to_rep_tid.count(sid))
//       sid_to_rep_tid[sid] = tid;
//   }

//   for (const auto &[sid, tid] : sid_to_rep_tid) {
//     auto        alloc_scope = trace_scope(std::format("Alloc SID {}", sid), "detail");
//     const auto &desc        = res.tensor_descs.at(tid);

//     auto storage     = std::make_unique<core::Storage>();
//     storage->mem_loc = desc.mem_loc;
//     storage->bytes   = desc.num_bytes();
//     storage->ptr     = nullptr;

//     if (!user_managed_sids.contains(sid)) {
//       MemoryBlock block;
//       block.mem_loc    = desc.mem_loc;
//       block.size_bytes = desc.num_bytes();

//       logger_->info("Allocating {} bytes for SID {} at {:?} memory", block.size_bytes, sid,
//                     to_string(desc.mem_loc));

//       // Use a standard scope block to control the RAII timer
//       {
//         auto sys_scope = trace_scope(
//             desc.mem_loc == core::MemLoc::Host ? "Host Malloc" : "Device Malloc", "syscall");
//         if (desc.mem_loc == core::MemLoc::Host) {
//           block.h_data = curaii::make_unique_host_ptr<std::byte>(block.size_bytes);
//         } else {
//           block.d_data = curaii::make_unique_device_ptr<std::byte>(block.size_bytes);
//         }
//       } // sys_scope naturally destructs here!

//       storage->ptr = static_cast<std::byte *>(block.get());
//       res.memory_blocks.emplace(sid, std::move(block));
//     } else {
//       logger_->info("SID {} is user-managed; skipping allocation.", sid);
//     }

//     res.storages.emplace(sid, std::move(storage));
//   }

//   // Temp test, trigger a 1b cuda memcopy to see how it shows up in the profiler
//   CUDA_CHECK(cudaDeviceSynchronize());
//   if (res.memory_blocks.size() >= 2) {
//     auto &block1 = res.memory_blocks.begin()->second;
//     auto &block2 = std::next(res.memory_blocks.begin())->second;
//     if (block1.mem_loc == core::MemLoc::Device && block2.mem_loc == core::MemLoc::Device) {
//       auto sys_scope = trace_scope("Test Memcpy", "syscall");
//       CUDA_CHECK(cudaMemcpy(block2.get(), block1.get(), 1, cudaMemcpyDeviceToDevice));
//       CUDA_CHECK(cudaDeviceSynchronize());
//     }
//   }
// }

void Compiler::Impl::allocate_buffers() {
  auto &g   = out_->graph;
  auto &res = out_->resources;

  res.memory_blocks.clear();
  res.storages.clear();

  // 1. Identify user-managed SIDs
  std::unordered_set<size_t> user_managed_sids;
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &node = g[v];
    for (size_t i = 0; i < node.out_tids.size(); ++i) {
      if (node.infer.owned_outputs[i]) {
        user_managed_sids.insert(res.tid_to_sid.at(node.out_tids[i]));
      }
    }
    for (size_t i = 0; i < node.in_tids.size(); ++i) {
      if (node.infer.owned_inputs[i]) {
        user_managed_sids.insert(res.tid_to_sid.at(node.in_tids[i]));
      }
    }
  }

  // 2. Map SID to representative TID
  std::map<size_t, size_t> sid_to_rep_tid;
  for (const auto &[tid, sid] : res.tid_to_sid) {
    if (!sid_to_rep_tid.count(sid)) {
      sid_to_rep_tid[sid] = tid;
    }
  }

  // 3. Build a pool of scavengable blocks from prev_
  // Key: {MemLoc, size_in_bytes}
  std::multimap<std::pair<core::MemLoc, size_t>, MemoryBlock> free_blocks;
  if (prev_) {
    for (auto &[prev_sid, block] : prev_->resources.memory_blocks) {
      free_blocks.emplace(std::make_pair(block.mem_loc, block.size_bytes), std::move(block));
    }
  }

  // 4. Allocate or Scavenge
  for (const auto &[sid, tid] : sid_to_rep_tid) {
    auto        alloc_scope = trace_scope(std::format("Alloc SID {}", sid), "detail");
    const auto &desc        = res.tensor_descs.at(tid);

    auto storage     = std::make_unique<core::Storage>();
    storage->mem_loc = desc.mem_loc;
    storage->bytes   = desc.num_bytes();
    storage->ptr     = nullptr;

    if (!user_managed_sids.contains(sid)) {
      MemoryBlock block;
      auto        pool_key = std::make_pair(desc.mem_loc, desc.num_bytes());
      auto        it       = free_blocks.find(pool_key);

      if (it != free_blocks.end()) {
        // We found an exact match! Scavenge it.
        logger_->info("Reusing {} bytes for SID {} at {:?} memory", desc.num_bytes(), sid,
                      to_string(desc.mem_loc));
        block = std::move(it->second);
        free_blocks.erase(it);
      } else {
        // No match found, allocate fresh memory.
        block.mem_loc    = desc.mem_loc;
        block.size_bytes = desc.num_bytes();

        logger_->info("Allocating {} bytes for SID {} at {:?} memory", block.size_bytes, sid,
                      to_string(desc.mem_loc));

        {
          auto sys_scope = trace_scope(
              desc.mem_loc == core::MemLoc::Host ? "Host Malloc" : "Device Malloc", "syscall");
          if (desc.mem_loc == core::MemLoc::Host) {
            block.h_data = curaii::make_unique_host_ptr<std::byte>(block.size_bytes);
          } else {
            block.d_data = curaii::make_unique_device_ptr<std::byte>(block.size_bytes);
          }
        }
      }

      storage->ptr = static_cast<std::byte *>(block.get());
      res.memory_blocks.emplace(sid, std::move(block));
    } else {
      logger_->info("SID {} is user-managed; skipping allocation.", sid);
    }

    res.storages.emplace(sid, std::move(storage));
  }

  // Temp test, trigger a 1b cuda memcopy to see how it shows up in the profiler
  CUDA_CHECK(cudaDeviceSynchronize());
  if (res.memory_blocks.size() >= 2) {
    auto &block1 = res.memory_blocks.begin()->second;
    auto &block2 = std::next(res.memory_blocks.begin())->second;
    if (block1.mem_loc == core::MemLoc::Device && block2.mem_loc == core::MemLoc::Device) {
      auto sys_scope = trace_scope("Test Memcpy", "syscall");
      CUDA_CHECK(cudaMemcpy(block2.get(), block1.get(), 1, cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  // Any blocks left inside `free_blocks` will naturally go out of scope here and
  // safely deallocate, meaning memory for removed nodes is properly cleaned up.
}

// -------------------------------------------------------------------------------------------------
// Pass: Section Partitioning
// -------------------------------------------------------------------------------------------------
void Compiler::Impl::partition_sections() {
  auto &g         = out_->graph;
  auto  num_verts = boost::num_vertices(g);

  std::vector<size_t> parent(num_verts);
  std::iota(parent.begin(), parent.end(), 0);

  auto find = [&](size_t i) {
    while (i != parent[i]) {
      parent[i] = parent[parent[i]];
      i         = parent[i];
    }
    return i;
  };

  auto unite = [&](size_t i, size_t j) {
    size_t root_i = find(i);
    size_t root_j = find(j);
    if (root_i != root_j)
      parent[root_i] = root_j;
  };

  for (auto e : boost::make_iterator_range(boost::edges(g))) {
    auto u = boost::source(e, g);
    auto v = boost::target(e, g);
    if (g[u].infer.kind == core::TaskKind::Sync && g[v].infer.kind == core::TaskKind::Sync) {
      unite(u, v);
    }
  }

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    if (g[v].infer.kind == core::TaskKind::Async) {
      std::vector<size_t> sync_preds;
      for (auto e : boost::make_iterator_range(boost::in_edges(v, g))) {
        auto p = boost::source(e, g);
        if (g[p].infer.kind == core::TaskKind::Sync)
          sync_preds.push_back(p);
      }
      if (!sync_preds.empty()) {
        for (size_t i = 1; i < sync_preds.size(); ++i)
          unite(sync_preds[0], sync_preds[i]);
      }

      std::vector<size_t> sync_succs;
      for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
        auto s = boost::target(e, g);
        if (g[s].infer.kind == core::TaskKind::Sync)
          sync_succs.push_back(s);
      }
      if (!sync_succs.empty()) {
        for (size_t i = 1; i < sync_succs.size(); ++i)
          unite(sync_succs[0], sync_succs[i]);
      }
    }
  }

  std::map<size_t, size_t> root_to_section_id;
  out_->sections.clear();
  int next_sec_id = 0;
  node_to_section_map_.clear();

  auto get_section_id = [&](size_t v_idx) {
    size_t root = find(v_idx);
    if (root_to_section_id.find(root) == root_to_section_id.end()) {
      Section s;
      s.id   = next_sec_id;
      s.name = std::format("section-{}", next_sec_id);
      out_->sections.push_back(s);
      root_to_section_id[root] = next_sec_id++;
    }
    return root_to_section_id[root];
  };

  std::vector<GraphPlan::vertex_descriptor> topo;
  boost::topological_sort(g, std::back_inserter(topo));

  for (auto v : std::views::reverse(topo)) {
    auto &np = g[v];
    if (np.infer.kind == core::TaskKind::Sync) {
      size_t sec_id = get_section_id(v);
      out_->sections[sec_id].sync_topo.push_back(v);
      node_to_section_map_[np.spec.name] = sec_id;
    }
  }

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    if (g[v].infer.kind != core::TaskKind::Async)
      continue;

    std::set<size_t> unique_cons_sections;
    for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
      auto s = boost::target(e, g);
      if (g[s].infer.kind == core::TaskKind::Sync) {
        unique_cons_sections.insert(get_section_id(s));
      }
    }
    for (size_t sec_id : unique_cons_sections) {
      out_->sections[sec_id].async_cons.push_back(v);
    }

    std::set<size_t> unique_prod_sections;
    for (auto e : boost::make_iterator_range(boost::in_edges(v, g))) {
      auto p = boost::source(e, g);
      if (g[p].infer.kind == core::TaskKind::Sync) {
        unique_prod_sections.insert(get_section_id(p));
      }
    }
    for (size_t sec_id : unique_prod_sections) {
      out_->sections[sec_id].async_prod.push_back(v);
    }
  }
}

// void Compiler::Impl::assign_streams() {
//   out_->resources.streams.clear();
//   for (auto &sec : out_->sections) {
//     curaii::CudaStream stream;
//     sec.stream = stream.get();
//     out_->resources.streams.emplace(sec.id, std::move(stream));
//   }
// }

void Compiler::Impl::assign_streams() {
  out_->resources.streams.clear();

  // 1. Only set up the iterators if prev_ actually exists
  if (prev_) {
    auto prev_it  = prev_->resources.streams.begin();
    auto prev_end = prev_->resources.streams.end();

    for (auto &sec : out_->sections) {
      if (prev_it != prev_end) {
        // Scavenge an existing stream
        auto &old_stream = prev_it->second;
        sec.stream       = old_stream.get();

        out_->resources.streams.emplace(sec.id, std::move(old_stream));
        ++prev_it;
      } else {
        // Fallback: Create a new stream (ran out of old ones)
        curaii::CudaStream stream;
        sec.stream = stream.get();
        out_->resources.streams.emplace(sec.id, std::move(stream));
      }
    }
  } else {
    // 2. No previous graph at all, just create fresh streams for everything
    for (auto &sec : out_->sections) {
      curaii::CudaStream stream;
      sec.stream = stream.get();
      out_->resources.streams.emplace(sec.id, std::move(stream));
    }
  }
}

void Compiler::Impl::create_storage_adapters() {
  auto &g   = out_->graph;
  auto &res = out_->resources;

  res.node_storage_adapters.clear();

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &np      = g[v];
    auto        adapter = std::make_unique<TaskStorageAdapter>(np.in_tids, np.out_tids, res);
    res.node_storage_adapters.emplace(np.spec.name, std::move(adapter));
  }
}

template <class To, class From>
std::unique_ptr<To> dynamic_unique_ptr_cast(std::unique_ptr<From> &&ptr) noexcept {
  if (auto casted = dynamic_cast<To *>(ptr.get())) {
    ptr.release();
    return std::unique_ptr<To>(casted);
  }
  return nullptr;
}

template <class TaskInterface, class Factory, class Ctx>
std::unique_ptr<core::ITask>
Compiler::Impl::create_or_update_task(Factory &factory, const NodePlan &np, const Ctx &ctx) {

  // 1. Helper to synchronize the correct streams based on Ctx type
  auto sync_streams = [&]() {
    if constexpr (std::is_same_v<Ctx, core::SyncCreateCtx>) {
      if (ctx.stream) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
      } else {
        logger_->warn("SyncCreateCtx has null stream; skipping synchronization.");
      }
    } else if constexpr (std::is_same_v<Ctx, core::AsyncCreateCtx>) {
      if (ctx.producer_stream) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.producer_stream));
      } else {
        logger_->warn("AsyncCreateCtx has null producer_stream; skipping synchronization.");
      }
      if (ctx.consumer_stream) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.consumer_stream));
      } else {
        logger_->warn("AsyncCreateCtx has null consumer_stream; skipping synchronization.");
      }
    }
  };

  // 2. Helper to accurately profile Task Creation
  auto do_create = [&]() {
    auto scope = trace_scope(std::format("Create Task: {}", np.spec.name), "detail");
    auto task  = factory.create(np.infer.input_descs, np.spec.settings, ctx);
    sync_streams();
    return task;
  }; // scope naturally destructs here, capturing the fully synced time

  // 3. Helper to accurately profile Task Updating
  auto do_update = [&](std::unique_ptr<TaskInterface> prev_task) {
    auto scope = trace_scope(std::format("Update Task: {}", np.spec.name), "detail");
    auto task  = factory.update(std::move(prev_task), np.infer.input_descs, np.spec.settings, ctx);
    sync_streams();
    return task;
  };

  // --- Main Logic ---

  std::unique_ptr<core::ITask> *prev_ptr_ref = nullptr;
  if (prev_ && !prev_->resources.tasks.empty()) {
    auto &prev_tasks = prev_->resources.tasks;
    if (auto it = prev_tasks.find(np.spec.name); it != prev_tasks.end()) {
      prev_ptr_ref = &it->second;
    }
  }

  if (!prev_ptr_ref) {
    return do_create();
  }

  bool kind_mismatch       = false;
  bool found_in_prev_graph = false;

  auto [vi, vi_end] = boost::vertices(prev_->graph);
  for (; vi != vi_end; ++vi) {
    const NodePlan &prev_node = prev_->graph[*vi];
    if (prev_node.spec.name == np.spec.name) {
      found_in_prev_graph = true;
      if (prev_node.spec.kind != np.spec.kind) {
        kind_mismatch = true;
      }
      break;
    }
  }

  if (!found_in_prev_graph || kind_mismatch) {
    return do_create();
  }

  auto prev_task_typed = dynamic_unique_ptr_cast<TaskInterface>(std::move(*prev_ptr_ref));

  if (!prev_task_typed) {
    return do_create();
  }

  return do_update(std::move(prev_task_typed));
}

void Compiler::Impl::instantiate_tasks() {
  auto &g     = out_->graph;
  auto &tasks = out_->resources.tasks;

  tasks.clear();

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &np = g[v];

    if (np.infer.kind == core::TaskKind::Sync) {
      size_t sid    = node_to_section_map_.at(np.spec.name);
      auto   stream = out_->resources.streams.at(sid).get();

      core::SyncCreateCtx ctx{.stream = stream};
      auto               &factory = registry_.get_sync(np.spec.kind);

      auto task = create_or_update_task<core::ISyncTask>(factory, np, ctx);
      tasks.emplace(np.spec.name, std::move(task));

    } else if (np.infer.kind == core::TaskKind::Async) {
      void *prod_stream = nullptr;
      for (auto e : boost::make_iterator_range(boost::in_edges(v, g))) {
        auto p = boost::source(e, g);
        if (g[p].infer.kind == core::TaskKind::Sync) {
          size_t sid  = node_to_section_map_.at(g[p].spec.name);
          prod_stream = out_->resources.streams.at(sid).get();
          break;
        }
      }

      void *cons_stream = nullptr;
      for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
        auto s = boost::target(e, g);
        if (g[s].infer.kind == core::TaskKind::Sync) {
          size_t sid  = node_to_section_map_.at(g[s].spec.name);
          cons_stream = out_->resources.streams.at(sid).get();
          break;
        }
      }

      core::AsyncCreateCtx ctx{.producer_stream = static_cast<cudaStream_t>(prod_stream),
                               .consumer_stream = static_cast<cudaStream_t>(cons_stream)};

      auto &factory = registry_.get_async(np.spec.kind);

      auto task = create_or_update_task<core::IAsyncTask>(factory, np, ctx);
      tasks.emplace(np.spec.name, std::move(task));
    }
  }
}

std::shared_ptr<spdlog::logger> create_task_logger(const std::string &node_name,
                                                   const std::string &node_kind) {
  auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [thread %t] [%^%l%$] %v");
  auto logger_name = fmt::format("TaskLogger-{}-{}", node_kind, node_name);
  auto logger      = std::make_shared<spdlog::logger>(logger_name, sink);
  logger->set_level(spdlog::default_logger()->level());
  return logger;
}

void Compiler::Impl::bind_tasks() {
  auto &g   = out_->graph;
  auto &res = out_->resources;

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &np = g[v];

    auto it_task = res.tasks.find(np.spec.name);
    if (it_task == res.tasks.end())
      continue;
    core::ITask *task = it_task->second.get();

    if (auto it_adapter = res.node_storage_adapters.find(np.spec.name);
        it_adapter != res.node_storage_adapters.end()) {
      task->bind_storage_access(it_adapter->second.get());
    }

    auto logger = create_task_logger(np.spec.name, np.spec.kind);
    task->bind_logger(std::move(logger));
  }
}

// -------------------------------------------------------------------------------------------------
// Visualization Helpers (Internal)
// -------------------------------------------------------------------------------------------------

namespace {

std::string escape_dot_label(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      break;
    default:
      out += c;
      break;
    }
  }
  return out;
}

std::string format_tdesc(const core::TDesc &d) {
  std::ostringstream ss;
  ss << "{\\n";
  ss << "  shape: " << escape_dot_label(nlohmann::json(d.shape).dump()) << ",\\n";
  ss << "  dtype: " << escape_dot_label(nlohmann::json(d.dtype).dump()) << "\\n";
  ss << "  mem_loc: " << escape_dot_label(nlohmann::json(d.mem_loc).dump()) << "\\n";
  ss << "  strides: " << escape_dot_label(nlohmann::json(d.strides).dump()) << "\\n";
  ss << "  offset: " << d.offset << "\\n";
  ss << "}";
  return ss.str();
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Debugging: Graphviz Dump (Implementation)
// -------------------------------------------------------------------------------------------------

void Compiler::Impl::dump_graphviz(const std::string &filename) {
  if (config_.log_dir.empty()) {
    return;
  }

  std::ofstream file(config_.log_dir / filename);
  if (!file.is_open()) {
    return;
  }

  auto &g        = out_->graph;
  auto &res      = out_->resources;
  auto &sections = out_->sections;

  file << "digraph GraphPlan {\n";
  file << "  rankdir=LR;\n";
  file << "  compound=true;\n";
  file << "  node [fontname=\"Helvetica\", shape=box, style=filled];\n";
  file << "  edge [fontname=\"Helvetica\", fontsize=10];\n\n";

  auto fmt_id = [&](int tid) -> std::string {
    if (res.tid_to_sid.contains(tid)) {
      return std::format("{}(s:{})", tid, res.tid_to_sid.at(tid));
    }
    return std::to_string(tid);
  };

  auto get_visual_id = [&](size_t v, bool is_source) -> std::string {
    if (g[v].infer.kind == core::TaskKind::Async) {
      return is_source ? std::format("v{}_out", v) : std::format("v{}_in", v);
    }
    return std::format("v{}", v);
  };

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &np = g[v];

    std::ostringstream label_base;
    label_base << (np.spec.name.empty() ? "(unnamed)" : np.spec.name);

    std::string in_str = "[";
    for (size_t i = 0; i < np.in_tids.size(); ++i) {
      in_str += (i ? "," : "") + fmt_id(np.in_tids[i]);
    }
    in_str += "]";

    std::string out_str = "[";
    for (size_t i = 0; i < np.out_tids.size(); ++i) {
      const int   out_tid = np.out_tids[i];
      std::string id_text = fmt_id(out_tid);

      bool is_alias = false;
      if (res.tid_to_sid.count(out_tid)) {
        const size_t out_sid = res.tid_to_sid.at(out_tid);
        for (int in_tid : np.in_tids) {
          if (res.tid_to_sid.count(in_tid) && res.tid_to_sid.at(in_tid) == out_sid) {
            is_alias = true;
            break;
          }
        }
      }

      out_str += (i ? "," : "") + id_text + (is_alias ? "*" : "");
    }
    out_str += "]";

    const std::string ids_line = "\nIn: " + in_str + "\nOut: " + out_str;

    if (np.infer.kind == core::TaskKind::Async) {
      const std::string label_in = label_base.str() + "\n(Producer/Write)" + ids_line + "\n";
      file << std::format("  v{}_in [label=\"{}\", shape=invhouse, fillcolor=\"#e6f2ff\", "
                          "color=\"#0066cc\", style=\"filled,dashed\"];\n",
                          v, escape_dot_label(label_in));

      const std::string label_out = label_base.str() + "\n(Consumer/Read)" + ids_line + "\n";
      file << std::format("  v{}_out [label=\"{}\", shape=house, fillcolor=\"#ffe6e6\", "
                          "color=\"#cc0000\", style=\"filled,dashed\"];\n",
                          v, escape_dot_label(label_out));

      file << std::format("  v{}_in -> v{}_out [style=dotted, color=\"#888888\", penwidth=2, "
                          "arrowh=none, label=\"Async Signal\"];\n",
                          v, v);
    } else {
      const std::string label = label_base.str() + "\n(" + np.spec.kind + ")" + ids_line + "\n";
      file << std::format("  v{} [label=\"{}\", fillcolor=\"#ccffcc\"];\n", v,
                          escape_dot_label(label));
    }
  }

  file << "\n";

  for (auto e : boost::make_iterator_range(boost::edges(g))) {
    const auto  u  = boost::source(e, g);
    const auto  v  = boost::target(e, g);
    const auto &ep = g[e];

    const std::string u_vis = get_visual_id(u, true);
    const std::string v_vis = get_visual_id(v, false);

    std::ostringstream edge_lbl;
    edge_lbl << "tid:" << ep.tid;
    if (res.tid_to_sid.count(ep.tid)) {
      edge_lbl << " (s:" << res.tid_to_sid.at(ep.tid) << ")";
    }
    edge_lbl << "\\n" << format_tdesc(ep.desc);

    file << std::format("  {} -> {} [taillabel=\"{}\", headlabel=\"{}\", label=\"{}\"];\n", u_vis,
                        v_vis, ep.spec.out_idx, ep.spec.in_idx, edge_lbl.str());
  }

  file << "\n";

  for (const auto &sec : sections) {
    file << std::format("  subgraph cluster_section_{} {{\n", sec.id);
    file << std::format("    label=\"Section {} (Stream {})\\l\";\n", sec.id, (void *)sec.stream);
    file << "    style=rounded; color=gray; bgcolor=\"#f8f8f8\";\n";

    for (auto vd : sec.sync_topo) {
      file << std::format("    v{};\n", vd);
    }
    for (auto vd : sec.async_prod) {
      file << std::format("    v{}_in;\n", vd);
    }
    for (auto vd : sec.async_cons) {
      file << std::format("    v{}_out;\n", vd);
    }

    file << "  }\n";
  }

  file << "}\n";
}

// -------------------------------------------------------------------------------------------------
// Public API PIMPL forwarding
// -------------------------------------------------------------------------------------------------

Compiler::Compiler(core::Registry &registry, Config config)
    : impl_(std::make_unique<Impl>(registry, std::move(config))) {}

Compiler::~Compiler() = default;

std::unique_ptr<CompilerOutput> Compiler::compile(const core::GraphSpec          &gspec,
                                                  std::unique_ptr<CompilerOutput> prev) {
  return impl_->run(gspec, std::move(prev));
}

} // namespace holoflow::runtime