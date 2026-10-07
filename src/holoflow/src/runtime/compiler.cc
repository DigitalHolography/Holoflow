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
#include "diagnostics_file_writer.hh"
#include "holoflow/runtime/diagnostics.hh"
#include "holoflow/runtime/tracing.hh"
#include "section_cuda_graph.hh"

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/topological_sort.hpp>
#include <format>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <ranges>
#include <set>
#include <stack>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "curaii/cuda.hh"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow/runtime/graph_display.hh"
#include "holoflow/runtime/graph_exec.hh"
#include "spdlog/sinks/base_sink.h"
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

using tracing::ScopedTrace;

namespace {
class CompilerLogBufferSink final : public spdlog::sinks::base_sink<std::mutex> {
public:
  void clear() {
    std::lock_guard lock(mutex_);
    buffer_.clear();
  }
  std::string take() {
    std::lock_guard lock(mutex_);
    return std::exchange(buffer_, {});
  }

private:
  void sink_it_(const spdlog::details::log_msg &message) override {
    spdlog::memory_buf_t text;
    formatter_->format(message, text);
    buffer_.append(text.data(), text.size());
  }
  void        flush_() override {}
  std::string buffer_;
};
} // namespace

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
  ~Impl() {
    // Also covers exceptions raised while reporting a compilation failure.
    if (out_)
      out_->resources.section_cuda_graphs.clear();
    if (prev_)
      prev_->resources.section_cuda_graphs.clear();
  }

  std::unique_ptr<CompilerOutput> run(const core::GraphSpec          &gspec,
                                      std::unique_ptr<CompilerOutput> prev);

private:
  // --- State ---
  core::Registry                        &registry_;
  Compiler::Config                       config_;
  std::shared_ptr<spdlog::logger>        logger_;
  std::shared_ptr<CompilerLogBufferSink> log_buffer_;

  const core::GraphSpec          *gspec_ = nullptr;
  std::unique_ptr<CompilerOutput> prev_;
  std::unique_ptr<CompilerOutput> out_;
  // Unused allocations stay alive until their executable dependencies have been retired.
  std::multimap<std::pair<core::MemLoc, size_t>, MemoryBlock> retired_blocks_;
  uint64_t                                                    compilation_generation_ = 1;

  // Auxiliary Map: Node Name -> Section ID
  std::unordered_map<std::string, size_t> node_to_section_map_;

  // --- Helpers ---
  void        setup_logging();
  void        publish_log() noexcept;
  ScopedTrace trace_scope(std::string name, std::string category = "pass");
  void        dump_graphviz(const std::string &filename);
  void        dump_json(const std::string &filename, const core::GraphSpec &gspec);
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
  void invalidate_task_graphs(std::string_view name) noexcept;
  void carry_section_cuda_graphs();

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
  ScopedTrace initialization("Compiler Initialization");
  ScopedTrace logging("Setup Compiler Logging");
  setup_logging();
}

std::unique_ptr<CompilerOutput> Compiler::Impl::run(const core::GraphSpec          &gspec,
                                                    std::unique_ptr<CompilerOutput> prev) {
  if (log_buffer_)
    log_buffer_->clear();
  // Publish on every exit, including exceptions during failure reporting/cleanup.
  struct LogPublication {
    Impl &compiler;
    ~LogPublication() { compiler.publish_log(); }
  } log_publication{*this};
  tracing::Capture capture(config_.log_dir.empty() ? std::filesystem::path{}
                                                   : config_.log_dir / config_.trace_filename,
                           config_.enable_profiling);
  std::optional<ScopedTrace> total_trace;
  total_trace.emplace("Total Compilation");

  try {
    run_pass("Initialize Compilation", [&] {
      gspec_                  = &gspec;
      prev_                   = std::move(prev);
      out_                    = std::make_unique<CompilerOutput>();
      compilation_generation_ = prev_ ? prev_->resources.compilation_generation + 1 : 1;
    });
    run_pass("Dump Graph Spec", [&] { dump_json("graph_spec.json", gspec); });
    if (prev_) {
      // Stop/wait is the caller's responsibility. Drain before any dependent resources change.
      run_pass("Drain Previous CUDA Streams", [&] {
        for (auto &[id, stream] : prev_->resources.streams) {
          auto scope = trace_scope(std::format("Synchronize Previous Stream {}", id), "detail");
          CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        }
      });
    }
    run_pass("Validate Spec", [&] { validate_spec(); });
    run_pass("Build Graph Plan", [&] { build_graph_structure(); });
    run_pass("Type Inference", [&] { run_type_inference(); });

    run_pass("Tensor IDs", [&] { assign_tensor_ids(); });
    run_pass("Storage Mapping", [&] { assign_storage_ids(); });
    run_pass("Buffer Consistency", [&] { verify_buffer_consistency(); });
    run_pass("Buffer Allocation", [&] { allocate_buffers(); });

    run_pass("Storage Adapters", [&] { create_storage_adapters(); });

    run_pass("Section Partitioning", [&] { partition_sections(); });
    run_pass("Stream Assignment", [&] { assign_streams(); });
    run_pass("Task Instantiation", [&] { instantiate_tasks(); });
    run_pass("Task Binding", [&] { bind_tasks(); });
    run_pass("Carry Compatible Section CUDA Graphs", [&] { carry_section_cuda_graphs(); });
    run_pass("Inspect Section CUDA Graphs", [&] {
      out_->resources.max_section_cuda_graphs    = config_.max_section_cuda_graphs;
      out_->resources.section_cuda_graph_log_dir = config_.log_dir;
      refresh_section_cuda_graphs(out_->graph, out_->sections, out_->resources, false);
    });

    if (config_.dump_dot_on_failure) {
      run_pass("Dump Graphviz", [&] { dump_graphviz("compilation_success.dot"); });
    }
  } catch (const std::exception &e) {
    logger_->error("Compilation Failed: {}", e.what());
    if (config_.dump_dot_on_failure) {
      run_pass("Dump Graphviz", [&] { dump_graphviz("compilation_failure.dot"); });
    }

    total_trace->set_outcome(tracing::Outcome::Failure);
    total_trace.reset(); // Close the measured scope before cleanup/export.

    try {
      CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception &cuda_e) {
      logger_->error("CUDA error during cleanup: {}", cuda_e.what());
    }

    // Clear graphs before releasing tasks or retired buffers, including partially transferred
    // state.
    if (out_)
      out_->resources.section_cuda_graphs.clear();
    if (prev_)
      prev_->resources.section_cuda_graphs.clear();
    retired_blocks_.clear();

    try {
      CUDA_CHECK(cudaGetLastError());
    } catch (const std::exception &cuda_e) {
      logger_->error("CUDA error during cleanup: {}", cuda_e.what());
    }

    logger_->flush();
    throw;
  }

  total_trace->set_outcome(tracing::Outcome::Success);
  total_trace.reset();

  return std::move(out_);
}

void Compiler::Impl::setup_logging() {
  {
    ScopedTrace retirement("Retire Compiler Logger", "detail");
    spdlog::drop("compiler");
  }
  if (!config_.log_dir.empty()) {
    ScopedTrace buffer("Create Compiler Log Buffer", "detail");
    log_buffer_ = std::make_shared<CompilerLogBufferSink>();
    logger_     = std::make_shared<spdlog::logger>("compiler", log_buffer_);
    spdlog::initialize_logger(logger_);
  } else {
    logger_ = spdlog::stdout_color_mt("compiler");
  }
  ScopedTrace configuration("Configure Compiler Logger", "detail");
  logger_->set_level(config_.verbose_tracing ? spdlog::level::trace : spdlog::level::info);
}

void Compiler::Impl::publish_log() noexcept {
  if (!log_buffer_)
    return;
  try {
    ScopedTrace submission("Submit Compiler Log", "detail");
    section_diagnostics_file_writer().submit_text(
        config_.log_dir / "compiler.log",
        [text = log_buffer_->take()]() mutable { return std::move(text); });
  } catch (...) {
    // Diagnostic failures must not mask a successful compilation or its original exception.
    try {
      spdlog::warn("Could not submit compiler log for background writing");
    } catch (...) {
    }
  }
}

ScopedTrace Compiler::Impl::trace_scope(std::string name, std::string category) {
  return ScopedTrace(std::move(name), category);
}

void Compiler::Impl::dump_graphviz(const std::string &filename) {
  if (config_.log_dir.empty()) {
    return;
  }

  std::error_code error;
  std::filesystem::create_directories(config_.log_dir, error);
  if (error)
    return;
  std::ofstream file(config_.log_dir / filename);
  if (!file.is_open()) {
    return;
  }

  const auto graph_name = std::filesystem::path(filename).stem().string();
  file << to_dot(*out_, GraphCompiledDumpPreferences{}, graph_name);
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
  std::map<size_t, std::vector<std::string>> owners;
  auto                                      &g   = out_->graph;
  auto                                      &res = out_->resources;

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &node = g[v];
    for (size_t i = 0; i < node.infer.owned_inputs.size(); ++i) {
      if (node.infer.owned_inputs[i]) {
        owners[res.tid_to_sid.at(node.in_tids[i])].push_back(node.spec.name + ":in");
      }
    }
    for (size_t i = 0; i < node.infer.owned_outputs.size(); ++i) {
      if (node.infer.owned_outputs[i]) {
        owners[res.tid_to_sid.at(node.out_tids[i])].push_back(node.spec.name + ":out");
      }
    }
  }

  for (const auto &[sid, nodeList] : owners) {
    if (nodeList.size() > 1) {
      throw CompilerException(std::format("Storage ID {} has multiple owners", sid));
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
  auto &free_blocks = retired_blocks_;
  free_blocks.clear();
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
      MemoryBlock                 block;
      core::ExecutionInvalidation allocation_failure{[this]() noexcept {
        if (prev_)
          prev_->resources.section_cuda_graphs.clear();
      }};
      // A scavenged block may temporarily be local before insertion into the output map.
      core::ExecutionUpdateGuard allocation_guard(&allocation_failure);
      auto                       pool_key = std::make_pair(desc.mem_loc, desc.num_bytes());
      auto                       it       = free_blocks.find(pool_key);

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

  // Leftovers are retired only after dependent graph caches have been destroyed.
}

// -------------------------------------------------------------------------------------------------
// Pass: Section Partitioning
// -------------------------------------------------------------------------------------------------
void Compiler::Impl::partition_sections() {
  auto &g         = out_->graph;
  auto  num_verts = boost::num_vertices(g);

  for (auto e : boost::make_iterator_range(boost::edges(g))) {
    const auto source = boost::source(e, g);
    const auto target = boost::target(e, g);
    if (g[source].infer.kind == core::TaskKind::Async &&
        g[target].infer.kind == core::TaskKind::Async) {
      throw CompilerException(
          std::format("Consecutive asynchronous nodes '{}' and '{}' are not supported",
                      g[source].spec.name, g[target].spec.name));
    }
  }

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

  for (auto &section : out_->sections) {
    // A synchronizing producer must run before ordinary producers: its barrier covers all preceding
    // sync-node work before an ordinary queue is allowed to publish its GPU-backed input.
    const auto ordinary_begin =
        std::stable_partition(section.async_prod.begin(), section.async_prod.end(),
                              [&](auto v) { return g[v].infer.synchronizes_producer_stream; });
    section.has_synchronizing_async_producer = ordinary_begin != section.async_prod.begin();
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
  // Construct the callback before moving any captured task out of the previous resource map.
  core::ExecutionInvalidation invalidation{
      [this, &np]() noexcept { invalidate_task_graphs(np.spec.name); }};

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
    core::ExecutionUpdateGuard ownership_guard(&invalidation);
    auto scope      = trace_scope(std::format("Update Task: {}", np.spec.name), "detail");
    auto update_ctx = ctx;
    update_ctx.execution_invalidation = &invalidation;
    if (factory.execution_update_policy() == core::ExecutionUpdatePolicy::AlwaysInvalidate)
      invalidation.invalidate();
    auto task =
        factory.update(std::move(prev_task), np.infer.input_descs, np.spec.settings, update_ctx);
    // A post-update synchronization error must invalidate before the returned task is unwound.
    core::ExecutionUpdateGuard update_guard(&invalidation);
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
    invalidate_task_graphs(np.spec.name);
    return do_create();
  }

  if (!dynamic_cast<TaskInterface *>(prev_ptr_ref->get()))
    invalidate_task_graphs(np.spec.name);
  auto prev_task_typed = dynamic_unique_ptr_cast<TaskInterface>(std::move(*prev_ptr_ref));

  if (!prev_task_typed) {
    return do_create();
  }

  return do_update(std::move(prev_task_typed));
}

// -------------------------------------------------------------------------------------------------
// Execution cache compatibility
// -------------------------------------------------------------------------------------------------

void Compiler::Impl::invalidate_task_graphs(std::string_view name) noexcept {
  if (!prev_)
    return;
  for (const auto &section : prev_->sections) {
    auto it = prev_->resources.section_cuda_graphs.find(section.id);
    if (it == prev_->resources.section_cuda_graphs.end())
      continue;
    auto &graphs  = *it->second;
    bool  depends = false;
    for (auto v : section.sync_topo)
      depends |= prev_->graph[v].spec.name == name;
    for (auto v : boost::make_iterator_range(boost::vertices(prev_->graph))) {
      const auto &node = prev_->graph[v];
      if (node.spec.name != name)
        continue;
      auto owns_domain = [&](const auto &tids, const auto &owned) {
        for (size_t port = 0; port < tids.size(); ++port) {
          if (!owned[port])
            continue;
          auto sid = prev_->resources.tid_to_sid.at(tids[port]);
          if (std::ranges::find(graphs.storage_ids, sid) != graphs.storage_ids.end())
            return true;
        }
        return false;
      };
      depends |= owns_domain(node.in_tids, node.infer.owned_inputs) ||
                 owns_domain(node.out_tids, node.infer.owned_outputs);
    }
    if (!depends)
      continue;
    graphs.clear();
    try {
      graphs.compilation_invalidation_reason = "task_update: " + std::string(name);
      logger_->info("[CUDA graphs] Invalidated section {} before updating {}", section.name, name);
    } catch (...) {
      // Invalidation also runs during exception unwinding; reporting must never block cleanup.
    }
  }
}

namespace {

// This describes the current unfused recording plan, independently of factory configuration.
// Future lowering/fusion must include generated-kernel identity and specialization constants here.
nlohmann::json section_structure(const CompilerOutput &output, const Section &section) {
  auto                  result = nlohmann::json::array();
  std::set<std::string> names;
  for (const auto *vertices : {&section.sync_topo, &section.async_cons, &section.async_prod}) {
    auto tasks = nlohmann::json::array();
    for (auto v : *vertices) {
      const auto &node = output.graph[v];
      tasks.push_back({node.spec.name, node.spec.kind});
      names.insert(node.spec.name);
    }
    result.push_back(std::move(tasks));
  }
  std::set<std::tuple<std::string, int, std::string, int>> wiring;
  for (auto edge : boost::make_iterator_range(boost::edges(output.graph))) {
    const auto &source = output.graph[boost::source(edge, output.graph)].spec.name;
    const auto &target = output.graph[boost::target(edge, output.graph)].spec.name;
    if (names.contains(source) || names.contains(target)) {
      const auto &ports = output.graph[edge].spec;
      wiring.emplace(source, ports.out_idx, target, ports.in_idx);
    }
  }
  result.push_back(wiring);
  return result;
}

nlohmann::json section_bindings(const CompilerOutput &output, const Section &section) {
  auto result = nlohmann::json::array();
  for (auto v : section.sync_topo) {
    const auto &node     = output.graph[v];
    auto        bindings = nlohmann::json::array();
    for (const auto *tids : {&node.in_tids, &node.out_tids}) {
      auto ports = nlohmann::json::array();
      for (int tid : *tids) {
        const auto &desc = output.resources.tensor_descs.at(tid);
        // The public TDesc serializer intentionally excludes offsets; captures cannot exclude them.
        ports.push_back({output.resources.tid_to_sid.at(tid), desc, desc.offset});
      }
      bindings.push_back(std::move(ports));
    }
    bindings.push_back(node.infer.owned_inputs);
    bindings.push_back(node.infer.owned_outputs);
    result.push_back(std::move(bindings));
  }
  return result;
}

} // namespace

void Compiler::Impl::carry_section_cuda_graphs() {
  out_->resources.compilation_generation = compilation_generation_;
  // Destroy executables referencing allocations that will not survive compilation, before free.
  if (prev_) {
    for (auto &[id, graphs] : prev_->resources.section_cuda_graphs) {
      bool retires_resource = false;
      for (auto &[key, block] : retired_blocks_)
        for (const auto &domain : graphs->pointers)
          retires_resource |=
              std::ranges::find(domain, static_cast<std::byte *>(block.get())) != domain.end();
      if (retires_resource) {
        graphs->clear();
        graphs->compilation_invalidation_reason = "resource_retirement";
      }
    }
  }
  for (const auto &section : out_->sections) {
    auto        graphs = std::make_unique<SectionCudaGraphs>();
    std::string reason = prev_ ? "section_structure" : "initial_compilation";
    if (prev_) {
      for (const auto &old_section : prev_->sections) {
        if (section_structure(*prev_, old_section) != section_structure(*out_, section))
          continue;
        auto it = prev_->resources.section_cuda_graphs.find(old_section.id);
        if (it == prev_->resources.section_cuda_graphs.end())
          continue;
        if (old_section.stream != section.stream)
          reason = "stream_change";
        else if (section_bindings(*prev_, old_section) != section_bindings(*out_, section))
          reason = "descriptor_or_binding_change";
        else if (it->second->executables.empty())
          reason = it->second->compilation_invalidation_reason.empty()
                       ? "no_cached_executables"
                       : it->second->compilation_invalidation_reason;
        else {
          graphs = std::move(it->second);
          prev_->resources.section_cuda_graphs.erase(it);
          graphs->carried_from_previous_compilation = true;
          reason.clear();
        }
        break;
      }
    }
    graphs->compilation_generation          = compilation_generation_;
    graphs->compilation_invalidation_reason = std::move(reason);
    graphs->launches.store(0);
    graphs->ordinary_iterations.store(0);
    graphs->pointer_misses.store(0);
    graphs->tuple_misses.store(0);
    graphs->refresh_count = 0;
    logger_->info("[CUDA graphs] {} compilation {}: carried {}, reason {}", section.name,
                  compilation_generation_, graphs->carried_from_previous_compilation,
                  graphs->compilation_invalidation_reason);
    out_->resources.section_cuda_graphs.emplace(section.id, std::move(graphs));
  }
  if (prev_)
    prev_->resources.section_cuda_graphs.clear();
  retired_blocks_.clear();
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
void Compiler::Impl::dump_json(const std::string &filename, const core::GraphSpec &gspec) {
  if (!config_.log_dir.empty())
    dump_graph_spec_async(config_.log_dir / filename, gspec);
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
