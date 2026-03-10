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
#include <numeric>
#include <queue>
#include <ranges>
#include <set>
#include <stack>
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
// Observability & Scoped Tracer
// -------------------------------------------------------------------------------------------------
// This class uses RAII (Resource Acquisition Is Initialization) to automatically log
// the start and end times of compiler passes.
class ScopedTrace {
public:
  using Clock = std::chrono::steady_clock;

  ScopedTrace(std::string name, std::shared_ptr<spdlog::logger> logger)
      : name_(std::move(name)), logger_(std::move(logger)), start_(Clock::now()) {
    if (logger_)
      logger_->trace(">> Begin Pass: {}", name_);
  }

  ~ScopedTrace() {
    auto dur = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    if (logger_)
      logger_->info("<< End Pass:   {} ({:.3f} ms)", name_, dur);
  }

private:
  std::string                     name_;
  std::shared_ptr<spdlog::logger> logger_;
  Clock::time_point               start_;
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
// The "Pointer to Implementation" (PIMPL) idiom is used here to hide the heavy
// Boost.Graph templates from the public header file. This improves compilation times
// for users of the Compiler class.
// See: https://en.cppreference.com/w/cpp/language/pimpl

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

  const core::GraphSpec          *gspec_ = nullptr;
  std::unique_ptr<CompilerOutput> prev_;
  std::unique_ptr<CompilerOutput> out_;

  // Auxiliary Map: Node Name -> Section ID
  std::unordered_map<std::string, size_t> node_to_section_map_;

  // --- Helpers ---
  void        setup_logging();
  ScopedTrace trace_scope(std::string name);
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
    auto scope = trace_scope(name);
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
  auto total_trace = trace_scope("Total Compilation");

  gspec_ = &gspec;
  prev_  = std::move(prev);
  out_   = std::make_unique<CompilerOutput>();

  try {
    // 1. Structure
    run_pass("Validate Spec", [&] { validate_spec(); });
    run_pass("Build Graph Plan", [&] { build_graph_structure(); });
    run_pass("Type Inference", [&] { run_type_inference(); });

    // 2. Memory Planning (The New Logic)
    run_pass("Tensor IDs", [&] { assign_tensor_ids(); });
    run_pass("Storage Mapping", [&] { assign_storage_ids(); });
    run_pass("Buffer Allocation", [&] { allocate_buffers(); });

    // 3. Resource Preparation
    run_pass("Storage Adapters", [&] { create_storage_adapters(); });

    // 4. Execution Planning
    run_pass("Section Partitioning", [&] { partition_sections(); });
    run_pass("Stream Assignment", [&] { assign_streams(); });
    run_pass("Task Instantiation", [&] { instantiate_tasks(); });
    run_pass("Task Binding", [&] { bind_tasks(); });

    if (config_.dump_dot_on_failure) {
      dump_graphviz("compilation_success.dot");
    }
  } catch (const std::exception &e) {
    logger_->error("Compilation Failed: {}", e.what());
    if (config_.dump_dot_on_failure) {
      dump_graphviz("compilation_failure.dot");
    }
    throw;
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

ScopedTrace Compiler::Impl::trace_scope(std::string name) {
  return ScopedTrace(std::move(name), logger_);
}

// -------------------------------------------------------------------------------------------------
// Pass: Validate Spec
// -------------------------------------------------------------------------------------------------
// Checks for basic structural errors: duplicates names, unregistered node types,
// and multiple edges writing to the same input slot.
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
    const auto &es  = (*gspec_)[e];
    const auto  dst = (*gspec_)[boost::target(e, *gspec_)];
    // Key format: "node_name:input_index"
    std::string label = std::format("{}:{}", dst.name, es.in_idx);

    if (!edge_dsts.insert(label).second) {
      throw CompilerException(std::format("Multiple edges targeting: {}", label));
    }
  }
}

// -------------------------------------------------------------------------------------------------
// Pass: Build Graph Structure
// -------------------------------------------------------------------------------------------------
// Converts the high-level configuration (GraphSpec) into the runtime representation
// (GraphPlan), which is a Boost Adjacency List.
void Compiler::Impl::build_graph_structure() {
  using VSpec = core::GraphSpec::vertex_descriptor;
  using VPlan = GraphPlan::vertex_descriptor;
  std::map<VSpec, VPlan> v_map;
  auto                  &g = out_->graph;

  // 1. Copy Nodes
  for (auto v : boost::make_iterator_range(boost::vertices(*gspec_))) {
    NodePlan np;
    np.spec  = (*gspec_)[v];
    v_map[v] = boost::add_vertex(np, g);
  }

  // 2. Copy Edges
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
// Propagates data types (e.g., float32 vs int8, image resolution) through the graph.
// We must process nodes in "Topological Order" (dependencies first) so that when we
// visit a node, we know exactly what its inputs look like.
//
// See: https://en.wikipedia.org/wiki/Topological_sorting
void Compiler::Impl::run_type_inference() {
  auto                                     &g = out_->graph;
  std::vector<GraphPlan::vertex_descriptor> topo_order;

  try {
    // Boost's topological_sort outputs nodes in reverse topological order (sinks first).
    boost::topological_sort(g, std::back_inserter(topo_order));
  } catch (const boost::not_a_dag &) {
    throw CompilerException("Graph contains a cycle (loop), which is not allowed.");
  }

  // Iterate in reverse (Source -> Sink)
  for (auto v : std::views::reverse(topo_order)) {
    auto                    &node      = g[v];
    auto                     in_degree = boost::in_degree(v, g);
    std::vector<core::TDesc> input_descs(in_degree);

    // 1. Gather Input Descriptions from incoming edges
    for (auto e : boost::make_iterator_range(boost::in_edges(v, g))) {
      const auto &edge_plan = g[e];
      if (edge_plan.spec.in_idx >= input_descs.size()) {
        throw CompilerException("Input index out of bounds");
      }
      input_descs[edge_plan.spec.in_idx] = edge_plan.desc;
    }

    // 2. Call the User-Defined Factory to infer outputs based on inputs
    const auto &factory = registry_.get(node.spec.kind);
    node.infer          = factory.infer(input_descs, node.spec.settings);

    // 3. Propagate Output Descriptions to outgoing edges
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
// Assigns a unique ID to every tensor flowing through edges.
void Compiler::Impl::assign_tensor_ids() {
  auto &g        = out_->graph;
  auto &res      = out_->resources;
  int   next_tid = 0;

  // Topological sort ensures flow direction
  std::vector<GraphPlan::vertex_descriptor> topo;
  boost::topological_sort(g, std::back_inserter(topo));

  for (auto v : std::views::reverse(topo)) {
    auto &node = g[v];

    // A. Inputs: Snapshot TIDs from incoming edges
    node.in_tids.resize(node.infer.input_descs.size());
    for (auto e : boost::make_iterator_range(boost::in_edges(v, g))) {
      const auto &ep               = g[e];
      node.in_tids[ep.spec.in_idx] = ep.tid;
      // Store Logical Description
      res.tensor_descs[ep.tid] = ep.desc;
    }

    // B. Outputs: Assign NEW Unique TIDs (Static Single Assignment)
    node.out_tids.resize(node.infer.output_descs.size());
    auto out_edges = boost::out_edges(v, g);

    for (size_t i = 0; i < node.out_tids.size(); ++i) {
      int tid               = next_tid++;
      node.out_tids[i]      = tid;
      res.tensor_descs[tid] = node.infer.output_descs[i];

      // Stamp TID on outgoing edges
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

      // A. Check for In-Place / Aliasing
      // If this output reuses an input buffer, inherit the SID.
      for (const auto &ip : node.infer.in_place) {
        if (ip.out_idx == static_cast<int>(out_idx)) {
          int in_tid = node.in_tids[ip.in_idx];

          if (res.tid_to_sid.contains(in_tid)) {
            sid = (int)res.tid_to_sid.at(in_tid);
          } else {
            // Should never happen in valid topological order
            throw CompilerException(
                std::format("Node '{}': In-place input TID {} has no Storage ID assigned.",
                            node.spec.name, in_tid));
          }
          break;
        }
      }

      // B. If no alias found, assign a NEW Storage ID
      if (sid == -1) {
        sid = next_sid++;
      }

      // C. Record Logical -> Physical Mapping
      res.tid_to_sid[out_tid] = sid;
    }
  }
}

void Compiler::Impl::verify_buffer_consistency() {
  // Logic: Ensure a single tensor ID is not "Owned" (created) by multiple nodes simultaneously.
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

void Compiler::Impl::allocate_buffers() {
  auto &g   = out_->graph;
  auto &res = out_->resources;

  res.memory_blocks.clear();
  res.storages.clear();

  // A. Identify SIDs that are "Owned" by user tasks (Framework should NOT alloc)
  std::unordered_set<size_t> user_managed_sids;
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &node = g[v];
    for (size_t i = 0; i < node.out_tids.size(); ++i) {
      if (node.infer.owned_outputs[i]) {
        int    tid = node.out_tids[i];
        size_t sid = res.tid_to_sid.at(tid);
        user_managed_sids.insert(sid);
      }
    }
    for (size_t i = 0; i < node.in_tids.size(); ++i) {
      if (node.infer.owned_inputs[i]) {
        int    tid = node.in_tids[i];
        size_t sid = res.tid_to_sid.at(tid);
        user_managed_sids.insert(sid);
      }
    }
  }

  // B. Group TIDs by their SID to find a representative Descriptor
  // (We just need one valid descriptor per SID to know size/loc)
  std::map<size_t, size_t> sid_to_rep_tid;
  for (const auto &[tid, sid] : res.tid_to_sid) {
    if (!sid_to_rep_tid.count(sid))
      sid_to_rep_tid[sid] = tid;
  }

  // C. Create Storage Identities & Physical Blocks
  for (const auto &[sid, tid] : sid_to_rep_tid) {
    const auto &desc = res.tensor_descs.at(tid);

    // 1. Create the Stable Identity Object
    auto storage     = std::make_unique<core::Storage>();
    storage->mem_loc = desc.mem_loc;
    storage->bytes   = desc.num_bytes();
    storage->ptr     = nullptr; // Default to null

    // 2. Allocate Physical Memory (if not user-managed)
    if (!user_managed_sids.contains(sid)) {

      MemoryBlock block;
      block.mem_loc    = desc.mem_loc;
      block.size_bytes = desc.num_bytes();

      logger_->info("Allocating {} bytes for SID {} at {:?} memory", block.size_bytes, sid,
                    to_string(desc.mem_loc));

      if (desc.mem_loc == core::MemLoc::Host) {
        block.h_data = curaii::make_unique_host_ptr<std::byte>(block.size_bytes);
      } else {
        block.d_data = curaii::make_unique_device_ptr<std::byte>(block.size_bytes);
      }

      // Link Identity -> Physical Pointer
      storage->ptr = static_cast<std::byte *>(block.get());

      // Store ownership
      res.memory_blocks.emplace(sid, std::move(block));
    }

    else {
      logger_->info("SID {} is user-managed; skipping allocation.", sid);
    }

    // Store Identity
    res.storages.emplace(sid, std::move(storage));
  }
}

// -------------------------------------------------------------------------------------------------
// Pass: Section Partitioning (Connected Components)
// -------------------------------------------------------------------------------------------------
// This method groups "Sync" nodes (CPU/Synchronous) that are connected to each other
// into "Sections". Each Section will run on a specific CUDA stream.
//
// Algorithm: Disjoint-Set Union (DSU) / Union-Find.
// This is an efficient algorithm to find connected components in a graph.
// It supports two operations:
//   1. Find(i): Determine which set element 'i' belongs to.
//   2. Union(i, j): Merge the sets containing 'i' and 'j'.
//
// See: https://en.wikipedia.org/wiki/Disjoint-set_data_structure
void Compiler::Impl::partition_sections() {
  //
  auto &g         = out_->graph;
  auto  num_verts = boost::num_vertices(g);

  // Initialize DSU: Each node is its own parent initially.
  std::vector<size_t> parent(num_verts);
  std::iota(parent.begin(), parent.end(), 0);

  // DSU: Find with Path Compression (flattens the tree for O(1) avg lookup)
  auto find = [&](size_t i) {
    while (i != parent[i]) {
      parent[i] = parent[parent[i]]; // path compression
      i         = parent[i];
    }
    return i;
  };

  // DSU: Unite
  auto unite = [&](size_t i, size_t j) {
    size_t root_i = find(i);
    size_t root_j = find(j);
    if (root_i != root_j)
      parent[root_i] = root_j;
  };

  // 1. Unite adjacent Sync nodes directly
  for (auto e : boost::make_iterator_range(boost::edges(g))) {
    auto u = boost::source(e, g);
    auto v = boost::target(e, g);
    if (g[u].infer.kind == core::TaskKind::Sync && g[v].infer.kind == core::TaskKind::Sync) {
      unite(u, v);
    }
  }

  // 2. Unite Sync nodes separated by "Walls" (Async nodes).
  // If an Async node has multiple Sync inputs, those inputs effectively belong to
  // the same logical execution section (they must be ready before the Async node starts).
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    if (g[v].infer.kind == core::TaskKind::Async) {
      // Group all Sync Predecessors
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

      // Group all Sync Successors
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

  // 3. Create Sections from DSU Roots
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

  // 4. Assign Nodes to Sections (Respecting Topological Order)
  std::vector<GraphPlan::vertex_descriptor> topo;
  boost::topological_sort(g, std::back_inserter(topo));

  // Sync Nodes: Add directly to their section's topo list
  for (auto v : std::views::reverse(topo)) {
    auto &np = g[v];
    if (np.infer.kind == core::TaskKind::Sync) {
      size_t sec_id = get_section_id(v);
      out_->sections[sec_id].sync_topo.push_back(v);
      node_to_section_map_[np.spec.name] = sec_id;
    }
  }

  // Async Nodes: Attach as consumers/producers to adjacent sections
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    if (g[v].infer.kind != core::TaskKind::Async)
      continue;

    // Consumers (Output side of Async Node -> Sync Node)
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

    // Producers (Input side of Async Node <- Sync Node)
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

  // Log Section Info (and its nodes in topo order)
  for (const auto &sec : out_->sections) {
    logger_->info("Section {}: ", sec.name);
    logger_->info("  Async Consumers:");
    for (const auto &anode : sec.async_cons) {
      logger_->info("    - {}", g[anode].spec.name);
    }
    logger_->info("  Sync Nodes:");
    for (const auto &snode : sec.sync_topo) {
      logger_->info("    - {}", g[snode].spec.name);
    }
    logger_->info("  Async Producers:");
    for (const auto &anode : sec.async_prod) {
      logger_->info("    - {}", g[anode].spec.name);
    }
  }

  logger_->info("Total Sections Created: {}", out_->sections.size());
  logger_->flush();
}

void Compiler::Impl::assign_streams() {
  out_->resources.streams.clear();
  for (auto &sec : out_->sections) {
    // RAII CudaStream allocation
    curaii::CudaStream stream;
    sec.stream = stream.get();
    out_->resources.streams.emplace(sec.id, std::move(stream));
  }
}

void Compiler::Impl::create_storage_adapters() {
  auto &g   = out_->graph;
  auto &res = out_->resources;

  res.node_storage_adapters.clear();

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &np = g[v];

    // Create the adapter specific to this node's TIDs
    auto adapter = std::make_unique<TaskStorageAdapter>(np.in_tids, np.out_tids, res);

    // Store it keyed by node name for easy retrieval during instantiation
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
  // 1. Check if we have previous state to query
  std::unique_ptr<core::ITask> *prev_ptr_ref = nullptr;
  if (prev_ && !prev_->resources.tasks.empty()) {
    auto &prev_tasks = prev_->resources.tasks;
    if (auto it = prev_tasks.find(np.spec.name); it != prev_tasks.end()) {
      prev_ptr_ref = &it->second;
    }
  }

  // 2. If no previous task exists, just create new
  if (!prev_ptr_ref) {
    return factory.create(np.infer.input_descs, np.spec.settings, ctx);
  }

  // 3. Verify the Node Kind hasn't changed in the Graph Plan
  // (We must iterate prev_graph to find the node definition for this name)
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

  // If the node wasn't in the old graph (unlikely if we have a task)
  // or the kind changed (e.g. "Gaussian" -> "Median"), we must recreate.
  if (!found_in_prev_graph || kind_mismatch) {
    return factory.create(np.infer.input_descs, np.spec.settings, ctx);
  }

  // 4. Attempt to cast the previous task to the specific interface (ISyncTask/IAsyncTask)
  // assuming dynamic_unique_ptr_cast is available in your utils
  auto prev_task_typed = dynamic_unique_ptr_cast<TaskInterface>(std::move(*prev_ptr_ref));

  if (!prev_task_typed) {
    // Fallback safety: Cast failed, create new
    return factory.create(np.infer.input_descs, np.spec.settings, ctx);
  }

  // 5. Update existing task
  return factory.update(std::move(prev_task_typed), np.infer.input_descs, np.spec.settings, ctx);
}

void Compiler::Impl::instantiate_tasks() {
  auto &g     = out_->graph;
  auto &tasks = out_->resources.tasks;

  // Clear current tasks to be safe, though usually empty at this stage
  tasks.clear();

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &np = g[v];

    if (np.infer.kind == core::TaskKind::Sync) {
      // --- Sync Node Logic ---
      size_t sid    = node_to_section_map_.at(np.spec.name);
      auto   stream = out_->resources.streams.at(sid).get();

      core::SyncCreateCtx ctx{.stream = stream};
      auto               &factory = registry_.get_sync(np.spec.kind);

      // Try to Update, otherwise Create
      auto task = create_or_update_task<core::ISyncTask>(factory, np, ctx);

      // Bind logger if needed (based on your commented code)
      // task->bind_logger(create_task_logger(np.spec.name, np.spec.kind));

      tasks.emplace(np.spec.name, std::move(task));

    } else if (np.infer.kind == core::TaskKind::Async) {
      // --- Async Node Logic ---

      // 1. Find Producer Stream (Incoming Edges)
      void *prod_stream = nullptr;
      for (auto e : boost::make_iterator_range(boost::in_edges(v, g))) {
        auto p = boost::source(e, g);
        if (g[p].infer.kind == core::TaskKind::Sync) {
          size_t sid  = node_to_section_map_.at(g[p].spec.name);
          prod_stream = out_->resources.streams.at(sid).get();
          break;
        }
      }

      // 2. Find Consumer Stream (Outgoing Edges)
      void *cons_stream = nullptr;
      for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
        auto s = boost::target(e, g);
        if (g[s].infer.kind == core::TaskKind::Sync) {
          size_t sid  = node_to_section_map_.at(g[s].spec.name);
          cons_stream = out_->resources.streams.at(sid).get();
          break;
        }
      }

      // 3. Create Context
      core::AsyncCreateCtx ctx{.producer_stream = static_cast<cudaStream_t>(prod_stream),
                               .consumer_stream = static_cast<cudaStream_t>(cons_stream)};

      auto &factory = registry_.get_async(np.spec.kind);

      // Try to Update, otherwise Create
      auto task = create_or_update_task<core::IAsyncTask>(factory, np, ctx);

      // Bind logger if needed
      // task->bind_logger(create_task_logger(np.spec.name, np.spec.kind));

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

  // Iterate via Graph to ensure we match nodes to tasks correctly
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &np = g[v];

    // 1. Retrieve the Task
    auto it_task = res.tasks.find(np.spec.name);
    if (it_task == res.tasks.end())
      continue;
    core::ITask *task = it_task->second.get();

    // 2. Bind Storage Adapter
    if (auto it_adapter = res.node_storage_adapters.find(np.spec.name);
        it_adapter != res.node_storage_adapters.end()) {

      task->bind_storage_access(it_adapter->second.get());
    }

    // 3. Bind Logger
    auto logger = create_task_logger(np.spec.name, np.spec.kind);
    task->bind_logger(std::move(logger));

    // 4. (Future) Bind Event Writers, Profilers, etc.
  }
}

// -------------------------------------------------------------------------------------------------
// Visualization Helpers (Internal)
// -------------------------------------------------------------------------------------------------

namespace {

// Helper to escape strings for DOT labels
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

// Helper to format Tensor Description
std::string format_tdesc(const core::TDesc &d) {
  std::ostringstream ss;
  // Using simple format to avoid heavy JSON dependency inside visualization if possible,
  // but matching your JSON style output:
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
  auto &res      = out_->resources; // Need resources for TID->SID lookup
  auto &sections = out_->sections;

  file << "digraph GraphPlan {\n";
  file << "  rankdir=LR;\n";
  file << "  compound=true;\n";
  file << "  node [fontname=\"Helvetica\", shape=box, style=filled];\n";
  file << "  edge [fontname=\"Helvetica\", fontsize=10];\n\n";

  // Formats "TID(s:SID)" or just "TID" if SID is missing.
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

  // --- 1. Define Nodes ---
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &np = g[v];

    std::ostringstream label_base;
    label_base << (np.spec.name.empty() ? "(unnamed)" : np.spec.name);

    // Format Input IDs
    std::string in_str = "[";
    for (size_t i = 0; i < np.in_tids.size(); ++i) {
      in_str += (i ? "," : "") + fmt_id(np.in_tids[i]);
    }
    in_str += "]";

    // Format Output IDs (detect aliasing)
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
      // Async split nodes.
      // Append "\n" to force left alignment via escape_dot_label.
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
      // Sync nodes.
      const std::string label = label_base.str() + "\n(" + np.spec.kind + ")" + ids_line + "\n";

      file << std::format("  v{} [label=\"{}\", fillcolor=\"#ccffcc\"];\n", v,
                          escape_dot_label(label));
    }
  }

  file << "\n";

  // --- 2. Define Edges ---
  for (auto e : boost::make_iterator_range(boost::edges(g))) {
    const auto  u  = boost::source(e, g);
    const auto  v  = boost::target(e, g);
    const auto &ep = g[e];

    const std::string u_vis = get_visual_id(u, /*is_source=*/true);
    const std::string v_vis = get_visual_id(v, /*is_source=*/false);

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

  // --- 3. Define Sections ---
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