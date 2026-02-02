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
    for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
      auto s = boost::target(e, g);
      if (g[s].infer.kind == core::TaskKind::Sync) {
        out_->sections[get_section_id(s)].async_cons.push_back(v);
      }
    }
    // Producers (Input side of Async Node <- Sync Node)
    for (auto e : boost::make_iterator_range(boost::in_edges(v, g))) {
      auto p = boost::source(e, g);
      if (g[p].infer.kind == core::TaskKind::Sync) {
        out_->sections[get_section_id(p)].async_prod.push_back(v);
      }
    }
  }
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

void Compiler::Impl::instantiate_tasks() {
  auto &g     = out_->graph;
  auto &tasks = out_->resources.tasks;

  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    const auto &np = g[v];

    if (np.infer.kind == core::TaskKind::Sync) {
      // Sync Node: Simple lookup in the node->section map
      size_t sid    = node_to_section_map_.at(np.spec.name);
      auto   stream = out_->resources.streams.at(sid).get();

      core::SyncCreateCtx ctx{.stream = stream};
      auto               &factory = registry_.get_sync(np.spec.kind);
      tasks.emplace(np.spec.name, factory.create(np.infer.input_descs, np.spec.settings, ctx));
    } else if (np.infer.kind == core::TaskKind::Async) {
      // Async Node: Needs streams from both Upstream (Producer) and Downstream (Consumer)

      // 1. Find Producer Stream (from Incoming Edges)
      void *prod_stream = nullptr;
      for (auto e : boost::make_iterator_range(boost::in_edges(v, g))) {
        auto p = boost::source(e, g);
        // We only care about Sync predecessors to identify the section
        if (g[p].infer.kind == core::TaskKind::Sync) {
          size_t sid  = node_to_section_map_.at(g[p].spec.name);
          prod_stream = out_->resources.streams.at(sid).get();
          break; // Found the section; partition logic guarantees all sync inputs form one component
        }
      }

      // 2. Find Consumer Stream (from Outgoing Edges)
      void *cons_stream = nullptr;
      for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
        auto s = boost::target(e, g);
        // We only care about Sync successors to identify the section
        if (g[s].infer.kind == core::TaskKind::Sync) {
          size_t sid  = node_to_section_map_.at(g[s].spec.name);
          cons_stream = out_->resources.streams.at(sid).get();
          break; // Found the section; partition logic guarantees all sync outputs form one
                 // component
        }
      }

      // 3. Create Context
      // Note: Cast raw pointers to cudaStream_t if your ctx expects typed streams,
      // or keep as void* depending on your core::AsyncCreateCtx definition.
      core::AsyncCreateCtx ctx{.producer_stream = static_cast<cudaStream_t>(prod_stream),
                               .consumer_stream = static_cast<cudaStream_t>(cons_stream)};

      auto &factory = registry_.get_async(np.spec.kind);
      tasks.emplace(np.spec.name, factory.create(np.infer.input_descs, np.spec.settings, ctx));
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

// #include "holoflow/runtime/compiler.hh"

// #include <algorithm>
// #include <array>
// #include <boost/graph/breadth_first_search.hpp>
// #include <boost/graph/depth_first_search.hpp>
// #include <boost/graph/topological_sort.hpp>
// #include <boost/range/iterator_range_core.hpp>
// #include <chrono>
// #include <exception>
// #include <filesystem>
// #include <format>
// #include <fstream>
// #include <memory>
// #include <spdlog/fmt/ranges.h>
// #include <spdlog/sinks/stdout_color_sinks.h>
// #include <string>
// #include <string_view>
// #include <vector>

// #include "bug.hh"
// #include "curaii/cuda.hh"
// #include "holoflow/core/graph_spec.hh"
// #include "holoflow/core/tasks.hh"
// #include "holoflow/core/tensor.hh"
// #include "holoflow/runtime/graph_exec.hh"
// #include "logger.hh"

// namespace holoflow::runtime {

// Compiler::Compiler(core::Registry &registry, const std::filesystem::path &log_dir)
//     : registry_(registry), log_dir_(log_dir) {}

// std::unique_ptr<CompilerOutput> Compiler::compile(const core::GraphSpec          &gspec,
//                                                   std::unique_ptr<CompilerOutput> prev) {
//   gspec_ = gspec;
//   prev_  = std::move(prev);
//   out_   = std::make_unique<CompilerOutput>();
//   node_timings_.clear();

//   using Clock            = std::chrono::steady_clock;
//   const auto total_start = Clock::now();

//   std::vector<StepTiming> step_timings;
//   step_timings.reserve(14);

//   auto measure_step = [&](std::string_view name, auto &&fn) {
//     const auto start = Clock::now();
//     fn();
//     const auto end      = Clock::now();
//     const auto duration = std::chrono::duration<double, std::milli>(end - start).count();
//     step_timings.push_back(StepTiming{std::string{name}, duration});
//   };

//   measure_step("check_duplicate_names", [&] { check_duplicate_names(); });
//   measure_step("check_duplicate_edge_dst", [&] { check_duplicate_edge_dst(); });
//   measure_step("check_factories_registered", [&] { check_factories_registered(); });
//   measure_step("build_graph_plan", [&] { build_graph_plan(); });
//   measure_step("check_typing", [&] { check_typing(); });
//   measure_step("assign_tensor_ids", [&] { assign_tensor_ids(); });
//   measure_step("check_buffer_temporal_consistency", [&] { check_buffer_temporal_consistency();
//   }); measure_step("check_buffer_spatial_consistency", [&] { check_buffer_spatial_consistency();
//   }); measure_step("create_tensor_buffers", [&] { create_tensor_buffers(); });
//   measure_step("create_sections", [&] { create_sections(); });
//   measure_step("assign_cuda_streams", [&] { assign_cuda_streams(); });
//   measure_step("create_nodes_collection", [&] { create_nodes_collection(); });

//   const double total_duration_ms =
//       std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();

//   write_profile_report(step_timings, total_duration_ms);

//   return std::move(out_);
// }

// void Compiler::write_profile_report(const std::vector<StepTiming> &step_timings,
//                                     double                         total_duration_ms) const {
//   if (log_dir_.empty()) {
//     return;
//   }

//   try {
//     std::filesystem::create_directories(log_dir_);
//   } catch (const std::exception &) {
//     return;
//   }

//   const auto now       = std::chrono::system_clock::now();
//   const auto timestamp = std::chrono::floor<std::chrono::seconds>(now);

//   std::string filename;
//   try {
//     filename = std::format("compiler_profile_{:%Y-%m-%d_%H-%M-%S}.log", timestamp);
//   } catch (const std::exception &) {
//     filename = "compiler_profile.log";
//   }

//   const auto    filepath = log_dir_ / filename;
//   std::ofstream out(filepath, std::ios::trunc);
//   if (!out.is_open()) {
//     return;
//   }

//   out << "Compiler Profiling Report\n";
//   out << fmt::format("Total duration: {:.3f} ms\n\n", total_duration_ms);
//   out << "Steps:\n";
//   for (const auto &step : step_timings) {
//     out << fmt::format("  - {:<36} {:>10.3f} ms\n", step.name, step.duration_ms);
//   }

//   if (!node_timings_.empty()) {
//     out << "\nNodes:\n";
//     for (const auto &node : node_timings_) {
//       out << fmt::format("  - {:<36} {:>10.3f} ms\n", node.name, node.duration_ms);
//     }
//   }
// }

// void Compiler::check_duplicate_names() {
//   std::set<std::string> names;
//   for (const auto &v : boost::make_iterator_range(boost::vertices(gspec_))) {
//     const auto &ns = gspec_[v];
//     if (!names.insert(ns.name).second) {
//       throw std::logic_error("Duplicate node name: " + ns.name);
//     }
//   }
// }

// void Compiler::check_duplicate_edge_dst() {
//   std::set<std::string> dsts;
//   for (const auto &e : boost::make_iterator_range(boost::edges(gspec_))) {
//     const auto &es    = gspec_[e];
//     const auto  dst   = gspec_[boost::target(e, gspec_)];
//     auto        label = fmt::format("{}:{}", dst.name, es.in_idx);
//     if (!dsts.insert(label).second) {
//       throw std::logic_error("Duplicate edge destination: " + label);
//     }
//   }
// }

// void Compiler::check_factories_registered() {
//   for (const auto &v : boost::make_iterator_range(boost::vertices(gspec_))) {
//     const auto &ns = gspec_[v];
//     if (!registry_.is_registered(ns.kind)) {
//       throw std::logic_error("No factory registered for kind: " + ns.kind);
//     }
//   }
// }

// void Compiler::build_graph_plan() {
//   using VGraphSpec = core::GraphSpec::vertex_descriptor;
//   using VGraphPlan = GraphPlan::vertex_descriptor;
//   using VMap       = std::map<VGraphSpec, VGraphPlan>;

//   auto vertices = boost::make_iterator_range(boost::vertices(gspec_));
//   auto edges    = boost::make_iterator_range(boost::edges(gspec_));
//   out_->graph   = GraphPlan();
//   VMap vmap;

//   // Build vertices
//   for (const auto &vs : vertices) {
//     const auto &ns = gspec_[vs];
//     NodePlan    np;
//     np.spec = ns;
//     auto vp = boost::add_vertex(np, out_->graph);
//     vmap.insert({vs, vp});
//   }

//   // Build edges
//   for (const auto &e : edges) {
//     const auto &es    = gspec_[e];
//     const auto  vsrcp = vmap.at(boost::source(e, gspec_));
//     const auto  vdstp = vmap.at(boost::target(e, gspec_));
//     EdgePlan    ep;
//     ep.spec = es;
//     boost::add_edge(vsrcp, vdstp, ep, out_->graph);
//   }
// }

// void Compiler::check_typing() {
//   auto &g = out_->graph;

//   using V = GraphPlan::vertex_descriptor;

//   // 1) Compute topo order (sources -> sinks)
//   std::vector<V> topo;
//   topo.reserve(boost::num_vertices(g));

//   try {
//     boost::topological_sort(g, std::back_inserter(topo)); // reverse topo: sinks -> sources
//   } catch (const boost::not_a_dag &) {
//     throw std::logic_error("Graph is not a DAG (cycle detected)");
//   }

//   std::reverse(topo.begin(), topo.end()); // now sources -> sinks

//   // 2) Process vertices in topo order
//   for (V v : topo) {
//     auto &np = g[v];

//     // Incoming edges
//     auto in_edges = boost::in_edges(v, g);

//     // Determine input count. Your original code sized by in_degree.
//     // This assumes every input slot has exactly one incoming edge and all are connected.
//     const auto in_degree = boost::in_degree(v, g);

//     std::vector<core::TDesc> idescs(in_degree);

//     // Fill by in_idx
//     for (auto e : boost::make_iterator_range(in_edges)) {
//       const auto &ep = g[e];

//       if (ep.spec.in_idx < 0 || ep.spec.in_idx >= static_cast<int>(in_degree)) {
//         throw std::logic_error("Invalid in_idx for node " + np.spec.name);
//       }

//       // At this point, because of topo order, ep.desc MUST already be set by the producer node.
//       // If you have a "null/invalid" state for TDesc, validate it here.
//       idescs.at(ep.spec.in_idx) = ep.desc;
//     }

//     // Optional: detect missing input slots (if TDesc can be default-constructed)
//     // e.g., if (idescs[i].is_invalid()) throw ...

//     // Infer node typing
//     const auto &factory = registry_.get(np.spec.kind);
//     const auto  infer   = factory.infer(idescs, np.spec.settings);
//     np.infer            = infer;

//     // Assign outgoing edge descs + verify all outputs are connected
//     auto          out_edges = boost::out_edges(v, g);
//     std::set<int> seen;

//     for (auto e : boost::make_iterator_range(out_edges)) {
//       auto &ep = g[e];

//       if (ep.spec.out_idx < 0 || ep.spec.out_idx >= static_cast<int>(infer.output_descs.size()))
//       {
//         throw std::logic_error("Invalid out_idx (" + std::to_string(ep.spec.out_idx) +
//                                ") for edge between nodes " + np.spec.name + " and " +
//                                g[boost::target(e, g)].spec.name);
//       }

//       ep.desc = infer.output_descs.at(ep.spec.out_idx);
//       seen.insert(ep.spec.out_idx);
//     }

//     // TODO: is this check necessary? What if the user wants to ignore some outputs?
//     // for (int i = 0; i < static_cast<int>(infer.output_descs.size()); ++i) {
//     //   if (!seen.contains(i)) {
//     //     throw std::logic_error("Output " + std::to_string(i) + " of node " + np.spec.name +
//     //                            " is not connected");
//     //   }
//     // }
//   }
// }

// // void Compiler::check_typing() {
// //   struct CheckTypingVisitor : boost::default_bfs_visitor {
// //   public:
// //     CheckTypingVisitor(GraphPlan &g, const core::Registry &reg) : g_(g), reg_(reg) {}

// //     void discover_vertex(GraphPlan::vertex_descriptor v, const GraphPlan &) {
// //       auto &np        = g_[v];
// //       auto  in_degree = boost::in_degree(v, g_);
// //       auto  in_edges  = boost::in_edges(v, g_);
// //       // auto  out_degree = boost::out_degree(v, g_);
// //       auto out_edges = boost::out_edges(v, g_);

// //       // Gather input descs by input index
// //       std::vector<core::TDesc> idescs(in_degree);
// //       for (auto e : boost::make_iterator_range(in_edges)) {
// //         const auto &ep = g_[e];
// //         if (ep.spec.in_idx < 0 || ep.spec.in_idx >= static_cast<int>(in_degree)) {
// //           throw std::logic_error("Invalid in_idx for node " + np.spec.name);
// //         }
// //         idescs.at(ep.spec.in_idx) = ep.desc;
// //       }

// //       // Call factory
// //       const auto &factory = reg_.get(np.spec.kind);
// //       const auto  infer   = factory.infer(idescs, np.spec.settings);
// //       np.infer            = infer;

// //       // Assign outgoing edge descs
// //       // Verify every out_idx in [0, out_degree) is connected
// //       std::set<int> seen;
// //       for (auto e : boost::make_iterator_range(out_edges)) {
// //         auto &ep = g_[e];
// //         if (ep.spec.out_idx < 0 || ep.spec.out_idx >=
// //         static_cast<int>(infer.output_descs.size())) {
// //           throw std::logic_error("Invalid out_idx (" + std::to_string(ep.spec.out_idx) +
// //                                  ") for edge between nodes " + np.spec.name + " and " +
// //                                  g_[boost::target(e, g_)].spec.name);
// //         }

// //         ep.desc = infer.output_descs.at(ep.spec.out_idx);
// //         seen.insert(ep.spec.out_idx);
// //       }

// //       // if (seen.size() < out_degree) {
// //       //   throw std::logic_error("Not all outputs of node " + np.spec.name + " are
// connected");
// //       // }
// //       for (int i = 0; i < static_cast<int>(infer.output_descs.size()); i++) {
// //         if (!seen.contains(i)) {
// //           throw std::logic_error("Output " + std::to_string(i) + " of node " + np.spec.name +
// //                                  " is not connected");
// //         }
// //       }
// //     }

// //   private:
// //     GraphPlan            &g_;
// //     const core::Registry &reg_;
// //   };

// //   using ColorMap = std::vector<boost::default_color_type>;
// //   ColorMap color_map(boost::num_vertices(out_->graph));
// //   auto     color_map_p = boost::make_iterator_property_map(
// //       color_map.begin(), boost::get(boost::vertex_index, out_->graph));

// //   CheckTypingVisitor visitor(out_->graph, registry_);
// //   for (auto v : boost::make_iterator_range(boost::vertices(out_->graph))) {
// //     bool is_source = boost::in_degree(v, out_->graph) == 0;
// //     if (is_source && color_map_p[v] == boost::white_color) {
// //       boost::queue<GraphPlan::vertex_descriptor> q;
// //       boost::breadth_first_visit(out_->graph, v, q, visitor, color_map_p);
// //     }
// //   }
// // }

// void Compiler::assign_tensor_ids() {
//   auto &g = out_->graph;
//   using V = GraphPlan::vertex_descriptor;
//   using E = GraphPlan::edge_descriptor;

//   // 1) Topological order (sources -> sinks)
//   std::vector<V> topo;
//   topo.reserve(boost::num_vertices(g));

//   try {
//     boost::topological_sort(g, std::back_inserter(topo)); // reverse topo
//   } catch (const boost::not_a_dag &) {
//     throw std::logic_error("Graph is not a DAG (cycle detected)");
//   }
//   std::reverse(topo.begin(), topo.end());

//   // 2) Assign tensor ids in topo order
//   int next_id = 0;

//   for (V v : topo) {
//     auto &np = g[v];

//     auto in_edges  = boost::make_iterator_range(boost::in_edges(v, g));
//     auto out_edges = boost::make_iterator_range(boost::out_edges(v, g));

//     // Map input tids by input index
//     std::vector<int> input_tids(np.infer.input_descs.size(), -1);
//     for (auto e : in_edges) {
//       const auto &ep = g[e];

//       if (ep.spec.in_idx < 0 || ep.spec.in_idx >= static_cast<int>(input_tids.size())) {
//         throw std::logic_error("Invalid in_idx for node " + np.spec.name);
//       }

//       // In topo order, producer must have assigned ep.tid already.
//       if (ep.tid < 0) {
//         throw std::logic_error("Unassigned input tid for node " + np.spec.name + " (input " +
//                                std::to_string(ep.spec.in_idx) + ")");
//       }

//       input_tids.at(ep.spec.in_idx) = ep.tid;
//     }

//     // Optional: ensure all required inputs are connected
//     for (int i = 0; i < static_cast<int>(input_tids.size()); ++i) {
//       if (input_tids[i] < 0) {
//         throw std::logic_error("Missing input " + std::to_string(i) + " for node " +
//         np.spec.name);
//       }
//     }

//     // Precompute in-place output tids
//     const int        n_out = static_cast<int>(np.infer.output_descs.size());
//     std::vector<int> inplace_tids(n_out, -1);
//     for (const auto &ip : np.infer.in_place) {
//       if (ip.in_idx < 0 || ip.in_idx >= static_cast<int>(input_tids.size()) || ip.out_idx < 0 ||
//           ip.out_idx >= n_out) {
//         throw std::logic_error("Invalid in_place mapping for node " + np.spec.name);
//       }
//       inplace_tids.at(ip.out_idx) = input_tids.at(ip.in_idx);
//     }

//     // Group outgoing edges by output index
//     std::vector<std::vector<E>> groups(n_out);
//     for (auto e : out_edges) {
//       const auto &ep = g[e];
//       if (ep.spec.out_idx < 0 || ep.spec.out_idx >= n_out) {
//         throw std::logic_error("Invalid out_idx for node " + np.spec.name);
//       }
//       groups.at(ep.spec.out_idx).push_back(e);
//     }

//     np.in_tids = input_tids;
//     np.out_tids.assign(n_out, -1);

//     // Assign tids per output group
//     for (int i = 0; i < n_out; ++i) {
//       const int tid     = (inplace_tids[i] != -1) ? inplace_tids[i] : next_id++;
//       np.out_tids.at(i) = tid;

//       for (E e : groups.at(i)) {
//         g[e].tid = tid;
//       }
//     }
//   }
// }

// // void Compiler::assign_tensor_ids() {
// //   struct AssignTensorIdsVisitor : boost::default_dfs_visitor {
// //     AssignTensorIdsVisitor(GraphPlan &g) : g_(g) {}

// //     void discover_vertex(GraphPlan::vertex_descriptor v, const GraphPlan &) {
// //       auto &np        = g_[v];
// //       auto  in_edges  = boost::make_iterator_range(boost::in_edges(v, g_));
// //       auto  out_edges = boost::make_iterator_range(boost::out_edges(v, g_));

// //       // Map input tids by input index
// //       std::vector<int> input_tids(np.infer.input_descs.size(), -1);
// //       for (auto e : in_edges) {
// //         const auto &ep                = g_[e];
// //         input_tids.at(ep.spec.in_idx) = ep.tid;
// //       }

// //       // Precompute in-place output tids
// //       int              n_out = static_cast<int>(np.infer.output_descs.size());
// //       std::vector<int> inplace_tids(n_out, -1);
// //       for (const auto &ip : np.infer.in_place) {
// //         inplace_tids.at(ip.out_idx) = input_tids.at(ip.in_idx);
// //       }

// //       // Group outgoing edges by output index
// //       using E = GraphPlan::edge_descriptor;
// //       std::vector<std::vector<E>> groups(n_out);
// //       for (auto e : out_edges) {
// //         const auto &ep = g_[e];
// //         groups.at(ep.spec.out_idx).push_back(e);
// //       }

// //       np.in_tids = input_tids;
// //       np.out_tids.assign(n_out, -1);

// //       // Assign tids per output group
// //       for (int i = 0; i < n_out; i++) {
// //         int inplace_tid   = inplace_tids.at(i);
// //         int tid           = inplace_tid != -1 ? inplace_tid : next_id_++;
// //         np.out_tids.at(i) = tid;
// //         for (const auto &e : groups.at(i)) {
// //           auto &ep = g_[e];
// //           ep.tid   = tid;
// //         }
// //       }
// //     }

// //   private:
// //     GraphPlan &g_;
// //     int        next_id_ = 0;
// //   };

// //   using ColorMap = std::vector<boost::default_color_type>;
// //   ColorMap color_map(boost::num_vertices(out_->graph));
// //   auto     color_map_p = boost::make_iterator_property_map(
// //       color_map.begin(), boost::get(boost::vertex_index, out_->graph));

// //   AssignTensorIdsVisitor visitor(out_->graph);
// //   for (auto v : boost::make_iterator_range(boost::vertices(out_->graph))) {
// //     bool is_source = boost::in_degree(v, out_->graph) == 0;
// //     if (is_source && color_map_p[v] == boost::white_color) {
// //       boost::depth_first_visit(out_->graph, v, visitor, color_map_p);
// //     }
// //   }
// // }

// void Compiler::check_buffer_temporal_consistency() {
//   for (const auto &v : boost::make_iterator_range(boost::vertices(out_->graph))) {
//     const auto &np        = out_->graph[v];
//     const int   n_out     = static_cast<int>(np.infer.output_descs.size());
//     const auto  out_edges = boost::make_iterator_range(boost::out_edges(v, out_->graph));

//     // Count temporal restrictions per output index
//     std::vector<int> restricted(n_out, 0);
//     for (const auto &e : out_edges) {
//       const auto &ep  = out_->graph[e];
//       const auto &dst = out_->graph[boost::target(e, out_->graph)];

//       const bool owned   = dst.infer.owned_inputs.at(ep.spec.in_idx);
//       const bool inplace = std::ranges::any_of(
//           dst.infer.in_place, [&](const auto &ip) { return ip.in_idx == ep.spec.out_idx; });

//       if (owned || inplace) {
//         restricted.at(ep.spec.out_idx)++;
//       }
//     }

//     // Check no output is temporally restricted more than once
//     for (int i = 0; i < n_out; i++) {
//       if (restricted.at(i) > 1) {
//         throw std::logic_error(fmt::format(
//             "Output {} of node {} is temporally restricted more than once", i, np.spec.name));
//       }
//     }
//   }
// }

// void Compiler::check_buffer_spatial_consistency() {
//   // Count distinct tids
//   int nb_tids = 0;
//   for (const auto &v : boost::make_iterator_range(boost::vertices(out_->graph))) {
//     const auto &np   = out_->graph[v];
//     const auto  tids = std::array{std::span{np.in_tids}, std::span{np.out_tids}} |
//     std::views::join;

//     if (!tids.empty()) {
//       int max = std::ranges::max(tids);
//       nb_tids = std::max(nb_tids, max + 1);
//     }
//   }

//   // Count owner per tid
//   std::vector<int> tid_owners(nb_tids, 0);
//   for (const auto &v : boost::make_iterator_range(boost::vertices(out_->graph))) {
//     const auto &np         = out_->graph[v];
//     int         nb_inputs  = static_cast<int>(np.infer.input_descs.size());
//     int         nb_outputs = static_cast<int>(np.infer.output_descs.size());

//     for (int i = 0; i < nb_inputs; i++) {
//       if (np.infer.owned_inputs.at(i)) {
//         int tid = np.in_tids.at(i);
//         tid_owners.at(tid)++;
//       }
//     }

//     for (int i = 0; i < nb_outputs; i++) {
//       if (np.infer.owned_outputs.at(i)) {
//         int tid = np.out_tids.at(i);
//         tid_owners.at(tid)++;
//       }
//     }
//   }

//   // Check no tid has more than one owner
//   for (int tid = 0; tid < nb_tids; tid++) {
//     if (tid_owners.at(tid) > 1) {
//       throw std::logic_error(fmt::format("Tensor id {} has more than one owner", tid));
//     }
//   }
// }

// void Compiler::create_tensor_buffers() {
//   // Find owned tids
//   std::set<int> owned_tids;
//   for (const auto &v : boost::make_iterator_range(boost::vertices(out_->graph))) {
//     const auto &np        = out_->graph[v];
//     const int   n_inputs  = static_cast<int>(np.infer.input_descs.size());
//     const int   n_outputs = static_cast<int>(np.infer.output_descs.size());

//     for (int i = 0; i < n_inputs; i++) {
//       if (np.infer.owned_inputs.at(i)) {
//         int tid = np.in_tids.at(i);
//         owned_tids.insert(tid);
//       }
//     }

//     for (int i = 0; i < n_outputs; i++) {
//       if (np.infer.owned_outputs.at(i)) {
//         int tid = np.out_tids.at(i);
//         owned_tids.insert(tid);
//       }
//     }
//   }

//   // Create buffers
//   out_->resources.tensors.clear();
//   for (const auto &e : boost::make_iterator_range(boost::edges(out_->graph))) {
//     const auto &ep = out_->graph[e];
//     if (!owned_tids.contains(ep.tid)) {
//       out_->resources.tensors.try_emplace(ep.tid, core::Tensor(ep.desc));
//     }
//   }
// }

// void Compiler::create_sections() {
//   using V = GraphPlan::vertex_descriptor;

//   auto is_async = [&](V v) { return out_->graph[v].infer.kind == core::TaskKind::Async; };
//   auto is_sync  = [&](V v) { return out_->graph[v].infer.kind == core::TaskKind::Sync; };

//   // --- Build undirected "bounce graph" over Sync nodes ---
//   std::unordered_map<V, std::vector<V>> adj;
//   adj.reserve(boost::num_vertices(out_->graph));

//   auto add_undirected = [&](V a, V b) {
//     adj[a].push_back(b);
//     adj[b].push_back(a);
//   };

//   // 1) Bounce on Sync<->Sync edges (ignore any edge touching a wall)
//   for (auto e : boost::make_iterator_range(boost::edges(out_->graph))) {
//     const V u = boost::source(e, out_->graph);
//     const V v = boost::target(e, out_->graph);
//     if (is_sync(u) && is_sync(v)) {
//       add_undirected(u, v);
//     }
//   }

//   // 2) For each wall, connect all Sync inputs together AND all Sync outputs together
//   auto clique = [&](const std::vector<V> &xs) {
//     for (std::size_t i = 0; i < xs.size(); ++i)
//       for (std::size_t j = i + 1; j < xs.size(); ++j)
//         add_undirected(xs[i], xs[j]);
//   };

//   for (auto w : boost::make_iterator_range(boost::vertices(out_->graph))) {
//     if (!is_async(w))
//       continue;

//     std::vector<V> preds;
//     for (auto e : boost::make_iterator_range(boost::in_edges(w, out_->graph))) {
//       const V p = boost::source(e, out_->graph);
//       if (is_sync(p))
//         preds.push_back(p);
//     }

//     std::vector<V> succs;
//     for (auto e : boost::make_iterator_range(boost::out_edges(w, out_->graph))) {
//       const V s = boost::target(e, out_->graph);
//       if (is_sync(s))
//         succs.push_back(s);
//     }

//     // Optional small dedup to avoid quadratic duplicates
//     auto dedup = [](std::vector<V> &v) {
//       std::sort(v.begin(), v.end());
//       v.erase(std::unique(v.begin(), v.end()), v.end());
//     };
//     dedup(preds);
//     dedup(succs);

//     clique(preds); // inputs connected
//     clique(succs); // outputs connected
//   }

//   // dedup adjacency lists (optional)
//   for (auto &[v, ns] : adj) {
//     std::sort(ns.begin(), ns.end());
//     ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
//   }

//   // --- Connected components on derived undirected graph (Sync only) ---
//   std::unordered_map<V, int> comp;
//   comp.reserve(boost::num_vertices(out_->graph));

//   int            next_comp = 0;
//   std::vector<V> stack;
//   stack.reserve(256);

//   for (auto v : boost::make_iterator_range(boost::vertices(out_->graph))) {
//     if (!is_sync(v))
//       continue;
//     if (comp.contains(v))
//       continue;

//     const int cid = next_comp++;
//     comp.emplace(v, cid);
//     stack.push_back(v);

//     while (!stack.empty()) {
//       const V cur = stack.back();
//       stack.pop_back();

//       auto it = adj.find(cur);
//       if (it == adj.end())
//         continue;

//       for (V nb : it->second) {
//         if (!comp.contains(nb)) {
//           comp.emplace(nb, cid);
//           stack.push_back(nb);
//         }
//       }
//     }
//   }

//   // --- Initialize sections ---
//   out_->sections.clear();
//   out_->sections.resize(next_comp);
//   for (int cid = 0; cid < next_comp; ++cid) {
//     auto &sec  = out_->sections[cid];
//     sec.id     = cid;
//     sec.name   = fmt::format("section-{}", cid);
//     sec.stream = 0;
//   }

//   // Global topo, then filter Sync nodes into their section to keep executable order
//   std::vector<V> topo;
//   topo.reserve(boost::num_vertices(out_->graph));
//   boost::topological_sort(out_->graph, std::back_inserter(topo));
//   std::reverse(topo.begin(), topo.end());

//   sync_section_map_.clear();
//   async_cons_section_map_.clear();
//   async_prod_section_map_.clear();

//   for (V v : topo) {
//     if (!is_sync(v))
//       continue;
//     const int cid = comp.at(v);
//     out_->sections[cid].sync_topo.push_back(v);
//     sync_section_map_.try_emplace(out_->graph[v].spec.name, cid);
//   }

//   // Helpers: component of any Sync predecessor/successor
//   auto comp_of_any_sync_pred = [&](V a) -> std::optional<int> {
//     for (auto e : boost::make_iterator_range(boost::in_edges(a, out_->graph))) {
//       const V p = boost::source(e, out_->graph);
//       if (is_sync(p))
//         return comp.at(p);
//     }
//     return std::nullopt;
//   };
//   auto comp_of_any_sync_succ = [&](V a) -> std::optional<int> {
//     for (auto e : boost::make_iterator_range(boost::out_edges(a, out_->graph))) {
//       const V s = boost::target(e, out_->graph);
//       if (is_sync(s))
//         return comp.at(s);
//     }
//     return std::nullopt;
//   };

//   // --- Attach wall halves (consumer to input section, producer to output section) ---
//   for (auto w : boost::make_iterator_range(boost::vertices(out_->graph))) {
//     if (!is_async(w))
//       continue;
//     auto &np = out_->graph[w];

//     // Consumer belongs to OUTPUT section (upstream)
//     if (auto in_sec = comp_of_any_sync_succ(w)) {
//       out_->sections[*in_sec].async_cons.push_back(w);
//       async_cons_section_map_.try_emplace(np.spec.name, *in_sec);
//     }

//     // Producer belongs to INPUT section (downstream)
//     if (auto out_sec = comp_of_any_sync_pred(w)) {
//       out_->sections[*out_sec].async_prod.push_back(w);
//       async_prod_section_map_.try_emplace(np.spec.name, *out_sec);
//     }

//     // Degenerate: wall with no sync preds/succs => choose policy; here: own section as consumer
//     if (!comp_of_any_sync_pred(w) && !comp_of_any_sync_succ(w)) {
//       Section sec;
//       sec.id     = static_cast<int>(out_->sections.size());
//       sec.name   = fmt::format("section-{}", sec.id);
//       sec.stream = 0;
//       sec.async_cons.push_back(w);
//       async_cons_section_map_.try_emplace(np.spec.name, sec.id);
//       out_->sections.push_back(std::move(sec));
//     }
//   }
// }

// /*
// void Compiler::create_sections() {
//   auto is_section_start = [&](GraphPlan::vertex_descriptor v) {
//     return boost::in_degree(v, out_->graph) == 0 ||
//            out_->graph[v].infer.kind == core::TaskKind::Async;
//   };

//   // Find section starts
//   std::vector<GraphPlan::vertex_descriptor> starts;
//   for (const auto &v : boost::make_iterator_range(boost::vertices(out_->graph))) {
//     if (is_section_start(v)) {
//       starts.push_back(v);
//     }
//   }

//   // DFS to build sections
//   auto build_section_dfs = [this](auto self, GraphPlan::vertex_descriptor v, Section &sec,
//                                   bool is_root) -> void {
//     auto &np = out_->graph[v];
//     if (np.infer.kind == core::TaskKind::Sync) {
//       sec.sync_topo.push_back(v);
//       sync_section_map_.try_emplace(np.spec.name, sec.id);
//     } else if (is_root) {
//       sec.async_cons.push_back(v);
//       async_cons_section_map_.try_emplace(np.spec.name, sec.id);
//     } else {
//       sec.async_prod.push_back(v);
//       async_prod_section_map_.try_emplace(np.spec.name, sec.id);
//       return;
//     }

//     //! BUG: Inplace children should be run last in case several children share
//     //! the same inplace input.

//     for (auto e : boost::make_iterator_range(boost::out_edges(v, out_->graph))) {
//       auto vd = boost::target(e, out_->graph);
//       self(self, vd, sec, false);
//     }
//   };

//   // Build sections
//   int next_section_id = 0;
//   out_->sections.clear();
//   for (const auto &v : starts) {
//     Section sec;
//     sec.id     = next_section_id++;
//     sec.name   = fmt::format("section-{}", sec.id);
//     sec.stream = 0;

//     build_section_dfs(build_section_dfs, v, sec, true);
//     out_->sections.push_back(sec);
//   }
// }
// */

// void Compiler::assign_cuda_streams() {
//   out_->resources.streams.clear();
//   for (auto &sec : out_->sections) {
//     auto stream = curaii::CudaStream();
//     sec.stream  = stream.get();
//     out_->resources.streams.try_emplace(sec.id, std::move(stream));
//   }
// }

// namespace {

// template <class To, class From>
// std::unique_ptr<To> dynamic_unique_ptr_cast(std::unique_ptr<From> &&ptr) noexcept {
//   if (auto casted = dynamic_cast<To *>(ptr.get())) {
//     ptr.release();
//     return std::unique_ptr<To>(casted);
//   }
//   return nullptr;
// }

// template <class Task, class Factory, class Ctx>
// std::unique_ptr<Task> create_or_update_task(
//     Factory &factory, std::map<std::string, std::unique_ptr<holoflow::core::ITask>> *prev_tasks,
//     GraphPlan *prev_graph, NodePlan np, std::span<const core::TDesc> input_descs,
//     const nlohmann::json &settings, const Ctx &ctx, std::string_view expected_kind) {

//   std::unique_ptr<core::ITask> *prev = nullptr;
//   if (prev_tasks) {
//     if (auto it = prev_tasks->find(np.spec.name); it != prev_tasks->end())
//       prev = &it->second;
//   }

//   if (!prev || !prev_graph) {
//     return factory.create(input_descs, settings, ctx);
//   }

//   auto [vi, vi_end] = boost::vertices(*prev_graph);
//   for (; vi != vi_end; ++vi) {
//     const NodePlan &node = (*prev_graph)[*vi];
//     if (node.spec.name == np.spec.name && node.spec.kind != np.spec.kind) {
//       return factory.create(input_descs, settings, ctx);
//     }
//   }

//   auto prev_task = dynamic_unique_ptr_cast<Task>(std::move(*prev));
//   HOLOFLOW_CHECK(prev_task != nullptr, "Previous task for node {} is not a {} task",
//   np.spec.name,
//                  expected_kind);

//   return factory.update(std::move(prev_task), input_descs, settings, ctx);
// }

// std::shared_ptr<spdlog::logger> create_task_logger(const std::string &node_name,
//                                                    const std::string &node_kind) {
//   auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
//   sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [thread %t] [%^%l%$] %v");
//   auto logger_name = fmt::format("TaskLogger-{}-{}", node_kind, node_name);
//   auto logger      = std::make_shared<spdlog::logger>(logger_name, sink);
//   logger->set_level(spdlog::default_logger()->level());
//   return logger;
// }

// } // namespace

// void Compiler::create_nodes_collection() {
//   out_->resources.tasks.clear();
//   auto *prev_tasks = prev_ ? &prev_->resources.tasks : nullptr;
//   auto *prev_graph = prev_ ? &prev_->graph : nullptr;

//   auto &out_tasks = out_->resources.tasks;
//   auto  vertices  = boost::make_iterator_range(boost::vertices(out_->graph));

//   for (auto v : vertices) {
//     const auto &np          = out_->graph[v];
//     const auto &input_descs = np.infer.input_descs;
//     const auto &settings    = np.spec.settings;
//     const auto  name        = std::string_view{np.spec.name};
//     const auto  node_start  = std::chrono::steady_clock::now();
//     logger()->trace("Creating task for node '{}'", name);

//     logger()->debug("sync_section_map_: {}", fmt::join(sync_section_map_, ", "));
//     logger()->debug("async_prod_section_map_: {}", fmt::join(async_prod_section_map_, ", "));
//     logger()->debug("async_cons_section_map_: {}", fmt::join(async_cons_section_map_, ", "));

//     switch (np.infer.kind) {
//     case core::TaskKind::Sync: {
//       auto                     &factory = registry_.get_sync(np.spec.kind);
//       const int                 sid     = sync_section_map_.at(np.spec.name);
//       auto                     &section = out_->sections.at(sid);
//       const core::SyncCreateCtx ctx{.stream = section.stream};

//       auto task = create_or_update_task<core::ISyncTask>(factory, prev_tasks, prev_graph, np,
//                                                          input_descs, settings, ctx, "sync");

//       task->bind_logger(create_task_logger(np.spec.name, np.spec.kind));
//       out_tasks.emplace(np.spec.name, std::move(task));
//       break;
//     }

//     case core::TaskKind::Async: {
//       auto                      &factory = registry_.get_async(np.spec.kind);
//       const int                  prod_id = async_prod_section_map_.at(np.spec.name);
//       const int                  cons_id = async_cons_section_map_.at(np.spec.name);
//       auto                      &prod    = out_->sections.at(prod_id);
//       auto                      &cons    = out_->sections.at(cons_id);
//       const core::AsyncCreateCtx ctx{.producer_stream = prod.stream,
//                                      .consumer_stream = cons.stream};

//       auto task = create_or_update_task<core::IAsyncTask>(factory, prev_tasks, prev_graph, np,
//                                                           input_descs, settings, ctx, "async");
//       task->bind_logger(create_task_logger(np.spec.name, np.spec.kind));
//       out_tasks.emplace(np.spec.name, std::move(task));
//       break;
//     }

//     default:
//       HOLOFLOW_UNREACHABLE();
//     }

//     const auto node_duration =
//         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - node_start)
//             .count();
//     node_timings_.push_back(NodeTiming{std::string{name}, node_duration});
//   }
// }

// } // namespace holoflow::runtime