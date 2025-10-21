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

#include "holoflow/runtime/compiler.hh"

#include <algorithm>
#include <array>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <chrono>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <boost/range/iterator_range_core.hpp>

#include "bug.hh"
#include "logger.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow/runtime/graph_exec.hh"

namespace holoflow::runtime {

Compiler::Compiler(core::Registry &registry, const std::filesystem::path &log_dir)
    : registry_(registry), log_dir_(log_dir) {}

std::unique_ptr<CompilerOutput> Compiler::compile(const core::GraphSpec          &gspec,
                                                  std::unique_ptr<CompilerOutput> prev) {
  gspec_ = gspec;
  prev_  = std::move(prev);
  out_   = std::make_unique<CompilerOutput>();
  node_timings_.clear();

  using Clock            = std::chrono::steady_clock;
  const auto total_start = Clock::now();

  std::vector<StepTiming> step_timings;
  step_timings.reserve(14);

  auto measure_step = [&](std::string_view name, auto &&fn) {
    const auto start = Clock::now();
    fn();
    const auto end      = Clock::now();
    const auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    step_timings.push_back(StepTiming{std::string{name}, duration});
  };

  measure_step("check_duplicate_names", [&] { check_duplicate_names(); });
  measure_step("check_duplicate_edge_dst", [&] { check_duplicate_edge_dst(); });
  measure_step("check_factories_registered", [&] { check_factories_registered(); });
  measure_step("check_single_source", [&] { check_single_source(); });
  measure_step("check_single_input", [&] { check_single_input(); });
  measure_step("build_graph_plan", [&] { build_graph_plan(); });
  measure_step("check_typing", [&] { check_typing(); });
  measure_step("assign_tensor_ids", [&] { assign_tensor_ids(); });
  measure_step("check_buffer_temporal_consistency", [&] { check_buffer_temporal_consistency(); });
  measure_step("check_buffer_spatial_consistency", [&] { check_buffer_spatial_consistency(); });
  measure_step("create_tensor_buffers", [&] { create_tensor_buffers(); });
  measure_step("create_sections", [&] { create_sections(); });
  measure_step("assign_cuda_streams", [&] { assign_cuda_streams(); });
  measure_step("create_nodes_collection", [&] { create_nodes_collection(); });

  const double total_duration_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();

  write_profile_report(step_timings, total_duration_ms);

  return std::move(out_);
}

void Compiler::write_profile_report(const std::vector<StepTiming> &step_timings,
                                    double                         total_duration_ms) const {
  if (log_dir_.empty()) {
    return;
  }

  try {
    std::filesystem::create_directories(log_dir_);
  } catch (const std::exception &) {
    return;
  }

  const auto now       = std::chrono::system_clock::now();
  const auto timestamp = std::chrono::floor<std::chrono::seconds>(now);

  std::string filename;
  try {
    filename = std::format("compiler_profile_{:%Y-%m-%d_%H-%M-%S}.log", timestamp);
  } catch (const std::exception &) {
    filename = "compiler_profile.log";
  }

  const auto    filepath = log_dir_ / filename;
  std::ofstream out(filepath, std::ios::trunc);
  if (!out.is_open()) {
    return;
  }

  out << "Compiler Profiling Report\n";
  out << fmt::format("Total duration: {:.3f} ms\n\n", total_duration_ms);
  out << "Steps:\n";
  for (const auto &step : step_timings) {
    out << fmt::format("  - {:<36} {:>10.3f} ms\n", step.name, step.duration_ms);
  }

  if (!node_timings_.empty()) {
    out << "\nNodes:\n";
    for (const auto &node : node_timings_) {
      out << fmt::format("  - {:<36} {:>10.3f} ms\n", node.name, node.duration_ms);
    }
  }
}

void Compiler::check_duplicate_names() {
  std::set<std::string> names;
  for (const auto &v : boost::make_iterator_range(boost::vertices(gspec_))) {
    const auto &ns = gspec_[v];
    if (!names.insert(ns.name).second) {
      throw std::logic_error("Duplicate node name: " + ns.name);
    }
  }
}

void Compiler::check_duplicate_edge_dst() {
  std::set<std::string> dsts;
  for (const auto &e : boost::make_iterator_range(boost::edges(gspec_))) {
    const auto &es    = gspec_[e];
    const auto  dst   = gspec_[boost::target(e, gspec_)];
    auto        label = fmt::format("{}:{}", dst.name, es.out_idx);
    if (!dsts.insert(label).second) {
      throw std::logic_error("Duplicate edge destination: " + label);
    }
  }
}

void Compiler::check_factories_registered() {
  for (const auto &v : boost::make_iterator_range(boost::vertices(gspec_))) {
    const auto &ns = gspec_[v];
    if (!registry_.is_registered(ns.kind)) {
      throw std::logic_error("No factory registered for kind: " + ns.kind);
    }
  }
}

void Compiler::check_single_source() {
  int source_count = 0;
  for (const auto &v : boost::make_iterator_range(boost::vertices(gspec_))) {
    if (boost::in_degree(v, gspec_) == 0) {
      source_count++;
    }
  }
  if (source_count == 0) {
    throw std::logic_error("No source nodes found in the graph");
  }
  if (source_count > 1) {
    throw std::logic_error("Multiple source nodes found in the graph");
  }
}

void Compiler::check_single_input() {
  for (const auto &v : boost::make_iterator_range(boost::vertices(gspec_))) {
    if (boost::in_degree(v, gspec_) > 1) {
      const auto &np = gspec_[v];
      throw std::logic_error("Node " + np.name + " has multiple inputs");
    }
  }
}

void Compiler::build_graph_plan() {
  using VGraphSpec = core::GraphSpec::vertex_descriptor;
  using VGraphPlan = GraphPlan::vertex_descriptor;
  using VMap       = std::map<VGraphSpec, VGraphPlan>;

  auto vertices = boost::make_iterator_range(boost::vertices(gspec_));
  auto edges    = boost::make_iterator_range(boost::edges(gspec_));
  out_->graph   = GraphPlan();
  VMap vmap;

  // Build vertices
  for (const auto &vs : vertices) {
    const auto &ns = gspec_[vs];
    NodePlan    np;
    np.spec = ns;
    auto vp = boost::add_vertex(np, out_->graph);
    vmap.insert({vs, vp});
  }

  // Build edges
  for (const auto &e : edges) {
    const auto &es    = gspec_[e];
    const auto  vsrcp = vmap.at(boost::source(e, gspec_));
    const auto  vdstp = vmap.at(boost::target(e, gspec_));
    EdgePlan    ep;
    ep.spec = es;
    boost::add_edge(vsrcp, vdstp, ep, out_->graph);
  }
}

void Compiler::check_typing() {
  struct CheckTypingVisitor : boost::default_bfs_visitor {
  public:
    CheckTypingVisitor(GraphPlan &g, const core::Registry &reg) : g_(g), reg_(reg) {}

    void discover_vertex(GraphPlan::vertex_descriptor v, const GraphPlan &) {
      auto &np        = g_[v];
      auto  in_degree = boost::in_degree(v, g_);
      auto  in_edges  = boost::in_edges(v, g_);
      // auto  out_degree = boost::out_degree(v, g_);
      auto out_edges = boost::out_edges(v, g_);

      // Gather input descs by input index
      std::vector<core::TDesc> idescs(in_degree);
      for (auto e : boost::make_iterator_range(in_edges)) {
        const auto &ep = g_[e];
        if (ep.spec.in_idx < 0 || ep.spec.in_idx >= static_cast<int>(in_degree)) {
          throw std::logic_error("Invalid in_idx for node " + np.spec.name);
        }
        idescs.at(ep.spec.in_idx) = ep.desc;
      }

      // Call factory
      const auto &factory = reg_.get(np.spec.kind);
      const auto  infer   = factory.infer(idescs, np.spec.settings);
      np.infer            = infer;

      // Assign outgoing edge descs
      // Verify every out_idx in [0, out_degree) is connected
      std::set<int> seen;
      for (auto e : boost::make_iterator_range(out_edges)) {
        auto &ep = g_[e];
        if (ep.spec.out_idx < 0 || ep.spec.out_idx >= static_cast<int>(infer.output_descs.size())) {
          throw std::logic_error("Invalid out_idx (" + std::to_string(ep.spec.out_idx) +
                                 ") for edge between nodes " + np.spec.name + " and " +
                                 g_[boost::target(e, g_)].spec.name);
        }

        ep.desc = infer.output_descs.at(ep.spec.out_idx);
        seen.insert(ep.spec.out_idx);
      }

      // if (seen.size() < out_degree) {
      //   throw std::logic_error("Not all outputs of node " + np.spec.name + " are connected");
      // }
      for (int i = 0; i < static_cast<int>(infer.output_descs.size()); i++) {
        if (!seen.contains(i)) {
          throw std::logic_error("Output " + std::to_string(i) + " of node " + np.spec.name +
                                 " is not connected");
        }
      }
    }

  private:
    GraphPlan            &g_;
    const core::Registry &reg_;
  };

  using ColorMap = std::vector<boost::default_color_type>;
  ColorMap color_map(boost::num_vertices(out_->graph));
  auto     color_map_p = boost::make_iterator_property_map(
      color_map.begin(), boost::get(boost::vertex_index, out_->graph));

  CheckTypingVisitor visitor(out_->graph, registry_);
  for (auto v : boost::make_iterator_range(boost::vertices(out_->graph))) {
    bool is_source = boost::in_degree(v, out_->graph) == 0;
    if (is_source && color_map_p[v] == boost::white_color) {
      boost::queue<GraphPlan::vertex_descriptor> q;
      boost::breadth_first_visit(out_->graph, v, q, visitor, color_map_p);
    }
  }
}

void Compiler::assign_tensor_ids() {
  struct AssignTensorIdsVisitor : boost::default_dfs_visitor {
    AssignTensorIdsVisitor(GraphPlan &g) : g_(g) {}

    void discover_vertex(GraphPlan::vertex_descriptor v, const GraphPlan &) {
      auto &np        = g_[v];
      auto  in_edges  = boost::make_iterator_range(boost::in_edges(v, g_));
      auto  out_edges = boost::make_iterator_range(boost::out_edges(v, g_));

      // Map input tids by input index
      std::vector<int> input_tids(np.infer.input_descs.size(), -1);
      for (auto e : in_edges) {
        const auto &ep                = g_[e];
        input_tids.at(ep.spec.in_idx) = ep.tid;
      }

      // Precompute in-place output tids
      int              n_out = static_cast<int>(np.infer.output_descs.size());
      std::vector<int> inplace_tids(n_out, -1);
      for (const auto &ip : np.infer.in_place) {
        inplace_tids.at(ip.out_idx) = input_tids.at(ip.in_idx);
      }

      // Group outgoing edges by output index
      using E = GraphPlan::edge_descriptor;
      std::vector<std::vector<E>> groups(n_out);
      for (auto e : out_edges) {
        const auto &ep = g_[e];
        groups.at(ep.spec.out_idx).push_back(e);
      }

      np.in_tids = input_tids;
      np.out_tids.assign(n_out, -1);

      // Assign tids per output group
      for (int i = 0; i < n_out; i++) {
        int inplace_tid   = inplace_tids.at(i);
        int tid           = inplace_tid != -1 ? inplace_tid : next_id_++;
        np.out_tids.at(i) = tid;
        for (const auto &e : groups.at(i)) {
          auto &ep = g_[e];
          ep.tid   = tid;
        }
      }
    }

  private:
    GraphPlan &g_;
    int        next_id_ = 0;
  };

  using ColorMap = std::vector<boost::default_color_type>;
  ColorMap color_map(boost::num_vertices(out_->graph));
  auto     color_map_p = boost::make_iterator_property_map(
      color_map.begin(), boost::get(boost::vertex_index, out_->graph));

  AssignTensorIdsVisitor visitor(out_->graph);
  for (auto v : boost::make_iterator_range(boost::vertices(out_->graph))) {
    bool is_source = boost::in_degree(v, out_->graph) == 0;
    if (is_source && color_map_p[v] == boost::white_color) {
      boost::depth_first_visit(out_->graph, v, visitor, color_map_p);
    }
  }
}

void Compiler::check_buffer_temporal_consistency() {
  for (const auto &v : boost::make_iterator_range(boost::vertices(out_->graph))) {
    const auto &np        = out_->graph[v];
    const int   n_out     = static_cast<int>(np.infer.output_descs.size());
    const auto  out_edges = boost::make_iterator_range(boost::out_edges(v, out_->graph));

    // Count temporal restrictions per output index
    std::vector<int> restricted(n_out, 0);
    for (const auto &e : out_edges) {
      const auto &ep  = out_->graph[e];
      const auto &dst = out_->graph[boost::target(e, out_->graph)];

      const bool owned   = dst.infer.owned_inputs.at(ep.spec.in_idx);
      const bool inplace = std::ranges::any_of(
          dst.infer.in_place, [&](const auto &ip) { return ip.in_idx == ep.spec.out_idx; });

      if (owned || inplace) {
        restricted.at(ep.spec.out_idx)++;
      }
    }

    // Check no output is temporally restricted more than once
    for (int i = 0; i < n_out; i++) {
      if (restricted.at(i) > 1) {
        throw std::logic_error(fmt::format(
            "Output {} of node {} is temporally restricted more than once", i, np.spec.name));
      }
    }
  }
}

void Compiler::check_buffer_spatial_consistency() {
  // Count distinct tids
  int nb_tids = 0;
  for (const auto &v : boost::make_iterator_range(boost::vertices(out_->graph))) {
    const auto &np   = out_->graph[v];
    const auto  tids = std::array{std::span{np.in_tids}, std::span{np.out_tids}} | std::views::join;

    if (!tids.empty()) {
      int max = std::ranges::max(tids);
      nb_tids = std::max(nb_tids, max + 1);
    }
  }

  // Count owner per tid
  std::vector<int> tid_owners(nb_tids, 0);
  for (const auto &v : boost::make_iterator_range(boost::vertices(out_->graph))) {
    const auto &np         = out_->graph[v];
    int         nb_inputs  = static_cast<int>(np.infer.input_descs.size());
    int         nb_outputs = static_cast<int>(np.infer.output_descs.size());

    for (int i = 0; i < nb_inputs; i++) {
      if (np.infer.owned_inputs.at(i)) {
        int tid = np.in_tids.at(i);
        tid_owners.at(tid)++;
      }
    }

    for (int i = 0; i < nb_outputs; i++) {
      if (np.infer.owned_outputs.at(i)) {
        int tid = np.out_tids.at(i);
        tid_owners.at(tid)++;
      }
    }
  }

  // Check no tid has more than one owner
  for (int tid = 0; tid < nb_tids; tid++) {
    if (tid_owners.at(tid) > 1) {
      throw std::logic_error(fmt::format("Tensor id {} has more than one owner", tid));
    }
  }
}

void Compiler::create_tensor_buffers() {
  // Find owned tids
  std::set<int> owned_tids;
  for (const auto &v : boost::make_iterator_range(boost::vertices(out_->graph))) {
    const auto &np        = out_->graph[v];
    const int   n_inputs  = static_cast<int>(np.infer.input_descs.size());
    const int   n_outputs = static_cast<int>(np.infer.output_descs.size());

    for (int i = 0; i < n_inputs; i++) {
      if (np.infer.owned_inputs.at(i)) {
        int tid = np.in_tids.at(i);
        owned_tids.insert(tid);
      }
    }

    for (int i = 0; i < n_outputs; i++) {
      if (np.infer.owned_outputs.at(i)) {
        int tid = np.out_tids.at(i);
        owned_tids.insert(tid);
      }
    }
  }

  // Create buffers
  out_->resources.tensors.clear();
  for (const auto &e : boost::make_iterator_range(boost::edges(out_->graph))) {
    const auto &ep = out_->graph[e];
    if (!owned_tids.contains(ep.tid)) {
      out_->resources.tensors.try_emplace(ep.tid, core::Tensor(ep.desc));
    }
  }
}

void Compiler::create_sections() {
  auto is_section_start = [&](GraphPlan::vertex_descriptor v) {
    return boost::in_degree(v, out_->graph) == 0 ||
           out_->graph[v].infer.kind == core::TaskKind::Async;
  };

  // Find section starts
  std::vector<GraphPlan::vertex_descriptor> starts;
  for (const auto &v : boost::make_iterator_range(boost::vertices(out_->graph))) {
    if (is_section_start(v)) {
      starts.push_back(v);
    }
  }

  // DFS to build sections
  auto build_section_dfs = [this](auto self, GraphPlan::vertex_descriptor v, Section &sec,
                                  bool is_root) -> void {
    auto &np = out_->graph[v];
    if (np.infer.kind == core::TaskKind::Sync) {
      sec.sync_topo.push_back(v);
      sync_section_map_.try_emplace(np.spec.name, sec.id);
    } else if (is_root) {
      sec.async_cons.push_back(v);
      async_cons_section_map_.try_emplace(np.spec.name, sec.id);
    } else {
      sec.async_prod.push_back(v);
      async_prod_section_map_.try_emplace(np.spec.name, sec.id);
      return;
    }

    //! BUG: Inplace children should be run last in case several children share
    //! the same inplace input.

    for (auto e : boost::make_iterator_range(boost::out_edges(v, out_->graph))) {
      auto vd = boost::target(e, out_->graph);
      self(self, vd, sec, false);
    }
  };

  // Build sections
  int next_section_id = 0;
  out_->sections.clear();
  for (const auto &v : starts) {
    Section sec;
    sec.id     = next_section_id++;
    sec.name   = fmt::format("section-{}", sec.id);
    sec.stream = 0;

    build_section_dfs(build_section_dfs, v, sec, true);
    out_->sections.push_back(sec);
  }
}

void Compiler::assign_cuda_streams() {
  out_->resources.streams.clear();
  for (auto &sec : out_->sections) {
    auto stream = curaii::CudaStream();
    sec.stream  = stream.get();
    out_->resources.streams.try_emplace(sec.id, std::move(stream));
  }
}

namespace {

template <class To, class From>
std::unique_ptr<To> dynamic_unique_ptr_cast(std::unique_ptr<From> &&ptr) noexcept {
  if (auto casted = dynamic_cast<To *>(ptr.get())) {
    ptr.release();
    return std::unique_ptr<To>(casted);
  }
  return nullptr;
}

template <class Task, class Factory, class Ctx>
std::unique_ptr<Task> create_or_update_task(
    Factory &factory, std::map<std::string, std::unique_ptr<holoflow::core::ITask>> *prev_tasks,
    GraphPlan *prev_graph, NodePlan np, std::span<const core::TDesc> input_descs,
    const nlohmann::json &settings, const Ctx &ctx, std::string_view expected_kind) {

  std::unique_ptr<core::ITask> *prev = nullptr;
  if (prev_tasks) {
    if (auto it = prev_tasks->find(np.spec.name); it != prev_tasks->end())
      prev = &it->second;
  }

  if (!prev || !prev_graph) {
    return factory.create(input_descs, settings, ctx);
  }

  auto [vi, vi_end] = boost::vertices(*prev_graph);
  for (; vi != vi_end; ++vi) {
    const NodePlan &node = (*prev_graph)[*vi];
    if (node.spec.name == np.spec.name && node.spec.kind != np.spec.kind) {
      return factory.create(input_descs, settings, ctx);
    }
  }

  auto prev_task = dynamic_unique_ptr_cast<Task>(std::move(*prev));
  HOLOFLOW_CHECK(prev_task != nullptr, "Previous task for node {} is not a {} task", np.spec.name,
                 expected_kind);

  return factory.update(std::move(prev_task), input_descs, settings, ctx);
}

} // namespace

void Compiler::create_nodes_collection() {
  out_->resources.tasks.clear();
  auto *prev_tasks = prev_ ? &prev_->resources.tasks : nullptr;
  auto *prev_graph = prev_ ? &prev_->graph : nullptr;

  auto &out_tasks = out_->resources.tasks;
  auto  vertices  = boost::make_iterator_range(boost::vertices(out_->graph));

  for (auto v : vertices) {
    const auto &np          = out_->graph[v];
    const auto &input_descs = np.infer.input_descs;
    const auto &settings    = np.spec.settings;
    const auto  name        = std::string_view{np.spec.name};
    const auto  node_start  = std::chrono::steady_clock::now();
    logger()->trace("Creating task for node '{}'", name);

    switch (np.infer.kind) {
    case core::TaskKind::Sync: {
      auto                     &factory = registry_.get_sync(np.spec.kind);
      const int                 sid     = sync_section_map_.at(np.spec.name);
      auto                     &section = out_->sections.at(sid);
      const core::SyncCreateCtx ctx{.stream = section.stream};

      auto task = create_or_update_task<core::ISyncTask>(factory, prev_tasks, prev_graph, np,
                                                         input_descs, settings, ctx, "sync");
      out_tasks.emplace(np.spec.name, std::move(task));
      break;
    }

    case core::TaskKind::Async: {
      auto                      &factory = registry_.get_async(np.spec.kind);
      const int                  prod_id = async_prod_section_map_.at(np.spec.name);
      const int                  cons_id = async_cons_section_map_.at(np.spec.name);
      auto                      &prod    = out_->sections.at(prod_id);
      auto                      &cons    = out_->sections.at(cons_id);
      const core::AsyncCreateCtx ctx{.producer_stream = prod.stream,
                                     .consumer_stream = cons.stream};

      auto task = create_or_update_task<core::IAsyncTask>(factory, prev_tasks, prev_graph, np,
                                                          input_descs, settings, ctx, "async");
      out_tasks.emplace(np.spec.name, std::move(task));
      break;
    }

    default:
      HOLOFLOW_UNREACHABLE();
    }

    const auto node_duration =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - node_start)
            .count();
    node_timings_.push_back(NodeTiming{std::string{name}, node_duration});
  }
}

} // namespace holoflow::runtime