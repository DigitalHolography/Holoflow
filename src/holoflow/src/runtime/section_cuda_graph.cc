// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "section_cuda_graph.hh"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

#include "../logger.hh"

namespace holoflow::runtime {

SectionCudaGraphs::~SectionCudaGraphs() { clear(); }

void SectionCudaGraphs::clear() noexcept {
  for (auto executable : executables)
    if (executable)
      CUDA_CHECK_NT(cudaGraphExecDestroy(executable));
  executables.clear();
  executable_node_counts.clear();
  tuple_indices.clear();
  node_count = 0;
  enabled    = false;
}

std::optional<size_t>
SectionCudaGraphs::variant(const std::map<size_t, std::unique_ptr<core::Storage>> &storages) const {
  PointerTuple tuple;
  tuple.reserve(storage_ids.size());
  for (auto sid : storage_ids)
    tuple.push_back(reinterpret_cast<uintptr_t>(storages.at(sid)->ptr));
  auto found = tuple_indices.find(tuple);
  return found == tuple_indices.end() ? std::nullopt : std::optional<size_t>{found->second};
}

nlohmann::json SectionCudaGraphs::snapshot() const {
  std::lock_guard lock(diagnostic_mutex);
  auto            result        = diagnostics;
  result["launches"]            = launches.load(std::memory_order_relaxed);
  result["ordinary_iterations"] = ordinary_iterations.load(std::memory_order_relaxed);
  result["pointer_misses"]      = pointer_misses.load(std::memory_order_relaxed);
  result["tuple_misses"]        = tuple_misses.load(std::memory_order_relaxed);
  return result;
}

void SectionCudaGraphs::report_miss(
    const std::map<size_t, std::unique_ptr<core::Storage>> &storages) {
  auto bindings = nlohmann::json::array();
  bool unknown  = false;
  for (size_t i = 0; i < storage_ids.size(); ++i) {
    auto       ptr   = storages.at(storage_ids[i])->ptr;
    auto       it    = std::find(pointers[i].begin(), pointers[i].end(), ptr);
    const bool known = it != pointers[i].end();
    unknown |= !known;
    bindings.push_back({{"storage_id", storage_ids[i]},
                        {"known_pointer", known},
                        {"pointer_index", known ? nlohmann::json(it - pointers[i].begin())
                                                : nlohmann::json(nullptr)}});
    logger()->debug("[CUDA graphs] miss storage {} address {}", storage_ids[i],
                    static_cast<void *>(ptr));
  }
  (unknown ? pointer_misses : tuple_misses).fetch_add(1, std::memory_order_relaxed);
  enabled         = false;
  fallback_reason = unknown ? "Runtime pointer is outside its declared domain"
                            : "Runtime pointer tuple is outside the prepared combinations";
  std::lock_guard lock(diagnostic_mutex);
  diagnostics["enabled"]         = false;
  diagnostics["status"]          = "fallback";
  diagnostics["failure_stage"]   = "lookup";
  diagnostics["fallback_reason"] = fallback_reason;
  diagnostics["miss_bindings"]   = std::move(bindings);
  logger()->warn("[CUDA graphs] {}: {}; bindings {}", diagnostics["section"].get<std::string>(),
                 fallback_reason, diagnostics["miss_bindings"].dump());
}

namespace {

constexpr size_t planning_budget = 1'000'000;

struct GraphContext {
  const GraphPlan            &graph;
  const std::vector<Section> &sections;
  ExecResouces               &resources;
};

struct Owner {
  core::ITask          *task;
  std::string           name;
  size_t                port;
  bool                  input;
  std::optional<size_t> count;
  std::string           error;
};

std::map<size_t, Owner> storage_owners(GraphContext &out) {
  std::map<size_t, Owner> owners;
  for (auto v : boost::make_iterator_range(boost::vertices(out.graph))) {
    const auto &np = out.graph[v];
    for (bool input : {true, false}) {
      const auto &mask = input ? np.infer.owned_inputs : np.infer.owned_outputs;
      const auto &counts =
          input ? np.infer.owned_input_pointer_counts : np.infer.owned_output_pointer_counts;
      const auto &tids      = input ? np.in_tids : np.out_tids;
      const bool  malformed = !counts.empty() && counts.size() != tids.size();
      for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i])
          owners.emplace(out.resources.tid_to_sid.at(tids[i]),
                         Owner{out.resources.tasks.at(np.spec.name).get(), np.spec.name, i, input,
                               counts.empty() || malformed ? std::nullopt : counts[i],
                               malformed ? "Pointer count vector has wrong size" : ""});
    }
  }
  return owners;
}

struct SectionPlan {
  std::vector<size_t>                               sids, counts;
  std::vector<std::optional<core::PointerSequence>> sequences;
  std::vector<std::vector<size_t>>                  tuples;
  std::vector<std::vector<std::byte *>>             pointers;
  nlohmann::json                                    report;
  std::string                                       fatal;
  void block(const std::string &message) { report["blockers"].push_back(message); }
  void invalid(const std::string &message) {
    block(message);
    if (fatal.empty())
      fatal = message;
  }
  bool eligible() const { return report["blockers"].empty(); }
};

std::optional<size_t> checked_product(std::optional<size_t> value, size_t factor) {
  if (!value || (factor && *value > (std::numeric_limits<size_t>::max)() / factor))
    return std::nullopt;
  return *value * factor;
}

// Only finite declared sequences are simulated, never tasks or queue acquisition.
void plan_tuples(SectionPlan &plan, size_t limit) {
  size_t prefix = 0, period = 1;
  bool   bounded = true;
  for (const auto &sequence : plan.sequences) {
    if (!sequence)
      continue;
    prefix              = std::max(prefix, sequence->prefix.size());
    const size_t factor = sequence->cycle.size() / std::gcd(period, sequence->cycle.size());
    if (factor > planning_budget / period) {
      bounded = false;
      break;
    }
    period *= factor;
  }
  bounded &=
      prefix <= planning_budget && period <= planning_budget - std::min(prefix, planning_budget);
  if (!bounded)
    plan.report["planning_note"] =
        "Sequence planning budget exceeded; using conservative Cartesian domains";
  plan.report["planning_mode"]   = bounded ? "sequence" : "cartesian";
  plan.report["planning_budget"] = planning_budget;
  if (bounded) {
    plan.report["startup_steps"] = prefix;
    plan.report["period"]        = period;
  }

  std::vector<size_t> unordered;
  size_t              unordered_product = 1;
  for (size_t i = 0; i < plan.sids.size(); ++i) {
    if (bounded && plan.sequences[i])
      continue;
    unordered.push_back(i);
    if (plan.counts[i] > limit / unordered_product) {
      plan.block("Unspecified pointer product exceeds graph limit");
      plan.report["pruned_count_status"] = "exceeds_limit";
      return;
    }
    unordered_product *= plan.counts[i];
  }
  std::set<std::vector<size_t>> ordered;
  const size_t                  steps = bounded ? prefix + period : 1;
  for (size_t step = 0; step < steps; ++step) {
    std::vector<size_t> tuple(plan.sids.size(), 0);
    if (bounded)
      for (size_t i = 0; i < tuple.size(); ++i) {
        if (!plan.sequences[i])
          continue;
        const auto &s = *plan.sequences[i];
        tuple[i]      = step < s.prefix.size() ? s.prefix[step]
                                               : s.cycle[(step - s.prefix.size()) % s.cycle.size()];
      }
    ordered.insert(std::move(tuple));
    if (ordered.size() > limit / unordered_product) {
      plan.block("Reachable pointer combinations exceed graph limit");
      plan.report["pruned_count_status"] = "exceeds_limit";
      return;
    }
  }
  for (const auto &base : ordered) {
    for (size_t combination = 0; combination < unordered_product; ++combination) {
      auto   tuple     = base;
      size_t remaining = combination;
      for (auto i : unordered) {
        tuple[i] = remaining % plan.counts[i];
        remaining /= plan.counts[i];
      }
      plan.tuples.push_back(std::move(tuple));
    }
  }
  plan.report["pruned_count"]        = plan.tuples.size();
  plan.report["pruned_count_status"] = "exact";
}

SectionPlan inspect_section(GraphContext &out, const Section &sec,
                            const std::map<size_t, Owner> &owners) {
  SectionPlan plan;
  auto       &j = plan.report;
  j             = {{"section", sec.name},
                   {"section_id", sec.id},
                   {"enabled", false},
                   {"limit", out.resources.max_section_cuda_graphs},
                   {"status", "inspected"},
                   {"blockers", nlohmann::json::array()},
                   {"tasks", nlohmann::json::array()},
                   {"domains", nlohmann::json::array()},
                   {"raw_cartesian_count", nullptr},
                   {"pruned_count", nullptr},
                   {"pruned_count_status", "not_planned"},
                   {"failure_stage", ""},
                   {"fallback_reason", ""},
                   {"variants", 0},
                   {"node_count", 0},
                   {"reused", 0},
                   {"created", 0},
                   {"discarded", 0}};
  std::set<size_t> sids;
  for (auto v : sec.sync_topo) {
    const auto &np   = out.graph[v];
    auto       *task = static_cast<core::ISyncTask *>(out.resources.tasks.at(np.spec.name).get());
    const bool  supported   = task->supports_cuda_graph();
    const bool  owns_output = std::ranges::any_of(np.infer.owned_outputs, [](bool b) { return b; });
    j["tasks"].push_back({{"name", np.spec.name},
                          {"kind", np.spec.kind},
                          {"supports_cuda_graph", supported},
                          {"owns_output", owns_output}});
    if (!supported || owns_output)
      plan.block(np.spec.name + " requires ordinary execution");
    for (auto tid : np.in_tids)
      sids.insert(out.resources.tid_to_sid.at(tid));
    for (auto tid : np.out_tids)
      sids.insert(out.resources.tid_to_sid.at(tid));
  }
  std::optional<size_t> raw     = 1;
  bool                  unknown = false;
  for (auto sid : sids) {
    const auto owner = owners.find(sid);
    const auto count = owner == owners.end() ? std::optional<size_t>{1} : owner->second.count;
    auto       domain =
        nlohmann::json{{"storage_id", sid},
                       {"tensor_ids", nlohmann::json::array()},
                       {"declared_count", count ? nlohmann::json(*count) : nlohmann::json(nullptr)},
                       {"enumerated_count", nullptr},
                       {"enumeration", "skipped"},
                       {"sequence", "unspecified"}};
    for (const auto &[tid, storage_id] : out.resources.tid_to_sid)
      if (storage_id == sid)
        domain["tensor_ids"].push_back(tid);
    std::optional<core::PointerSequence> sequence;
    if (owner == owners.end()) {
      domain["owner"] = "compiler";
      sequence        = core::PointerSequence{{}, {0}};
    } else {
      const auto &o       = owner->second;
      domain["owner"]     = o.name;
      domain["port"]      = o.port;
      domain["direction"] = o.input ? "input" : "output";
      try {
        if (!o.error.empty())
          throw std::invalid_argument(o.error);
        sequence = o.input ? o.task->owned_input_pointer_sequence(o.port)
                           : o.task->owned_output_pointer_sequence(o.port);
      } catch (const std::exception &e) {
        domain["error"] = e.what();
        plan.invalid(std::format("{} storage {}: {}", o.name, sid, e.what()));
      }
    }
    if (!count) {
      unknown = true;
      plan.block(std::format("Storage {} has no declared pointer count", sid));
    } else if (*count == 0) {
      domain["error"] = "Zero pointer count";
      plan.invalid(std::format("Storage {} declares zero pointers", sid));
    } else {
      raw = checked_product(raw, *count);
    }
    if (sequence) {
      domain["sequence"]   = "exact";
      domain["prefix"]     = sequence->prefix;
      domain["cycle"]      = sequence->cycle;
      const auto bad_index = [&](size_t index) { return count && index >= *count; };
      if (sequence->cycle.empty() || std::ranges::any_of(sequence->prefix, bad_index) ||
          std::ranges::any_of(sequence->cycle, bad_index)) {
        domain["error"] = "Empty cycle or out-of-domain sequence index";
        plan.invalid(std::format("Storage {} has invalid pointer sequence", sid));
      }
    }
    logger()->info("[CUDA graphs] {} storage {} owner {}: declared {}, order {}", sec.name, sid,
                   domain["owner"].get<std::string>(), domain["declared_count"].dump(),
                   domain["sequence"].get<std::string>());
    j["domains"].push_back(std::move(domain));
    plan.sids.push_back(sid);
    plan.counts.push_back(count.value_or(0));
    plan.sequences.push_back(std::move(sequence));
  }
  j["raw_count_status"] = unknown ? "unknown" : raw ? "exact" : "overflow";
  if (!unknown && raw)
    j["raw_cartesian_count"] = *raw;
  if (sec.sync_topo.empty())
    plan.block("No synchronous tasks");
  if (out.resources.max_section_cuda_graphs == 0)
    plan.block("Section graphs disabled");
  if (!unknown && plan.fatal.empty() && !sids.empty() && out.resources.max_section_cuda_graphs != 0)
    plan_tuples(plan, out.resources.max_section_cuda_graphs);

  if (plan.eligible()) {
    for (size_t i = 0; i < plan.sids.size(); ++i) {
      const auto                              sid    = plan.sids[i];
      auto                                   &domain = j["domains"][i];
      std::optional<std::vector<std::byte *>> pointers;
      try {
        const auto owner = owners.find(sid);
        if (owner == owners.end()) {
          auto ptr = out.resources.storages.at(sid)->ptr;
          if (ptr)
            pointers = std::vector{ptr};
        } else {
          const auto &o = owner->second;
          pointers      = o.input ? o.task->owned_input_pointers(o.port)
                                  : o.task->owned_output_pointers(o.port);
        }
        domain["enumeration"] = pointers ? "returned" : "unknown";
        if (!pointers) {
          plan.block(std::format("Storage {} has no pointer enumeration", sid));
          plan.pointers.emplace_back();
          continue;
        }
        domain["enumerated_count"] = pointers->size();
        if (pointers->size() != plan.counts[i])
          throw std::invalid_argument("Pointer count mismatch");
        std::set<std::byte *> unique;
        for (auto ptr : *pointers) {
          if (!ptr || !unique.insert(ptr).second)
            throw std::invalid_argument("Null or duplicate pointer");
          logger()->trace("[CUDA graphs] {} storage {} address {}", sec.name, sid,
                          static_cast<void *>(ptr));
        }
        plan.pointers.push_back(std::move(*pointers));
      } catch (const std::exception &e) {
        domain["enumeration"] = "invalid";
        domain["error"]       = e.what();
        plan.invalid(std::format("Storage {}: {}", sid, e.what()));
        plan.pointers.emplace_back();
      }
    }
  }
  j["status"] = plan.eligible() ? "planned" : plan.fatal.empty() ? "fallback" : "error";
  if (!plan.eligible()) {
    j["failure_stage"]   = "inspection";
    j["fallback_reason"] = j["blockers"][0];
  }
  return plan;
}

// Conditional body graphs are part of the conditional node, not embedded task executables.
size_t validate_flat_graph(cudaGraph_t graph) {
  size_t count = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
  size_t total = count;
  for (auto node : nodes) {
    cudaGraphNodeType type;
    CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeGraph) {
      throw std::runtime_error("Recording contains an embedded child graph");
    }
    if (type == cudaGraphNodeTypeConditional) {
      cudaGraphNodeParams params{};
      CUDA_CHECK(cudaGraphNodeGetParams(node, &params));
      for (unsigned int i = 0; i < params.conditional.size; ++i) {
        total += validate_flat_graph(params.conditional.phGraph_out[i]);
      }
    }
  }
  return total;
}

void record_variant(GraphContext &out, const Section &sec, SectionCudaGraphs &set,
                    const SectionCudaGraphs::PointerTuple &tuple, nlohmann::json &report) {
  std::map<size_t, core::Storage> bindings;
  for (size_t i = set.storage_ids.size(); i-- > 0;) {
    auto sid = set.storage_ids[i];
    bindings.emplace(sid, *out.resources.storages.at(sid));
    bindings.at(sid).ptr = reinterpret_cast<std::byte *>(tuple[i]);
  }
  auto views = [&](const std::vector<int> &tids) {
    std::vector<core::TView> result;
    for (auto tid : tids) {
      result.push_back(
          {out.resources.tensor_descs.at(tid), &bindings.at(out.resources.tid_to_sid.at(tid))});
    }
    return result;
  };

  cudaGraph_t     graph      = nullptr;
  cudaGraphExec_t executable = nullptr;
  bool            capturing  = false;
  try {
    report["failure_stage"] = "capture_begin";
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(sec.stream, graph, nullptr, nullptr, 0,
                                             cudaStreamCaptureModeThreadLocal));
    capturing = true;
    for (auto v : sec.sync_topo) {
      const auto        &np      = out.graph[v];
      auto               inputs  = views(np.in_tids);
      auto               outputs = views(np.out_tids);
      core::CudaGraphCtx ctx{inputs, outputs, sec.stream, graph};
      report["failure_stage"]  = "record";
      report["recording_task"] = np.spec.name;
      try {
        static_cast<core::ISyncTask *>(out.resources.tasks.at(np.spec.name).get())
            ->record_cuda_graph(ctx);
      } catch (const std::exception &e) {
        throw std::runtime_error(std::format("{}: {}", np.spec.name, e.what()));
      }
    }
    report.erase("recording_task");
    report["failure_stage"] = "capture_end";
    cudaGraph_t captured    = nullptr;
    const auto  end_result  = cudaStreamEndCapture(sec.stream, &captured);
    capturing               = false;
    // EndCapture transfers the graph back, or destroys it when capture is invalidated.
    graph = captured;
    CUDA_CHECK(end_result);
    report["failure_stage"] = "validate_graph";
    const auto nodes        = validate_flat_graph(graph);
    report["failure_stage"] = "instantiate";
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    set.executables.push_back(executable);
    executable = nullptr;
    set.executable_node_counts.push_back(nodes);
    set.node_count += nodes;
    CUDA_CHECK_NT(cudaGraphDestroy(graph));
  } catch (...) {
    if (capturing) {
      cudaGraph_t captured = nullptr;
      (void)cudaStreamEndCapture(sec.stream, &captured);
      graph = captured;
    }
    if (executable)
      CUDA_CHECK_NT(cudaGraphExecDestroy(executable));
    if (graph)
      CUDA_CHECK_NT(cudaGraphDestroy(graph));
    throw;
  }
}

void prepare_section(GraphContext &out, const Section &sec, SectionPlan &plan,
                     SectionCudaGraphs &set) {
  auto             &j             = plan.report;
  const auto        started       = std::chrono::steady_clock::now();
  const size_t      previous_size = set.executables.size();
  size_t            reused = 0, created = 0, discarded = 0;
  SectionCudaGraphs next;
  next.storage_ids = plan.sids;
  next.pointers    = plan.pointers;
  try {
    if (plan.eligible()) {
      std::vector<SectionCudaGraphs::PointerTuple> desired;
      std::set<SectionCudaGraphs::PointerTuple>    desired_set;
      for (const auto &indices : plan.tuples) {
        SectionCudaGraphs::PointerTuple tuple;
        for (size_t i = 0; i < indices.size(); ++i)
          tuple.push_back(reinterpret_cast<uintptr_t>(plan.pointers[i][indices[i]]));
        desired_set.insert(tuple);
        desired.push_back(std::move(tuple));
      }
      // Release obsolete variants first so a refresh does not double the executable budget.
      for (const auto &[tuple, index] : set.tuple_indices) {
        if (!desired_set.contains(tuple) || set.storage_ids != next.storage_ids) {
          CUDA_CHECK_NT(cudaGraphExecDestroy(set.executables[index]));
          set.executables[index] = nullptr;
          ++discarded;
        }
      }
      next.executables.reserve(desired.size());
      next.executable_node_counts.reserve(desired.size());
      for (const auto &tuple : desired) {
        next.tuple_indices.emplace(tuple, next.executables.size());
        auto old = set.tuple_indices.find(tuple);
        if (old != set.tuple_indices.end() && set.executables[old->second]) {
          next.executables.push_back(set.executables[old->second]);
          set.executables[old->second] = nullptr;
          next.executable_node_counts.push_back(set.executable_node_counts[old->second]);
          next.node_count += set.executable_node_counts[old->second];
          ++reused;
        } else {
          j["variant_index"] = next.executables.size();
          record_variant(out, sec, next, tuple, j);
          ++created;
        }
      }
      j["failure_stage"] = "";
      j.erase("variant_index");
      next.enabled = true;
    }
  } catch (const std::exception &e) {
    plan.block(e.what());
    j["status"]          = "fallback";
    j["fallback_reason"] = e.what();
    next.clear();
    set.clear();
    // A poisoned context cannot use ordinary execution.
    try {
      CUDA_CHECK(cudaStreamSynchronize(sec.stream));
    } catch (...) {
      j["status"]        = "error";
      j["failure_stage"] = "cuda_context";
      throw;
    }
    (void)cudaGetLastError();
  }
  if (!next.enabled)
    discarded = previous_size + created;
  set.clear();
  set.storage_ids            = std::move(next.storage_ids);
  set.pointers               = std::move(next.pointers);
  set.tuple_indices          = std::move(next.tuple_indices);
  set.executables            = std::move(next.executables);
  set.executable_node_counts = std::move(next.executable_node_counts);
  set.node_count             = next.node_count;
  set.enabled                = next.enabled;
  set.variant_count          = plan.tuples.size();
  set.fallback_reason        = j["fallback_reason"].get<std::string>();
  set.construction_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  j["enabled"] = set.enabled;
  if (set.enabled)
    j["status"] = "ready";
  j["variants"]        = set.executables.size();
  j["node_count"]      = set.node_count;
  j["reused"]          = reused;
  j["created"]         = created;
  j["discarded"]       = discarded;
  j["construction_ms"] = set.construction_ms;
}

} // namespace

void write_section_cuda_graph_diagnostics(const ExecResouces &resources) {
  if (resources.section_cuda_graph_log_dir.empty())
    return;
  auto diagnostics = nlohmann::json::array();
  for (const auto &[id, graphs] : resources.section_cuda_graphs)
    diagnostics.push_back(graphs->snapshot());
  std::ofstream file(resources.section_cuda_graph_log_dir / "section_cuda_graphs.json");
  if (file)
    file << diagnostics.dump(2);
  else
    logger()->warn("[CUDA graphs] Could not write section_cuda_graphs.json");
}

void refresh_section_cuda_graphs(const GraphPlan &graph, const std::vector<Section> &sections,
                                 ExecResouces &resources, bool instantiate) {
  GraphContext             out{graph, sections, resources};
  const auto               owners = storage_owners(out);
  std::vector<SectionPlan> plans;
  std::string              fatal;
  for (const auto &sec : sections) {
    const auto started = std::chrono::steady_clock::now();
    plans.push_back(inspect_section(out, sec, owners));
    auto &plan = plans.back();
    if (!plan.fatal.empty() && fatal.empty())
      fatal = plan.fatal;
    auto &owned = resources.section_cuda_graphs[sec.id];
    if (!owned)
      owned = std::make_unique<SectionCudaGraphs>();
    auto &set = *owned;
    if (instantiate)
      ++set.refresh_count;
    plan.report["refresh_count"]                  = set.refresh_count;
    plan.report["cached_variants_before_refresh"] = set.executables.size();
    plan.report["inspection_ms"] =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    set.enabled         = false;
    set.fallback_reason = plan.report["fallback_reason"].get<std::string>();
    std::lock_guard lock(set.diagnostic_mutex);
    set.diagnostics = plan.report;
  }
  write_section_cuda_graph_diagnostics(resources);
  if (!fatal.empty())
    throw std::invalid_argument(fatal);
  for (size_t i = 0; i < sections.size(); ++i) {
    const auto &sec  = sections[i];
    auto       &set  = *resources.section_cuda_graphs.at(sec.id);
    auto       &plan = plans[i];
    try {
      if (instantiate)
        prepare_section(out, sec, plan, set);
    } catch (...) {
      {
        std::lock_guard lock(set.diagnostic_mutex);
        set.diagnostics = plan.report;
      }
      write_section_cuda_graph_diagnostics(resources);
      throw;
    }
    plan.report["preparation_ms"] =
        plan.report["inspection_ms"].get<double>() + plan.report.value("construction_ms", 0.0);
    {
      std::lock_guard lock(set.diagnostic_mutex);
      set.diagnostics = plan.report;
    }
    logger()->info(
        "[CUDA graphs] {}: raw {}, pruned {}, built {}, reused {}, {} ms; {}; blockers {}",
        sec.name, plan.report["raw_cartesian_count"].dump(), plan.report["pruned_count"].dump(),
        set.executables.size(), plan.report["reused"].dump(), plan.report["preparation_ms"].dump(),
        plan.report["status"].get<std::string>(), plan.report["blockers"].dump());
  }
  write_section_cuda_graph_diagnostics(resources);
}

} // namespace holoflow::runtime
