// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "section_cuda_graph.hh"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../logger.hh"

namespace holoflow::runtime {

namespace {

using StorageMap = std::map<size_t, std::unique_ptr<core::Storage>>;

// Materialize the current runtime binding of a section as the key used to select a prepared graph
// variant. The ordering must match SectionCudaGraphs::storage_ids exactly.
SectionCudaGraphs::PointerTuple current_pointer_tuple(const std::vector<size_t> &storage_ids,
                                                      const StorageMap          &storages) {
  SectionCudaGraphs::PointerTuple tuple;
  tuple.reserve(storage_ids.size());

  for (const size_t storage_id : storage_ids)
    tuple.push_back(reinterpret_cast<uintptr_t>(storages.at(storage_id)->ptr));

  return tuple;
}

} // namespace

SectionCudaGraphs::~SectionCudaGraphs() { clear(); }

// Destroy all instantiated CUDA graph executables owned by this section.
//
// Pointer-domain metadata is intentionally left intact: callers that refresh a section may still
// need it while transferring reusable state into the replacement cache. The owning object will
// release the remaining containers normally when it is destroyed.
void SectionCudaGraphs::clear() noexcept {
  for (auto executable : executables) {
    if (executable)
      CUDA_CHECK_NT(cudaGraphExecDestroy(executable));
  }

  executables.clear();
  executable_node_counts.clear();
  tuple_indices.clear();
  node_count = 0;
  enabled    = false;
}

// Return the prepared graph variant matching the section's current storage addresses.
std::optional<size_t>
SectionCudaGraphs::variant(const std::map<size_t, std::unique_ptr<core::Storage>> &storages) const {
  const auto tuple = current_pointer_tuple(storage_ids, storages);
  const auto found = tuple_indices.find(tuple);

  if (found == tuple_indices.end())
    return std::nullopt;

  return found->second;
}

// Return a thread-safe diagnostics snapshot augmented with runtime counters.
nlohmann::json SectionCudaGraphs::snapshot() const {
  std::lock_guard lock(diagnostic_mutex);

  auto result                   = diagnostics;
  result["launches"]            = launches.load(std::memory_order_relaxed);
  result["ordinary_iterations"] = ordinary_iterations.load(std::memory_order_relaxed);
  result["pointer_misses"]      = pointer_misses.load(std::memory_order_relaxed);
  result["tuple_misses"]        = tuple_misses.load(std::memory_order_relaxed);
  return result;
}

// Disable graph execution after a runtime binding violates the pointer-domain assumptions used
// during planning.
//
// A pointer miss means at least one storage address was never declared. A tuple miss means every
// individual address is known, but their combination was not among the states considered reachable
// by the planner.
void SectionCudaGraphs::report_miss(
    const std::map<size_t, std::unique_ptr<core::Storage>> &storages) {
  auto bindings            = nlohmann::json::array();
  bool has_unknown_pointer = false;

  for (size_t i = 0; i < storage_ids.size(); ++i) {
    const auto storage_id = storage_ids[i];
    auto      *ptr        = storages.at(storage_id)->ptr;
    const auto found      = std::find(pointers[i].begin(), pointers[i].end(), ptr);
    const bool known      = found != pointers[i].end();

    has_unknown_pointer |= !known;
    bindings.push_back({{"storage_id", storage_id},
                        {"known_pointer", known},
                        {"pointer_index", known ? nlohmann::json(found - pointers[i].begin())
                                                : nlohmann::json(nullptr)}});

    logger()->debug("[CUDA graphs] miss storage {} address {}", storage_id,
                    static_cast<void *>(ptr));
  }

  (has_unknown_pointer ? pointer_misses : tuple_misses).fetch_add(1, std::memory_order_relaxed);

  enabled         = false;
  fallback_reason = has_unknown_pointer
                        ? "Runtime pointer is outside its declared domain"
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

// Hard upper bound on sequence simulation work during graph planning. If an exact joint pointer
// cycle would require more steps, planning falls back to conservative Cartesian enumeration.
constexpr size_t planning_budget = 1'000'000;

struct GraphContext {
  const GraphPlan &graph;
  ExecResouces    &resources;
};

enum class PortDirection { input, output };

// Identifies the task and port responsible for a task-owned storage.
//
// `pointer_count` comes from graph inference and is required for finite graph planning. `error`
// carries structural inference errors so inspection can report them in the same diagnostics path as
// task-provided pointer metadata.
struct StorageOwner {
  core::ITask          *task;
  std::string           task_name;
  size_t                port;
  PortDirection         direction;
  std::optional<size_t> pointer_count;
  std::string           error;

  [[nodiscard]] bool is_input() const { return direction == PortDirection::input; }

  [[nodiscard]] std::optional<core::PointerSequence> pointer_sequence() const {
    return is_input() ? task->owned_input_pointer_sequence(port)
                      : task->owned_output_pointer_sequence(port);
  }

  [[nodiscard]] std::optional<std::vector<std::byte *>> enumerate_pointers() const {
    return is_input() ? task->owned_input_pointers(port) : task->owned_output_pointers(port);
  }
};

using StorageOwners = std::map<size_t, StorageOwner>;

// Planning information for one storage used by a section.
//
// A storage domain is deliberately kept as one object rather than four parallel vectors: the
// storage id, declared cardinality, temporal sequence and concrete pointer enumeration are one
// invariant and should evolve together.
struct StorageDomain {
  size_t                               storage_id    = 0;
  size_t                               pointer_count = 0;
  std::optional<core::PointerSequence> sequence;
  std::vector<std::byte *>             pointers;
};

using IndexTuple    = std::vector<size_t>;
using IndexTupleSet = std::set<IndexTuple>;

// Complete preparation plan for one execution section.
//
// `tuples` contains pointer *indices*, not addresses. Concrete addresses are resolved only during
// preparation after every storage domain has been validated and enumerated.
struct SectionPlan {
  std::vector<StorageDomain> domains;
  std::vector<IndexTuple>    tuples;
  nlohmann::json             report;
  std::string                fatal;

  void block(const std::string &message) { report["blockers"].push_back(message); }

  void invalid(const std::string &message) {
    block(message);
    if (fatal.empty())
      fatal = message;
  }

  [[nodiscard]] bool eligible() const { return report["blockers"].empty(); }
};

// Return lhs * rhs unless the product overflows size_t or a previous product already overflowed.
std::optional<size_t> checked_product(std::optional<size_t> value, size_t factor) {
  if (!value)
    return std::nullopt;

  if (factor != 0 && *value > (std::numeric_limits<size_t>::max)() / factor)
    return std::nullopt;

  return *value * factor;
}

// Collect the task-owned storages declared by graph inference.
//
// Compiler-owned storages do not appear in this map and are treated as single-pointer domains later
// during section inspection.
StorageOwners collect_storage_owners(GraphContext &context) {
  StorageOwners owners;

  for (const auto vertex : boost::make_iterator_range(boost::vertices(context.graph))) {
    const auto &node = context.graph[vertex];

    for (const bool input : {true, false}) {
      const auto  direction = input ? PortDirection::input : PortDirection::output;
      const auto &owned     = input ? node.infer.owned_inputs : node.infer.owned_outputs;
      const auto &counts =
          input ? node.infer.owned_input_pointer_counts : node.infer.owned_output_pointer_counts;
      const auto &tids = input ? node.in_tids : node.out_tids;

      const bool malformed_counts = !counts.empty() && counts.size() != tids.size();

      for (size_t port = 0; port < owned.size(); ++port) {
        if (!owned[port])
          continue;

        const auto storage_id = context.resources.tid_to_sid.at(tids[port]);
        owners.emplace(storage_id,
                       StorageOwner{context.resources.tasks.at(node.spec.name).get(),
                                    node.spec.name, port, direction,
                                    counts.empty() || malformed_counts
                                        ? std::nullopt
                                        : std::optional<size_t>{counts[port]},
                                    malformed_counts ? "Pointer count vector has wrong size" : ""});
      }
    }
  }

  return owners;
}

// Describes the finite interval that must be simulated to enumerate every reachable joint state of
// the declared pointer sequences.
//
// Each PointerSequence contains a finite, non-repeating prefix followed by an infinitely repeating
// cycle. For several sequences, all joint states are observed after:
//
//   max(prefix lengths) + lcm(cycle lengths)
//
// steps. If this horizon exceeds planning_budget, exact sequence planning is abandoned and all
// pointer domains are conservatively expanded as a Cartesian product.
struct SequenceHorizon {
  bool   exact         = true;
  size_t startup_steps = 0;
  size_t period        = 1;

  [[nodiscard]] size_t step_count() const { return exact ? startup_steps + period : 1; }
};

// Describes the storage domains whose ordering cannot be exploited by sequence planning.
//
// `indices` are indices into SectionPlan::domains. `combinations` is the size of their Cartesian
// product.
struct CartesianDomain {
  std::vector<size_t> indices;
  size_t              combinations = 1;
};

// Compute the finite simulation horizon required to observe every joint sequence state.
//
// The combined period is built as an incremental LCM. Computing the multiplicative factor before
// multiplication lets the planning budget double as an overflow guard.
SequenceHorizon compute_sequence_horizon(const SectionPlan &plan) {
  SequenceHorizon horizon;

  for (const auto &domain : plan.domains) {
    if (!domain.sequence)
      continue;

    const auto &sequence  = *domain.sequence;
    horizon.startup_steps = std::max(horizon.startup_steps, sequence.prefix.size());

    const size_t cycle_size = sequence.cycle.size();
    const size_t factor     = cycle_size / std::gcd(horizon.period, cycle_size);

    if (factor > planning_budget / horizon.period) {
      horizon.exact = false;
      return horizon;
    }

    horizon.period *= factor;
  }

  if (horizon.startup_steps > planning_budget ||
      horizon.period > planning_budget - horizon.startup_steps) {
    horizon.exact = false;
  }

  return horizon;
}

// Return the pointer index selected by a finite-prefix/repeating-cycle sequence at `step`.
//
// For example, prefix=[0,1] and cycle=[2,3] produces:
//
//   0, 1, 2, 3, 2, 3, 2, 3, ...
size_t pointer_index_at(const core::PointerSequence &sequence, size_t step) {
  if (step < sequence.prefix.size())
    return sequence.prefix[step];

  const size_t cycle_step = step - sequence.prefix.size();
  return sequence.cycle[cycle_step % sequence.cycle.size()];
}

// Determine which domains must be conservatively expanded as a Cartesian product.
//
// When sequence planning is exact, domains with a declared sequence are excluded because their
// ordering is handled by sequence simulation. When exact planning is unavailable, every domain is
// Cartesian.
//
// Returns nullopt when the Cartesian part alone already exceeds the configured graph limit.
std::optional<CartesianDomain> compute_cartesian_domain(const SectionPlan &plan, bool use_sequences,
                                                        size_t limit) {
  CartesianDomain cartesian;

  for (size_t i = 0; i < plan.domains.size(); ++i) {
    const auto &domain = plan.domains[i];

    if (use_sequences && domain.sequence)
      continue;

    if (domain.pointer_count > limit / cartesian.combinations)
      return std::nullopt;

    cartesian.indices.push_back(i);
    cartesian.combinations *= domain.pointer_count;
  }

  return cartesian;
}

// Build the pointer-index tuple generated by all declared sequences at one logical step.
//
// Entries belonging to unordered domains remain zero here and are filled later by Cartesian
// expansion.
IndexTuple sequence_tuple_at(const SectionPlan &plan, size_t step, bool use_sequences) {
  IndexTuple tuple(plan.domains.size(), 0);

  if (!use_sequences)
    return tuple;

  for (size_t i = 0; i < plan.domains.size(); ++i) {
    const auto &sequence = plan.domains[i].sequence;
    if (sequence)
      tuple[i] = pointer_index_at(*sequence, step);
  }

  return tuple;
}

// Enumerate every distinct tuple reachable from the declared pointer sequences.
//
// std::set both removes duplicate states and gives deterministic graph-variant ordering. The
// Cartesian multiplier participates in the limit check because every base tuple will later be
// expanded by that many unordered combinations.
std::optional<IndexTupleSet> enumerate_sequence_tuples(const SectionPlan     &plan,
                                                       const SequenceHorizon &horizon,
                                                       size_t cartesian_combinations,
                                                       size_t limit) {
  IndexTupleSet tuples;

  for (size_t step = 0; step < horizon.step_count(); ++step) {
    tuples.insert(sequence_tuple_at(plan, step, horizon.exact));

    if (tuples.size() > limit / cartesian_combinations)
      return std::nullopt;
  }

  return tuples;
}

// Expand unordered pointer domains over every sequence-controlled base tuple.
//
// The combination number is decoded as a mixed-radix integer. For counts [2,3], the generated
// states are [0,0], [1,0], [0,1], [1,1], [0,2], [1,2]. This avoids constructing intermediate
// Cartesian-product containers.
void expand_cartesian_domains(SectionPlan &plan, const IndexTupleSet &bases,
                              const CartesianDomain &cartesian) {
  plan.tuples.clear();
  plan.tuples.reserve(bases.size() * cartesian.combinations);

  for (const auto &base : bases) {
    for (size_t combination = 0; combination < cartesian.combinations; ++combination) {
      auto   tuple     = base;
      size_t remaining = combination;

      for (const size_t domain_index : cartesian.indices) {
        const size_t radix  = plan.domains[domain_index].pointer_count;
        tuple[domain_index] = remaining % radix;
        remaining /= radix;
      }

      plan.tuples.push_back(std::move(tuple));
    }
  }
}

// Record whether exact sequence planning or conservative Cartesian planning was selected.
void report_sequence_horizon(SectionPlan &plan, const SequenceHorizon &horizon) {
  auto &report = plan.report;

  report["planning_mode"]   = horizon.exact ? "sequence" : "cartesian";
  report["planning_budget"] = planning_budget;

  if (horizon.exact) {
    report["startup_steps"] = horizon.startup_steps;
    report["period"]        = horizon.period;
  } else {
    report["planning_note"] =
        "Sequence planning budget exceeded; using conservative Cartesian domains";
  }
}

// Mark tuple planning unavailable because the number of required graph variants exceeds the
// configured per-section limit.
void block_tuple_planning(SectionPlan &plan, const std::string &reason) {
  plan.block(reason);
  plan.report["pruned_count_status"] = "exceeds_limit";
}

// Enumerate the pointer-index tuples for which CUDA graph variants must be prepared.
//
// Only finite declarations are simulated: tasks are never executed and runtime queues are never
// acquired while planning. The algorithm is:
//
//   1. Compute the exact joint sequence horizon when it fits within planning_budget.
//   2. Identify domains that still require conservative Cartesian expansion.
//   3. Enumerate distinct states reachable from the declared sequences.
//   4. Expand the unordered domains over those states.
void plan_tuples(SectionPlan &plan, size_t limit) {
  const auto horizon = compute_sequence_horizon(plan);
  report_sequence_horizon(plan, horizon);

  const auto cartesian = compute_cartesian_domain(plan, horizon.exact, limit);
  if (!cartesian) {
    block_tuple_planning(plan, "Unspecified pointer product exceeds graph limit");
    return;
  }

  const auto bases = enumerate_sequence_tuples(plan, horizon, cartesian->combinations, limit);
  if (!bases) {
    block_tuple_planning(plan, "Reachable pointer combinations exceed graph limit");
    return;
  }

  expand_cartesian_domains(plan, *bases, *cartesian);
  plan.report["pruned_count"]        = plan.tuples.size();
  plan.report["pruned_count_status"] = "exact";
}

// Initialize the stable diagnostics schema for one section inspection.
nlohmann::json make_section_report(const Section &section, size_t graph_limit) {
  return {{"section", section.name},
          {"section_id", section.id},
          {"enabled", false},
          {"limit", graph_limit},
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
}

// Inspect synchronous tasks and collect every storage touched by the section.
//
// A section can be graph-recorded only when every synchronous task explicitly supports CUDA graph
// recording and no task owns an output storage. Unsupported sections are still inspected further so
// diagnostics can report their pointer-domain shape.
std::set<size_t> inspect_tasks(GraphContext &context, const Section &section, SectionPlan &plan) {
  std::set<size_t> storage_ids;

  for (const auto vertex : section.sync_topo) {
    const auto &node = context.graph[vertex];
    auto *task = static_cast<core::ISyncTask *>(context.resources.tasks.at(node.spec.name).get());

    const bool supports_cuda_graph = task->supports_cuda_graph();
    const bool owns_output =
        std::ranges::any_of(node.infer.owned_outputs, [](bool owned) { return owned; });

    plan.report["tasks"].push_back({{"name", node.spec.name},
                                    {"kind", node.spec.kind},
                                    {"supports_cuda_graph", supports_cuda_graph},
                                    {"owns_output", owns_output}});

    if (!supports_cuda_graph || owns_output)
      plan.block(node.spec.name + " requires ordinary execution");

    for (const auto tid : node.in_tids)
      storage_ids.insert(context.resources.tid_to_sid.at(tid));
    for (const auto tid : node.out_tids)
      storage_ids.insert(context.resources.tid_to_sid.at(tid));
  }

  return storage_ids;
}

// Append all tensor ids aliased by one storage to its diagnostics entry.
void report_storage_tensors(const ExecResouces &resources, size_t storage_id,
                            nlohmann::json &domain_report) {
  for (const auto &[tensor_id, sid] : resources.tid_to_sid) {
    if (sid == storage_id)
      domain_report["tensor_ids"].push_back(tensor_id);
  }
}

// Validate a declared pointer sequence against its storage cardinality.
//
// An empty cycle is never valid because a PointerSequence must define behavior after its finite
// prefix. If the pointer count is unknown, index-range validation is deferred, matching the
// original inspection behavior while the missing count itself blocks graph preparation.
bool valid_pointer_sequence(const core::PointerSequence &sequence,
                            std::optional<size_t>        pointer_count) {
  if (sequence.cycle.empty())
    return false;

  const auto out_of_domain = [pointer_count](size_t index) {
    return pointer_count && index >= *pointer_count;
  };

  return !std::ranges::any_of(sequence.prefix, out_of_domain) &&
         !std::ranges::any_of(sequence.cycle, out_of_domain);
}

// Inspect one storage's finite pointer-domain contract without acquiring any runtime buffer.
StorageDomain inspect_storage_domain(GraphContext &context, const Section &section,
                                     const StorageOwners &owners, size_t storage_id,
                                     SectionPlan &plan, std::optional<size_t> &raw_product,
                                     bool &has_unknown_count) {
  const auto  owner_it = owners.find(storage_id);
  const auto *owner    = owner_it == owners.end() ? nullptr : &owner_it->second;

  const std::optional<size_t> declared_count =
      owner ? owner->pointer_count : std::optional<size_t>{1};

  auto domain_report =
      nlohmann::json{{"storage_id", storage_id},
                     {"tensor_ids", nlohmann::json::array()},
                     {"declared_count",
                      declared_count ? nlohmann::json(*declared_count) : nlohmann::json(nullptr)},
                     {"enumerated_count", nullptr},
                     {"enumeration", "skipped"},
                     {"sequence", "unspecified"}};

  report_storage_tensors(context.resources, storage_id, domain_report);

  StorageDomain domain;
  domain.storage_id    = storage_id;
  domain.pointer_count = declared_count.value_or(0);

  if (!owner) {
    domain_report["owner"] = "compiler";
    domain.sequence        = core::PointerSequence{{}, {0}};
  } else {
    domain_report["owner"]     = owner->task_name;
    domain_report["port"]      = owner->port;
    domain_report["direction"] = owner->is_input() ? "input" : "output";

    try {
      if (!owner->error.empty())
        throw std::invalid_argument(owner->error);
      domain.sequence = owner->pointer_sequence();
    } catch (const std::exception &error) {
      domain_report["error"] = error.what();
      plan.invalid(std::format("{} storage {}: {}", owner->task_name, storage_id, error.what()));
    }
  }

  if (!declared_count) {
    has_unknown_count = true;
    plan.block(std::format("Storage {} has no declared pointer count", storage_id));
  } else if (*declared_count == 0) {
    domain_report["error"] = "Zero pointer count";
    plan.invalid(std::format("Storage {} declares zero pointers", storage_id));
  } else {
    raw_product = checked_product(raw_product, *declared_count);
  }

  if (domain.sequence) {
    domain_report["sequence"] = "exact";
    domain_report["prefix"]   = domain.sequence->prefix;
    domain_report["cycle"]    = domain.sequence->cycle;

    if (!valid_pointer_sequence(*domain.sequence, declared_count)) {
      domain_report["error"] = "Empty cycle or out-of-domain sequence index";
      plan.invalid(std::format("Storage {} has invalid pointer sequence", storage_id));
    }
  }

  logger()->info("[CUDA graphs] {} storage {} owner {}: declared {}, order {}", section.name,
                 storage_id, domain_report["owner"].get<std::string>(),
                 domain_report["declared_count"].dump(),
                 domain_report["sequence"].get<std::string>());

  plan.report["domains"].push_back(std::move(domain_report));
  return domain;
}

// Inspect finite pointer-domain metadata for every storage touched by the section.
//
// Returns true when at least one storage does not declare a pointer count. The raw Cartesian count
// is still reported independently and may itself be unavailable because of size_t overflow.
bool inspect_storage_domains(GraphContext &context, const Section &section,
                             const StorageOwners &owners, const std::set<size_t> &storage_ids,
                             SectionPlan &plan) {
  std::optional<size_t> raw_product       = 1;
  bool                  has_unknown_count = false;

  plan.domains.reserve(storage_ids.size());

  for (const size_t storage_id : storage_ids) {
    plan.domains.push_back(inspect_storage_domain(context, section, owners, storage_id, plan,
                                                  raw_product, has_unknown_count));
  }

  plan.report["raw_count_status"] = has_unknown_count ? "unknown"
                                    : raw_product     ? "exact"
                                                      : "overflow";

  if (!has_unknown_count && raw_product)
    plan.report["raw_cartesian_count"] = *raw_product;

  return has_unknown_count;
}

// Ask an owner for the concrete addresses in one storage domain.
//
// Compiler-owned storages have exactly one current address. Task-owned storages provide their full
// declared finite domain without dequeuing or otherwise advancing runtime state.
std::optional<std::vector<std::byte *>> enumerate_storage_pointers(GraphContext        &context,
                                                                   const StorageOwners &owners,
                                                                   const StorageDomain &domain) {
  const auto owner = owners.find(domain.storage_id);

  if (owner != owners.end())
    return owner->second.enumerate_pointers();

  auto *ptr = context.resources.storages.at(domain.storage_id)->ptr;
  if (!ptr)
    return std::nullopt;

  return std::vector<std::byte *>{ptr};
}

// Validate and store the concrete pointer enumeration for every planned storage domain.
//
// This phase runs only for sections that remain eligible after task/domain inspection. Once
// started, all domains are inspected even if an earlier one fails so diagnostics remain complete.
void enumerate_pointer_domains(GraphContext &context, const Section &section,
                               const StorageOwners &owners, SectionPlan &plan) {
  for (size_t i = 0; i < plan.domains.size(); ++i) {
    auto &domain        = plan.domains[i];
    auto &domain_report = plan.report["domains"][i];

    try {
      auto pointers                = enumerate_storage_pointers(context, owners, domain);
      domain_report["enumeration"] = pointers ? "returned" : "unknown";

      if (!pointers) {
        plan.block(std::format("Storage {} has no pointer enumeration", domain.storage_id));
        continue;
      }

      domain_report["enumerated_count"] = pointers->size();
      if (pointers->size() != domain.pointer_count)
        throw std::invalid_argument("Pointer count mismatch");

      std::set<std::byte *> unique;
      for (auto *ptr : *pointers) {
        if (!ptr || !unique.insert(ptr).second)
          throw std::invalid_argument("Null or duplicate pointer");

        logger()->trace("[CUDA graphs] {} storage {} address {}", section.name, domain.storage_id,
                        static_cast<void *>(ptr));
      }

      domain.pointers = std::move(*pointers);
    } catch (const std::exception &error) {
      domain_report["enumeration"] = "invalid";
      domain_report["error"]       = error.what();
      plan.invalid(std::format("Storage {}: {}", domain.storage_id, error.what()));
      domain.pointers.clear();
    }
  }
}

// Finalize the inspection status after all non-mutating checks and optional pointer enumeration.
void finalize_inspection_status(SectionPlan &plan) {
  auto &report     = plan.report;
  report["status"] = plan.eligible() ? "planned" : plan.fatal.empty() ? "fallback" : "error";

  if (!plan.eligible()) {
    report["failure_stage"]   = "inspection";
    report["fallback_reason"] = report["blockers"][0];
  }
}

// Inspect one section and build the complete finite pointer-domain plan needed for CUDA graph
// specialization. Inspection never executes section tasks.
SectionPlan inspect_section(GraphContext &context, const Section &section,
                            const StorageOwners &owners) {
  SectionPlan plan;
  plan.report = make_section_report(section, context.resources.max_section_cuda_graphs);

  const auto storage_ids = inspect_tasks(context, section, plan);
  const bool has_unknown_count =
      inspect_storage_domains(context, section, owners, storage_ids, plan);

  if (section.sync_topo.empty())
    plan.block("No synchronous tasks");

  if (context.resources.max_section_cuda_graphs == 0)
    plan.block("Section graphs disabled");

  // Tuple planning is useful diagnostically even when another non-fatal blocker already requires
  // ordinary execution. Fatal domain errors and unknown cardinalities, however, make finite tuple
  // planning ill-defined.
  if (!has_unknown_count && plan.fatal.empty() && !storage_ids.empty() &&
      context.resources.max_section_cuda_graphs != 0) {
    plan_tuples(plan, context.resources.max_section_cuda_graphs);
  }

  if (plan.eligible())
    enumerate_pointer_domains(context, section, owners, plan);

  finalize_inspection_status(plan);
  return plan;
}

// Validate the topology of a task recording and return its recursive node count.
//
// Embedded child-graph nodes are rejected because section recording expects task work to be
// captured directly into the section graph. Conditional body graphs are different: they are owned
// by the conditional node itself, so they are valid and are traversed recursively for diagnostics.
size_t validate_recorded_graph(cudaGraph_t graph) {
  size_t node_count = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));

  std::vector<cudaGraphNode_t> nodes(node_count);
  CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &node_count));

  size_t recursive_count = node_count;

  for (const auto node : nodes) {
    cudaGraphNodeType type;
    CUDA_CHECK(cudaGraphNodeGetType(node, &type));

    if (type == cudaGraphNodeTypeGraph)
      throw std::runtime_error("Recording contains an embedded child graph");

    if (type != cudaGraphNodeTypeConditional)
      continue;

    cudaGraphNodeParams params{};
    CUDA_CHECK(cudaGraphNodeGetParams(node, &params));

    for (unsigned int i = 0; i < params.conditional.size; ++i)
      recursive_count += validate_recorded_graph(params.conditional.phGraph_out[i]);
  }

  return recursive_count;
}

using StorageBindings = std::map<size_t, core::Storage>;

// Build temporary storage objects whose pointers correspond to one graph variant.
//
// The live runtime storages are never mutated during capture. TViews created from these bindings
// therefore describe the desired specialization while preserving runtime state.
StorageBindings make_variant_bindings(GraphContext &context, const SectionCudaGraphs &graphs,
                                      const SectionCudaGraphs::PointerTuple &tuple) {
  StorageBindings bindings;

  for (size_t i = graphs.storage_ids.size(); i-- > 0;) {
    const auto storage_id = graphs.storage_ids[i];
    auto [it, inserted] = bindings.emplace(storage_id, *context.resources.storages.at(storage_id));
    (void)inserted;
    it->second.ptr = reinterpret_cast<std::byte *>(tuple[i]);
  }

  return bindings;
}

// Resolve tensor ids into views backed by the temporary variant bindings.
std::vector<core::TView> make_views(GraphContext &context, StorageBindings &bindings,
                                    const std::vector<int> &tensor_ids) {
  std::vector<core::TView> views;
  views.reserve(tensor_ids.size());

  for (const auto tensor_id : tensor_ids) {
    const auto storage_id = context.resources.tid_to_sid.at(tensor_id);
    views.push_back({context.resources.tensor_descs.at(tensor_id), &bindings.at(storage_id)});
  }

  return views;
}

// Ask every synchronous task in topological order to record its CUDA work into the active section
// capture. Task errors are annotated with the task name before leaving the recording layer.
void record_section_tasks(GraphContext &context, const Section &section, StorageBindings &bindings,
                          cudaGraph_t graph, nlohmann::json &report) {
  for (const auto vertex : section.sync_topo) {
    const auto &node    = context.graph[vertex];
    auto        inputs  = make_views(context, bindings, node.in_tids);
    auto        outputs = make_views(context, bindings, node.out_tids);

    core::CudaGraphCtx graph_context{inputs, outputs, section.stream, graph};
    report["failure_stage"]  = "record";
    report["recording_task"] = node.spec.name;

    try {
      static_cast<core::ISyncTask *>(context.resources.tasks.at(node.spec.name).get())
          ->record_cuda_graph(graph_context);
    } catch (const std::exception &error) {
      throw std::runtime_error(std::format("{}: {}", node.spec.name, error.what()));
    }
  }

  report.erase("recording_task");
}

// Capture and instantiate one section specialization for a concrete tuple of storage addresses.
//
// Capture uses a graph created explicitly with cudaGraphCreate so task recording can reference the
// same graph through CudaGraphCtx. Cleanup is deliberately manual here because ending an
// invalidated stream capture may transfer ownership, destroy the original graph, or return a
// replacement graph.
void record_variant(GraphContext &context, const Section &section, SectionCudaGraphs &graphs,
                    const SectionCudaGraphs::PointerTuple &tuple, nlohmann::json &report) {
  auto bindings = make_variant_bindings(context, graphs, tuple);

  cudaGraph_t     graph      = nullptr;
  cudaGraphExec_t executable = nullptr;
  bool            capturing  = false;

  try {
    report["failure_stage"] = "capture_begin";
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(section.stream, graph, nullptr, nullptr, 0,
                                             cudaStreamCaptureModeThreadLocal));
    capturing = true;

    record_section_tasks(context, section, bindings, graph, report);

    report["failure_stage"] = "capture_end";
    cudaGraph_t captured    = nullptr;
    const auto  end_result  = cudaStreamEndCapture(section.stream, &captured);
    capturing               = false;

    // EndCapture transfers the completed graph back to the caller, or destroys it when capture was
    // invalidated. From this point onward `captured` is the graph whose ownership must be managed.
    graph = captured;
    CUDA_CHECK(end_result);

    report["failure_stage"] = "validate_graph";
    const size_t node_count = validate_recorded_graph(graph);

    report["failure_stage"] = "instantiate";
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));

    graphs.executables.push_back(executable);
    executable = nullptr;
    graphs.executable_node_counts.push_back(node_count);
    graphs.node_count += node_count;

    CUDA_CHECK_NT(cudaGraphDestroy(graph));
    graph = nullptr;
  } catch (...) {
    if (capturing) {
      cudaGraph_t captured = nullptr;
      (void)cudaStreamEndCapture(section.stream, &captured);
      graph = captured;
    }

    if (executable)
      CUDA_CHECK_NT(cudaGraphExecDestroy(executable));
    if (graph)
      CUDA_CHECK_NT(cudaGraphDestroy(graph));

    throw;
  }
}

// Convert planned pointer-index tuples into the concrete address tuples used as graph-cache keys.
std::vector<SectionCudaGraphs::PointerTuple> resolve_pointer_tuples(const SectionPlan &plan) {
  std::vector<SectionCudaGraphs::PointerTuple> tuples;
  tuples.reserve(plan.tuples.size());

  for (const auto &indices : plan.tuples) {
    SectionCudaGraphs::PointerTuple tuple;
    tuple.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) {
      tuple.push_back(reinterpret_cast<uintptr_t>(plan.domains[i].pointers.at(indices[i])));
    }

    tuples.push_back(std::move(tuple));
  }

  return tuples;
}

// Initialize the immutable pointer-domain metadata of a replacement graph cache.
void initialize_graph_cache_domains(const SectionPlan &plan, SectionCudaGraphs &graphs) {
  graphs.storage_ids.reserve(plan.domains.size());
  graphs.pointers.reserve(plan.domains.size());

  for (const auto &domain : plan.domains) {
    graphs.storage_ids.push_back(domain.storage_id);
    graphs.pointers.push_back(domain.pointers);
  }
}

// Destroy cached variants that cannot be reused by the refreshed plan.
//
// Obsolete executables are released before recording replacements so a refresh does not
// temporarily double the executable-memory budget.
size_t discard_obsolete_variants(SectionCudaGraphs &current, const SectionCudaGraphs &replacement,
                                 const std::set<SectionCudaGraphs::PointerTuple> &desired) {
  size_t discarded = 0;

  for (const auto &[tuple, index] : current.tuple_indices) {
    const bool same_domains = current.storage_ids == replacement.storage_ids;
    if (same_domains && desired.contains(tuple))
      continue;

    CUDA_CHECK_NT(cudaGraphExecDestroy(current.executables[index]));
    current.executables[index] = nullptr;
    ++discarded;
  }

  return discarded;
}

struct VariantRefreshStats {
  size_t reused    = 0;
  size_t created   = 0;
  size_t discarded = 0;
};

// Populate a replacement cache by transferring reusable executables and recording missing variants.
void build_graph_variants(GraphContext &context, const Section &section, SectionPlan &plan,
                          SectionCudaGraphs &current, SectionCudaGraphs &replacement,
                          VariantRefreshStats &stats) {
  const auto                                      desired = resolve_pointer_tuples(plan);
  const std::set<SectionCudaGraphs::PointerTuple> desired_set(desired.begin(), desired.end());

  stats.discarded = discard_obsolete_variants(current, replacement, desired_set);

  replacement.executables.reserve(desired.size());
  replacement.executable_node_counts.reserve(desired.size());

  for (const auto &tuple : desired) {
    replacement.tuple_indices.emplace(tuple, replacement.executables.size());

    const auto old = current.tuple_indices.find(tuple);
    if (old != current.tuple_indices.end() && current.executables[old->second]) {
      replacement.executables.push_back(current.executables[old->second]);
      current.executables[old->second] = nullptr;
      replacement.executable_node_counts.push_back(current.executable_node_counts[old->second]);
      replacement.node_count += current.executable_node_counts[old->second];
      ++stats.reused;
      continue;
    }

    plan.report["variant_index"] = replacement.executables.size();
    record_variant(context, section, replacement, tuple, plan.report);
    ++stats.created;
  }

  plan.report["failure_stage"] = "";
  plan.report.erase("variant_index");
  replacement.enabled = true;
}

// Move graph-cache state field-by-field because SectionCudaGraphs also contains synchronization and
// accounting members that must stay attached to the existing section object.
void install_graph_cache(SectionCudaGraphs &destination, SectionCudaGraphs &source) {
  destination.clear();
  destination.storage_ids            = std::move(source.storage_ids);
  destination.pointers               = std::move(source.pointers);
  destination.tuple_indices          = std::move(source.tuple_indices);
  destination.executables            = std::move(source.executables);
  destination.executable_node_counts = std::move(source.executable_node_counts);
  destination.node_count             = source.node_count;
  destination.enabled                = source.enabled;
}

// Recover from a capture/instantiation failure before ordinary execution is allowed to resume.
//
// A successful stream synchronization establishes that prior CUDA work completed and the context is
// usable; the graph optimization can then fall back normally. A synchronization failure indicates a
// poisoned CUDA context and is escalated to the caller.
void recover_section_stream(const Section &section, nlohmann::json &report) {
  try {
    CUDA_CHECK(cudaStreamSynchronize(section.stream));
  } catch (...) {
    report["status"]        = "error";
    report["failure_stage"] = "cuda_context";
    throw;
  }

  (void)cudaGetLastError();
}

// Build or refresh all executable graph variants for one already-inspected section.
//
// The refresh is transactional with respect to the replacement cache: reusable executables are
// transferred where possible, obsolete ones are released early, and any construction failure clears
// both caches before falling back to ordinary execution.
void prepare_section(GraphContext &context, const Section &section, SectionPlan &plan,
                     SectionCudaGraphs &graphs) {
  auto      &report        = plan.report;
  const auto started       = std::chrono::steady_clock::now();
  const auto previous_size = graphs.executables.size();

  VariantRefreshStats stats;
  SectionCudaGraphs   replacement;
  initialize_graph_cache_domains(plan, replacement);

  try {
    if (plan.eligible())
      build_graph_variants(context, section, plan, graphs, replacement, stats);
  } catch (const std::exception &error) {
    plan.block(error.what());
    report["status"]          = "fallback";
    report["fallback_reason"] = error.what();

    replacement.clear();
    graphs.clear();
    recover_section_stream(section, report);
  }

  if (!replacement.enabled)
    stats.discarded = previous_size + stats.created;

  install_graph_cache(graphs, replacement);

  graphs.variant_count   = plan.tuples.size();
  graphs.fallback_reason = report["fallback_reason"].get<std::string>();
  graphs.construction_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();

  report["enabled"] = graphs.enabled;
  if (graphs.enabled)
    report["status"] = "ready";

  report["variants"]        = graphs.executables.size();
  report["node_count"]      = graphs.node_count;
  report["reused"]          = stats.reused;
  report["created"]         = stats.created;
  report["discarded"]       = stats.discarded;
  report["construction_ms"] = graphs.construction_ms;
}

// Return the persistent graph-cache object associated with a section, creating it on first use.
SectionCudaGraphs &section_graph_cache(ExecResouces &resources, size_t section_id) {
  auto &graphs = resources.section_cuda_graphs[static_cast<int>(section_id)];
  if (!graphs)
    graphs = std::make_unique<SectionCudaGraphs>();
  return *graphs;
}

// Publish diagnostics atomically with respect to readers of SectionCudaGraphs::snapshot().
void publish_diagnostics(SectionCudaGraphs &graphs, const nlohmann::json &report) {
  std::lock_guard lock(graphs.diagnostic_mutex);
  graphs.diagnostics = report;
}

// Write all section diagnostics to disk if a diagnostics directory is configured.
void write_diagnostics_file(const ExecResouces &resources) {
  if (resources.section_cuda_graph_log_dir.empty())
    return;

  auto diagnostics = nlohmann::json::array();
  for (const auto &entry : resources.section_cuda_graphs)
    diagnostics.push_back(entry.second->snapshot());

  std::ofstream file(resources.section_cuda_graph_log_dir / "section_cuda_graphs.json");
  if (file) {
    file << diagnostics.dump(2);
  } else {
    logger()->warn("[CUDA graphs] Could not write section_cuda_graphs.json");
  }
}

struct InspectionBatch {
  std::vector<SectionPlan> plans;
  std::string              first_fatal;
};

// Inspect every section before instantiating any graphs.
//
// Completing the inspection pass first ensures diagnostics for all sections are available even when
// one malformed pointer contract ultimately makes the refresh fail.
InspectionBatch inspect_sections(GraphContext &context, const std::vector<Section> &sections,
                                 const StorageOwners &owners, bool instantiate) {
  InspectionBatch batch;
  batch.plans.reserve(sections.size());

  for (const auto &section : sections) {
    const auto started = std::chrono::steady_clock::now();
    auto       plan    = inspect_section(context, section, owners);

    if (!plan.fatal.empty() && batch.first_fatal.empty())
      batch.first_fatal = plan.fatal;

    auto &graphs = section_graph_cache(context.resources, section.id);
    if (instantiate)
      ++graphs.refresh_count;

    plan.report["refresh_count"]                  = graphs.refresh_count;
    plan.report["cached_variants_before_refresh"] = graphs.executables.size();
    plan.report["inspection_ms"] =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();

    graphs.enabled         = false;
    graphs.fallback_reason = plan.report["fallback_reason"].get<std::string>();
    publish_diagnostics(graphs, plan.report);

    batch.plans.push_back(std::move(plan));
  }

  return batch;
}

// Log the concise per-section preparation summary used during runtime/compiler diagnostics.
void log_preparation_summary(const Section &section, const SectionPlan &plan,
                             const SectionCudaGraphs &graphs) {
  logger()->info("[CUDA graphs] {}: raw {}, pruned {}, built {}, reused {}, {} ms; {}; blockers {}",
                 section.name, plan.report["raw_cartesian_count"].dump(),
                 plan.report["pruned_count"].dump(), graphs.executables.size(),
                 plan.report["reused"].dump(), plan.report["preparation_ms"].dump(),
                 plan.report["status"].get<std::string>(), plan.report["blockers"].dump());
}

// Prepare inspected sections and keep each diagnostics snapshot synchronized with its latest state.
void prepare_sections(GraphContext &context, const std::vector<Section> &sections,
                      std::vector<SectionPlan> &plans, bool instantiate) {
  for (size_t i = 0; i < sections.size(); ++i) {
    const auto &section = sections[i];
    auto       &plan    = plans[i];
    auto       &graphs  = *context.resources.section_cuda_graphs.at(section.id);

    try {
      if (instantiate)
        prepare_section(context, section, plan, graphs);
    } catch (...) {
      publish_diagnostics(graphs, plan.report);
      write_diagnostics_file(context.resources);
      throw;
    }

    plan.report["preparation_ms"] =
        plan.report["inspection_ms"].get<double>() + plan.report.value("construction_ms", 0.0);

    publish_diagnostics(graphs, plan.report);
    log_preparation_summary(section, plan, graphs);
  }
}

} // namespace

// Serialize the latest CUDA-graph diagnostics for all sections when logging is enabled.
void write_section_cuda_graph_diagnostics(const ExecResouces &resources) {
  write_diagnostics_file(resources);
}

// Recompute section CUDA-graph eligibility and, optionally, instantiate the required
// specializations.
//
// Refresh is intentionally split into two passes:
//
//   1. inspect every section and publish diagnostics without mutating CUDA graph executables;
//   2. reject fatal pointer-domain contracts, then build/reuse graph variants section by section.
//
// Non-fatal blockers leave the corresponding section on ordinary execution. Fatal metadata errors
// are reported for all sections before the first one is thrown to the caller.
void refresh_section_cuda_graphs(const GraphPlan &graph, const std::vector<Section> &sections,
                                 ExecResouces &resources, bool instantiate) {
  GraphContext context{graph, resources};
  const auto   owners = collect_storage_owners(context);
  auto         batch  = inspect_sections(context, sections, owners, instantiate);

  write_diagnostics_file(resources);

  if (!batch.first_fatal.empty())
    throw std::invalid_argument(batch.first_fatal);

  prepare_sections(context, sections, batch.plans, instantiate);
  write_diagnostics_file(resources);
}

} // namespace holoflow::runtime
