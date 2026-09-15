// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "section_cuda_graph.hh"

#include <algorithm>
#include <chrono>
#include <format>
#include <set>
#include <stdexcept>

#include "../logger.hh"

namespace holoflow::runtime {

SectionCudaGraphs::~SectionCudaGraphs() { clear(); }

void SectionCudaGraphs::clear() noexcept {
  for (auto executable : executables) {
    CUDA_CHECK_NT(cudaGraphExecDestroy(executable));
  }
  executables.clear();
  node_count = 0;
  enabled    = false;
}

std::optional<size_t>
SectionCudaGraphs::variant(const std::map<size_t, std::unique_ptr<core::Storage>> &storages) const {
  size_t index = 0;
  for (size_t i = 0; i < storage_ids.size(); ++i) {
    const auto it = pointer_indices[i].find(storages.at(storage_ids[i])->ptr);
    if (it == pointer_indices[i].end()) {
      return std::nullopt;
    }
    index = index * pointers[i].size() + it->second;
  }
  return index;
}

namespace {

struct Owner {
  core::ITask          *task;
  std::string           name;
  size_t                port;
  bool                  input;
  std::optional<size_t> count;
};

std::map<size_t, Owner> storage_owners(CompilerOutput &out) {
  std::map<size_t, Owner> owners;
  for (auto v : boost::make_iterator_range(boost::vertices(out.graph))) {
    const auto &np  = out.graph[v];
    auto        add = [&](bool input) {
      const auto &mask = input ? np.infer.owned_inputs : np.infer.owned_outputs;
      const auto &counts =
          input ? np.infer.owned_input_pointer_counts : np.infer.owned_output_pointer_counts;
      const auto &tids = input ? np.in_tids : np.out_tids;
      if (!counts.empty() && counts.size() != tids.size()) {
        throw std::invalid_argument(
            std::format("{}: pointer count vector has wrong size", np.spec.name));
      }
      for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
          owners.emplace(out.resources.tid_to_sid.at(tids[i]),
                         Owner{out.resources.tasks.at(np.spec.name).get(), np.spec.name, i, input,
                               counts.empty() ? std::nullopt : counts[i]});
        }
      }
    };
    add(true);
    add(false);
  }
  return owners;
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

void record_variant(CompilerOutput &out, const Section &sec, SectionCudaGraphs &set, size_t index) {
  std::map<size_t, core::Storage> bindings;
  for (size_t i = set.storage_ids.size(); i-- > 0;) {
    auto sid = set.storage_ids[i];
    bindings.emplace(sid, *out.resources.storages.at(sid));
    bindings.at(sid).ptr = set.pointers[i][index % set.pointers[i].size()];
    index /= set.pointers[i].size();
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
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(sec.stream, graph, nullptr, nullptr, 0,
                                             cudaStreamCaptureModeThreadLocal));
    capturing = true;
    for (auto v : sec.sync_topo) {
      const auto        &np      = out.graph[v];
      auto               inputs  = views(np.in_tids);
      auto               outputs = views(np.out_tids);
      core::CudaGraphCtx ctx{inputs, outputs, sec.stream, graph};
      try {
        static_cast<core::ISyncTask *>(out.resources.tasks.at(np.spec.name).get())
            ->record_cuda_graph(ctx);
      } catch (const std::exception &e) {
        throw std::runtime_error(std::format("{}: {}", np.spec.name, e.what()));
      }
    }
    cudaGraph_t captured   = nullptr;
    const auto  end_result = cudaStreamEndCapture(sec.stream, &captured);
    capturing              = false;
    // EndCapture transfers the graph back, or destroys it when capture is invalidated.
    graph = captured;
    CUDA_CHECK(end_result);
    const auto nodes = validate_flat_graph(graph);
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    set.executables.push_back(executable);
    executable = nullptr;
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

} // namespace

void build_section_cuda_graphs(CompilerOutput &out, size_t limit) {
  const auto owners = storage_owners(out);
  for (const auto &sec : out.sections) {
    auto  owned_set = std::make_unique<SectionCudaGraphs>();
    auto &set       = *owned_set;
    out.resources.section_cuda_graphs.emplace(sec.id, std::move(owned_set));
    const auto started = std::chrono::steady_clock::now();
    auto       prepare = [&]() {
      if (limit == 0 || sec.sync_topo.empty()) {
        set.fallback_reason = "Section graphs disabled or no synchronous tasks";
        return;
      }
      std::set<size_t> sids;
      for (auto v : sec.sync_topo) {
        const auto &np = out.graph[v];
        auto *task     = static_cast<core::ISyncTask *>(out.resources.tasks.at(np.spec.name).get());
        if (!task->supports_cuda_graph() ||
            std::ranges::any_of(np.infer.owned_outputs, [](bool owned) { return owned; })) {
          set.fallback_reason = std::format("{} requires ordinary execution", np.spec.name);
          return;
        }
        for (auto tid : np.in_tids)
          sids.insert(out.resources.tid_to_sid.at(tid));
        for (auto tid : np.out_tids)
          sids.insert(out.resources.tid_to_sid.at(tid));
      }
      size_t product = 1;
      for (auto sid : sids) {
        const auto owner = owners.find(sid);
        const auto count = owner == owners.end() ? std::optional<size_t>{1} : owner->second.count;
        if (!count) {
          set.fallback_reason = std::format("Storage {} has no declared pointer count", sid);
          return;
        }
        if (*count == 0) {
          throw std::invalid_argument(std::format("Storage {} declares zero pointers", sid));
        }
        if (*count > limit / product) {
          set.fallback_reason =
              std::format("Pointer product exceeds limit {} at storage {} ({} x {})", limit, sid,
                          product, *count);
          return;
        }
        product *= *count;
      }
      set.variant_count = product;
      for (auto sid : sids) {
        std::optional<std::vector<std::byte *>> pointers;
        auto                                    owner = owners.find(sid);
        if (owner == owners.end()) {
          if (out.resources.storages.at(sid)->ptr == nullptr) {
            set.fallback_reason = std::format("Storage {} has no allocated address", sid);
            return;
          }
          pointers = std::vector{out.resources.storages.at(sid)->ptr};
        } else {
          const auto &o = owner->second;
          pointers      = o.input ? o.task->owned_input_pointers(o.port)
                                  : o.task->owned_output_pointers(o.port);
          if (!pointers) {
            set.fallback_reason =
                std::format("{} port {} has no pointer enumeration", o.name, o.port);
            return;
          }
        }
        std::unordered_map<std::byte *, size_t> indices;
        for (auto ptr : *pointers) {
          if (!ptr || !indices.emplace(ptr, indices.size()).second) {
            throw std::invalid_argument(
                std::format("Storage {} has null or duplicate pointers", sid));
          }
        }
        const size_t expected = owner == owners.end() ? 1 : *owner->second.count;
        if (pointers->size() != expected) {
          throw std::invalid_argument(std::format("Storage {} pointer count mismatch", sid));
        }
        set.storage_ids.push_back(sid);
        set.pointers.push_back(std::move(*pointers));
        set.pointer_indices.push_back(std::move(indices));
      }
      try {
        set.executables.reserve(product);
        for (size_t index = 0; index < product; ++index)
          record_variant(out, sec, set, index);
        set.enabled = true;
      } catch (const std::exception &e) {
        set.clear();
        // A poisoned context cannot safely use the ordinary execution fallback.
        CUDA_CHECK(cudaStreamSynchronize(sec.stream));
        (void)cudaGetLastError();
        set.fallback_reason = e.what();
      }
    };
    prepare();
    set.construction_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    logger()->info("[CUDA graphs] {}: {} variants, {} nodes, {:.1f} ms; {}", sec.name,
                   set.executables.size(), set.node_count, set.construction_ms,
                   set.enabled ? "enabled" : set.fallback_reason);
  }
}

} // namespace holoflow::runtime
