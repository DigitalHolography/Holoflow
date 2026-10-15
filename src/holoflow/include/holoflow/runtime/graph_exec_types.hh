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

#pragma once

#include <atomic>
#include <boost/graph/adjacency_list.hpp>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "curaii/cuda.hh"
#include "holoflow/core/graph_spec.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"

namespace holoflow::runtime {

struct NodePlan {
  core::NodeSpec    spec;     ///< Node specification.
  core::InferResult infer;    ///< Inference metadata.
  std::vector<int>  in_tids;  ///< Input tensor IDs.
  std::vector<int>  out_tids; ///< Output tensor IDs.
};

struct EdgePlan {
  core::EdgeSpec spec; ///< Edge specification.
  core::TDesc    desc; ///< Tensor descriptor.
  int            tid;  ///< Tensor ID.
};

using GraphPlan = boost::adjacency_list<boost::vecS,           // OutEdgeList
                                        boost::vecS,           // VertexList
                                        boost::bidirectionalS, // Directed graph
                                        NodePlan,              // Vertex properties
                                        EdgePlan               // Edge properties
                                        >;

/// Represents a block of memory used during graph execution.
struct MemoryBlock {
  core::MemLoc mem_loc;    ///< Memory location (host or device).
  size_t       size_bytes; ///< Size of the memory block in bytes.

  curaii::unique_host_ptr<std::byte>   h_data; ///< Host memory (if applicable).
  curaii::unique_device_ptr<std::byte> d_data; ///< Device memory (if applicable).

  void *get(); ///< Returns a mutable pointer to the memory block.
};

/// Owns the prepared CUDA graph executables for one execution section.
/// Each variant records the section's work for one reachable combination of storage addresses.
/// Variants are built or reused while execution is stopped, before the section resumes; only the
/// section thread selects and launches them. A lookup miss disables graph execution and causes
/// fallback to ordinary task execution. Other threads read diagnostics through snapshot().
struct SectionCudaGraphs {
  /// Destroy all owned executable handles through clear().
  ~SectionCudaGraphs();
  SectionCudaGraphs()                                     = default;
  SectionCudaGraphs(const SectionCudaGraphs &)            = delete;
  SectionCudaGraphs &operator=(const SectionCudaGraphs &) = delete;

  /// Storage IDs participating in variant selection, in the order used by every PointerTuple.
  std::vector<size_t> storage_ids;
  /// Declared storage-base addresses for each storage_ids[i]; used to classify lookup misses.
  /// These pointers are borrowed from storage owners; this object does not own their memory.
  std::vector<std::vector<std::byte *>> pointers;
  /// One address per storage_ids entry, encoded as uintptr_t (addresses, not domain indices).
  /// For storage_ids {a, b}, the tuple is {address of storage a, address of storage b}.
  using PointerTuple = std::vector<uintptr_t>;
  /// Reachable address tuple -> index into executables and executable_node_counts.
  /// Only prepared combinations are included, rather than every combination of known pointers.
  std::map<PointerTuple, size_t> tuple_indices;
  /// Node count for executables[i], including nodes nested in conditional bodies.
  std::vector<size_t> executable_node_counts;
  /// Owned instantiated handles; each executes the entire section for its associated tuple.
  std::vector<cudaGraphExec_t> executables;
  /// Number of reachable variants from the latest preparation plan; may exceed the number of
  /// built executables when preparation fails. Use executables.size() for the installed count.
  size_t variant_count = 0;
  size_t node_count = 0; ///< Total nodes across executable variants, including conditional bodies.
  /// Wall-clock time of the latest preparation, including construction/reuse but not inspection.
  double construction_ms = 0;
  /// Reason for ordinary execution after an eligibility blocker, preparation failure, or miss.
  std::string fallback_reason;
  /// Whether the section may select and launch a prepared executable.
  bool enabled = false;

  /// Protects publication and copying of diagnostics, not executable selection or cache mutation.
  mutable std::mutex diagnostic_mutex;
  /// Published preparation/fallback report. snapshot() copies this instead of inspecting the
  /// executable containers, which can change while the section is stopped.
  nlohmann::json diagnostics = nlohmann::json::object();
  /// Runtime counters for graph launches, ordinary iterations, unknown individual addresses,
  /// and unprepared combinations of otherwise known addresses, respectively. Reset on compilation.
  std::atomic<uint64_t> launches{0}, ordinary_iterations{0}, pointer_misses{0}, tuple_misses{0};
  std::atomic<size_t>   prequeue_depth{1};
  std::atomic<size_t>   prequeue_peak{0};
  std::atomic<uint64_t> prequeue_submissions{0}, prequeue_retirements{0};
  /// Number of refreshes requesting instantiation in the current compilation (even if blocked).
  uint64_t refresh_count = 0;
  /// Compilation generation to which this section cache currently belongs.
  uint64_t compilation_generation = 0;
  /// True when the compiler transferred a compatible cache from the preceding compilation.
  bool carried_from_previous_compilation = false;
  /// Why a preceding cache could not be carried over, or why it was invalidated; empty on reuse.
  std::string compilation_invalidation_reason;
  /// Copy the published report under diagnostic_mutex and add current runtime counters and
  /// compilation metadata. Safe during section execution; counters are sampled independently.
  [[nodiscard]] nlohmann::json snapshot() const;
  /// On the section thread, classify a failed lookup, increment pointer_misses or tuple_misses,
  /// disable graph execution, and publish the offending bindings and fallback reason.
  /// Executables remain owned until clear() or a subsequent refresh.
  void report_miss(const std::map<size_t, std::unique_ptr<core::Storage>> &storages);

  /// Destroy executable handles, clear their counts and tuple lookup, and disable graph execution.
  /// Retains pointer domains, planning metadata, diagnostics, and counters for refresh/reporting.
  /// Call only when no section launch is using these executables.
  void clear() noexcept;
  /// Look up the executable index for current Storage::ptr addresses in storage_ids order.
  /// Returns nullopt for an unprepared tuple; does not check enabled or report the miss.
  /// Every listed storage ID must exist in storages. Called by the section thread during execution.
  [[nodiscard]] std::optional<size_t>
  variant(const std::map<size_t, std::unique_ptr<core::Storage>> &storages) const;
};

struct SectionCudaGraphInspection;

struct ExecResouces {
  std::map<size_t, MemoryBlock>                    memory_blocks; ///< StorageID -> MemoryBlock.
  std::map<size_t, std::unique_ptr<core::Storage>> storages;      ///< StorageID -> Storage.
  std::map<size_t, core::TDesc>                    tensor_descs;  ///< TensorID -> TDesc.
  std::map<size_t, size_t>                         tid_to_sid;    ///< TensorID -> StorageID.

  std::map<std::string, std::unique_ptr<core::IOStorageAccess>> node_storage_adapters;
  std::map<size_t, curaii::CudaStream>                          streams; ///< CUDA streams by ID.
  std::map<std::string, std::unique_ptr<core::ITask>>           tasks;   ///< Task instances by ID.
  size_t                                                        max_section_cuda_graphs = 0;
  std::filesystem::path                                         section_cuda_graph_log_dir;
  uint64_t                                                      compilation_generation = 0;
  // One-use plan for the first start; compilation has already validated these pointer domains.
  std::shared_ptr<SectionCudaGraphInspection> section_cuda_graph_inspection;
  // Declared last so graphs are destroyed before tasks, modules, streams, and buffers.
  std::map<int, std::unique_ptr<SectionCudaGraphs>> section_cuda_graphs;
  // std::map<int, core::Tensor>                         tensors; ///< Allocated tensors by ID.
};

struct Section {
  int                                       id;         ///< Section ID.
  std::string                               name;       ///< Section name (for logging).
  cudaStream_t                              stream;     ///< CUDA stream for this section.
  std::vector<GraphPlan::vertex_descriptor> sync_topo;  ///< Synchronous nodes in topological order.
  std::vector<GraphPlan::vertex_descriptor> async_cons; ///< Asynchronous consumer nodes.
  std::vector<GraphPlan::vertex_descriptor> async_prod; ///< Asynchronous producer nodes
  bool has_synchronizing_async_producer = false; ///< Producer supplies the section stream barrier.
  bool supports_prequeue = false; ///< Boundary ownership supports two outstanding iterations.
  std::string prequeue_fallback_reason;
};

} // namespace holoflow::runtime
