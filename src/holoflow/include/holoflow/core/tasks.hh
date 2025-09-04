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

/// @file tasks.hh
/// @brief Interfaces and runtime contexts for Holoflow tasks.
///
/// A task consumes zero or more input tensors and produces zero or more output
/// tensors. Tasks are either synchronous (blocking) or asynchronous
/// (decoupled push/pop).
///
/// @section overview Overview
/// - Contexts: @ref holoflow::core::SyncCtx, @ref holoflow::core::AsyncPushCtx,
///   @ref holoflow::core::AsyncPopCtx
/// - Control flow: @ref holoflow::core::OpResult
/// - Task interfaces: @ref holoflow::core::ITask, @ref
///   holoflow::core::ISyncTask, @ref holoflow::core::IAsyncTask
/// - Factories: @ref holoflow::core::ITaskFactory,
///   @ref holoflow::core::ISyncTaskFactory, @ref holoflow::core::IAsyncTaskFactory
/// - Inference result: @ref holoflow::core::InferResult
///
/// @section lifecycle Lifecycle (single source of truth)
/// - **Inputs**: Scheduler provides views. For owned inputs, it must call
///   @ref holoflow::core::ITask::acquire_input() and place the returned view in
///   the context.
/// - **Execution**:
///   - Sync: scheduler calls @ref holoflow::core::ISyncTask::execute().
///   - Async push: scheduler calls @ref holoflow::core::IAsyncTask::try_push().
///   - Async pop: scheduler calls @ref holoflow::core::IAsyncTask::try_pop().
/// - **Outputs**: Task writes views to the context. For owned outputs the task
///   overwrites slots with owned views. After downstream use, the scheduler
///   must call @ref holoflow::core::ITask::release_output() per owned slot.
/// - **Cancellation**: cooperative via `cancelled`.
///
/// @section ownership Ownership
/// - Acquire/release apply only to indices marked owned by factory inference.
/// - Calling acquire/release on non-owned indices is undefined.
/// - At most one acquired input group and one unreleased output group exist at
///   a time under SPSC assumptions.
/// - @todo Define policy for discarding an acquired input on cancellation
///   before use (abort API vs implicit discard vs push-then-drop vs aggregation
///   commit).
///
/// @section errors Errors vs control flow
/// - Exceptions report validation/runtime errors (indices, shapes, allocation,
///   internal failures).
/// - @ref holoflow::core::OpResult is only for control flow: NotReady,
///   Cancelled, Eof.
///
/// @section concurrency Concurrency
/// - Assumed SPSC for async tasks. MPMC is undefined behavior.
/// - @todo Document/extend semantics if MPMC is added later.

#pragma once

#include <atomic>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <vector>

#include "driver_types.h"
#include "holoflow/core/tensor.hh"

namespace holoflow::core {

/// Runtime execution context for a synchronous task.
struct SyncCtx {
  std::span<TView>   inputs;    ///< Scheduler-provided input views; some may be owned.
  std::span<TView>   outputs;   ///< Output slots; owned outputs are written by the task.
  std::atomic<bool> *cancelled; ///< Non-null cancellation flag.
};

/// Runtime execution context for an asynchronous task push operation.
struct AsyncPushCtx {
  std::span<TView>   inputs;    ///< Scheduler-provided input views; some may be owned.
  std::atomic<bool> *cancelled; ///< Non-null cancellation flag.
};

/// Runtime execution context for an asynchronous task pop operation.
struct AsyncPopCtx {
  std::span<TView>   outputs;   ///< Output slots; owned outputs are written by the task.
  std::atomic<bool> *cancelled; ///< Non-null cancellation flag.
};

/// Possible expected outcomes of a task operation.
enum class OpResult : uint8_t {
  Ok,        ///< Completed successfully.
  NotReady,  ///< Nothing to do now; caller may retry.
  Cancelled, ///< Aborted due to cancellation.
  Eof,       ///< End of stream.
};

/// @brief Abstract base interface for tasks with optional tensor ownership.
///
/// Provides ownership hooks for inputs and outputs. Only indices declared as
/// owned by the task (via factory inference) may be acquired or released.
///
/// @warning Calling acquire/release on non-owned indices is undefined behavior.
/// @warning Not calling acquire/release on owned indices is undefined behavior.
///
/// @par Ownership lifecycle
/// - Inputs (owned):
///   - Acquire with @ref acquire_input(int) before execution or push.
///   - The acquired view must not be used after the operation returns.
/// - Outputs (owned):
///   - The task assigns owned output views into the context during execute/pop.
///   - After downstream consumption, scheduler calls @ref release_output(int).
/// @todo Define rollback semantics on cancellation before use.
class ITask {
public:
  virtual ~ITask() = default;

  /// Acquire a writable view for an **owned** input index.
  /// Valid only for indices marked owned by inference.
  /// The view is transient and must not outlive the operation.
  /// @returns view or std::nullopt if not currently acquirable.
  /// @throws std::out_of_range on bad index.
  [[nodiscard]] virtual std::optional<TView> acquire_input(int index);

  /// Release a produced **owned** output at @p index after downstream use.
  /// @throws std::out_of_range on bad index.
  virtual void release_output(int index);
};

/// @brief Interface for synchronous (blocking) tasks.
///
/// A synchronous task consumes inputs and produces outputs within a single
/// blocking call to @ref execute().
///
/// @par Synchronous lifecycle
/// 1. Scheduler optionally acquires owned inputs via
///    @ref ITask::acquire_input(int).
/// 2. Scheduler calls @ref execute().
/// 3. Task may write owned output views into @ref SyncCtx::outputs,
///    overwriting the corresponding slots.
/// 4. Scheduler consumes outputs and calls @ref ITask::release_output(int)
///    for each owned output slot.
///
/// @note Inputs acquired via @ref ITask::acquire_input(int) must not be used
///       after @ref execute() returns.
class ISyncTask : public ITask {
public:
  virtual ~ISyncTask() = default;

  /// Execute in a blocking manner.
  /// Overwrites owned output slots in ctx.outputs.
  /// @returns control-flow result; errors via exceptions.
  [[nodiscard]] virtual OpResult execute(SyncCtx &ctx) = 0;
};

/// @brief Interface for asynchronous (decoupled) tasks.
///
/// Asynchronous tasks split their interaction into a producer-side push and a
/// consumer-side pop. Push submits inputs. Pop retrieves available outputs.
///
/// @par Push lifecycle (producer)
/// 1. Scheduler optionally acquires owned inputs via
///    @ref ITask::acquire_input(int).
/// 2. Scheduler calls @ref try_push().
/// 3. Task consumes submitted inputs.
///
/// @par Pop lifecycle (consumer)
/// 1. Scheduler calls @ref try_pop().
/// 2. On success, task writes outputs, overwriting slots for owned outputs.
/// 3. Scheduler consumes outputs and calls @ref ITask::release_output(int) for
///    each owned output slot.
class IAsyncTask : public ITask {
public:
  virtual ~IAsyncTask() = default;

  /// Producer-side submission.
  /// @returns control-flow result; errors via exceptions.
  [[nodiscard]] virtual OpResult try_push(AsyncPushCtx &ctx) = 0;

  /// Consumer-side retrieval.
  /// Overwrites owned output slots in ctx.outputs.
  /// @returns control-flow result; errors via exceptions.
  [[nodiscard]] virtual OpResult try_pop(AsyncPopCtx &ctx) = 0;
};

/// Describes an in-place link between an input and output tensor.
struct InPlace {
  int in_idx;  ///< Input index reused/aliased.
  int out_idx; ///< Output index reusing input storage.
};

/// Kind of task: synchronous or asynchronous.
enum class TaskKind {
  Sync,  /// Synchronous
  Async, /// Asynchronous
};

/// Result of task inference from a factory infer function. This provides
/// information about the task's input and output tensor shapes and other
/// properties to the scheduling system / compiler.
struct InferResult {
  std::vector<TDesc>   input_descs;   ///< Inferred input tensor descriptions
  std::vector<TDesc>   output_descs;  ///< Inferred output tensor descriptions
  std::vector<InPlace> in_place;      ///< In-place input-output tensor pairs
  std::vector<bool>    owned_inputs;  ///< Ownership status of input tensors
  std::vector<bool>    owned_outputs; ///< Ownership status of output tensors
  TaskKind             kind;          ///< Kind of task (sync or async)
};

/// Context for sync task creation.
struct SyncCreateCtx {
  cudaStream_t stream = static_cast<cudaStream_t>(0); ///< CUDA stream for task execution
};

/// Context for async task creation.
struct AsyncCreateCtx {
  /// CUDA streams for producer side.
  cudaStream_t producer_stream = static_cast<cudaStream_t>(0);
  /// CUDA streams for consumer side.
  cudaStream_t consumer_stream = static_cast<cudaStream_t>(0);
};

/// Base factory interface. Provides common inference API.
class ITaskFactory {
public:
  virtual ~ITaskFactory() = default;

  /// Infer metadata for a task without constructing it.
  ///
  /// @param input_descs  Upstream input tensor descriptors.
  /// @param jsettings    Task configuration (read-only view).
  /// @return             Inference result (I/O descs, ownership, in-place).
  /// @throws std::invalid_argument on inference failure.
  [[nodiscard]]
  virtual InferResult infer(std::span<const TDesc> input_descs,
                            const nlohmann::json  &jsettings) const = 0;
};

/// Factory for synchronous tasks.
class ISyncTaskFactory : public ITaskFactory {
public:
  virtual ~ISyncTaskFactory() = default;

  /// Create a new synchronous task instance.
  ///
  /// @param input_descs  Upstream input tensor descriptors.
  /// @param jsettings    Task configuration.
  /// @param ctx          Runtime handles (sync stream).
  /// @return             New task ready for `ISyncTask::execute()`.
  virtual std::unique_ptr<ISyncTask> create(std::span<const TDesc> input_descs,
                                            const nlohmann::json  &jsettings,
                                            const SyncCreateCtx   &ctx) const = 0;

  /// Update or replace an existing synchronous task.
  ///
  /// Use when inputs/settings change. Reuse internal allocations if possible.
  /// Otherwise, construct a replacement and return it.
  ///
  /// @param old_task     Existing task to reuse or replace.
  /// @param input_descs  New input descriptors.
  /// @param jsettings    New configuration.
  /// @param ctx          Runtime handles (sync stream).
  /// @return             Updated task. `old_task` must not be used afterwards.
  virtual std::unique_ptr<ISyncTask> update(std::unique_ptr<ISyncTask> old_task,
                                            std::span<const TDesc>     input_descs,
                                            const nlohmann::json      &jsettings,
                                            const SyncCreateCtx       &ctx) const = 0;
};

/// Factory for asynchronous tasks.
class IAsyncTaskFactory : public ITaskFactory {
public:
  virtual ~IAsyncTaskFactory() = default;

  /// Create a new asynchronous task instance.
  ///
  /// @param input_descs  Upstream input tensor descriptors.
  /// @param jsettings    Task configuration.
  /// @param ctx          Runtime handles (producer and consumer streams).
  /// @return             New task ready for `try_push()/try_pop()`
  virtual std::unique_ptr<IAsyncTask> create(std::span<const TDesc> input_descs,
                                             const nlohmann::json  &jsettings,
                                             const AsyncCreateCtx  &ctx) const = 0;

  /// Update or replace an existing asynchronous task.
  ///
  /// @param old_task     Existing task to reuse or replace.
  /// @param input_descs  New input descriptors.
  /// @param jsettings    New configuration.
  /// @param ctx          Runtime handles (producer and consumer streams).
  /// @return             Updated task. `old_task` must not be used afterwards.
  virtual std::unique_ptr<IAsyncTask> update(std::unique_ptr<IAsyncTask> old_task,
                                             std::span<const TDesc>      input_descs,
                                             const nlohmann::json       &jsettings,
                                             const AsyncCreateCtx       &ctx) const = 0;
};

} // namespace holoflow::core