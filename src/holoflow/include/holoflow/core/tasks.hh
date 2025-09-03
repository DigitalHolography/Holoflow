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

/// @file task.hh
/// @brief Interfaces and runtime contexts for Holoflow tasks.
///
/// This header defines the common interfaces and runtime contexts used by
/// synchronous and asynchronous tasks in Holoflow. A task is a computation unit
/// that consumes zero or more input tensors and produces zero or more output
/// tensors. Tasks may either be synchronous (blocking execution) or
/// asynchronous (decoupled push/pop).
///
/// @section overview Overview
/// - @ref SyncCtx, @ref AsyncPushCtx, @ref AsyncPopCtx: runtime contexts passed
///   to task operations.
/// - @ref OpResult: possible control-flow outcomes of a task operation.
/// - @ref ITask: base interface providing acquisition/release hooks for owned
///   tensors.
/// - @ref ISyncTask: interface for synchronous tasks, exposing @ref execute().
/// - @ref IAsyncTask: interface for asynchronous tasks, exposing @ref
///   try_push() and @ref try_pop().
/// - @ref InferResult: result structure returned by factory inference,
///   describing tensor descriptors, ownership flags, and in-place links.
///
/// @section ownership Ownership model
/// - By default, tensor views in contexts are non-owning views into externally
///   allocated buffers.
/// - Some tasks may require ownership of certain inputs or outputs. For those
///   indices:
/// - The scheduler calls @ref ITask::acquire_input() to obtain a writable
///   view into an owned input slot.
/// - The task fills @ref SyncCtx::outputs or @ref AsyncPopCtx::outputs with
///   mirrors into owned output buffers when producing results.
/// - The scheduler must later call @ref ITask::release_outputs() to release
///   the last produced outputs when it is done forwarding them downstream.
/// - This design assumes SPSC (single producer/single consumer) semantics, so
///   only one acquired input slot and one unreleased output group exist at any
///   given time.
///
/// @section todos TODOs
/// - Clarify the policy when an owned input has been acquired via
///   @ref ITask::acquire_input() but execution should be cancelled before
///   @ref IAsyncTask::try_push() is called. Possible options:
/// - Add an explicit @c abort_input(index) to roll back WRITING→FREE state.
/// - Implicitly discard uncommitted slots on cancellation.
/// - Require scheduler to @c try_push() and let the task drop the slot.
/// - Extend API for multi-slot aggregation tasks (N→1), where explicit commit
///   may be needed.
/// - Define how EOF interacts with owned inputs that were acquired but not yet
///   pushed.
///
/// @section error_handling Error handling
/// - Errors are signalled via exceptions for invalid indices, shape mismatches,
///   or internal failures.
/// - @ref OpResult is used strictly for control-flow outcomes: not-ready,
///   cancellation, or end-of-stream.

#pragma once

#include <atomic>
#include <cuda_runtime.h>
#include <optional>
#include <span>
#include <vector>

#include "holoflow/core/tensor.hh"

namespace holoflow::core {

/// Runtime execution context for a synchronous task.
struct SyncCtx {
  std::span<TView>   inputs;    ///< Input tensor views
  std::span<TView>   outputs;   ///< Output tensor views
  std::atomic<bool> *cancelled; ///< Cancellation flag
};

/// Runtime execution context for an asynchronous task push operation.
struct AsyncPushCtx {
  std::span<TView>   inputs;    ///< Input tensor views
  std::atomic<bool> *cancelled; ///< Cancellation flag
};

/// Runtime execution context for an asynchronous task pop operation.
struct AsyncPopCtx {
  std::span<TView>   outputs;   ///< Output tensor views
  std::atomic<bool> *cancelled; ///< Cancellation flag
};

/// Possible expected outcomes of a task operation.
enum class OpResult : uint8_t {
  Ok,        ///< Operation succeeded
  NotReady,  ///< Operation not ready to proceed
  Cancelled, ///< Operation was cancelled
  Eof,       ///< End of data stream
};

/// Interface for a task. A task is a computation unit that processes zero,
/// one, or more input tensors and produces zero, one, or more output tensors.
/// A task may be synchronous or asynchronous. See \ref ISyncTask and \ref
/// IAsyncTask for more details.
/// By default, inputs and outputs are assumed to be non-owning views into
/// external tensor data. However, tasks may also take ownership of one or
/// more input or output tensors. In that case, the task should use
/// - \ref ITask::acquire_input to expose a view into the owned tensor data,
/// - \ref ITask::release_outputs to release ownership of the output tensor
///   data.
/// Releasing inputs and acquiring outputs are specific to the task kind
/// (Sync/Async).
class ITask {
public:
  virtual ~ITask() = default;

  /// Acquire a view into an input tensor.
  /// The release of the view is specific to the task kind (Sync/Async).
  /// @param index The index of the input tensor.
  /// @return A view into the input tensor, or an empty optional if it cannot be
  /// acquired yet.
  [[nodiscard]] virtual std::optional<TView> acquire_input(int index);

  /// Release a view into an output tensor.
  /// The allocation of the view is specific to the task kind (Sync/Async).
  /// @param index The index of the output tensor.
  virtual void release_outputs(int index);
};

/// Interface for a synchronous task. A synchronous task is a computation unit
/// that processes input tensors and produces output tensors in a blocking
/// manner.
///
/// The lifecyle of this task should be as follow:
/// - acquire all owned inputs
/// - use inputs (owned or not)
/// - update owned inputs in context
/// - execute the task
/// - use outputs (owned or not)
/// - release all owned outputs
class ISyncTask : public ITask {
public:
  virtual ~ISyncTask() = default;

  /// Execute the task. If the task has owned input tensors, this function is
  /// responsible for releasing them. They cannot be used after this point.
  /// If the task has owned output tensors, this function is responsible for
  /// acquiring them and updating the output views in the context.
  [[nodiscard]] virtual OpResult execute(SyncCtx &ctx) = 0;
};

/// Interface for an asynchronous task. An asynchronous task is a computation
/// unit that processes input tensors and produces output tensors in a
/// non-blocking manner. The input consumption and output production may
/// occur concurrently.
///
/// The lifecyle of the producer should be as follow:
/// - acquire all owned inputs
/// - use inputs (owned or not)
/// - update owned inputs in context
/// - execute the task push operation
///
/// The lifecyle of the consumer should be as follow:
/// - execute the task pop operation
/// - use outputs (owned or not)
/// - release all owned outputs
class IAsyncTask : public ITask {
public:
  virtual ~IAsyncTask() = default;

  /// Try to push input tensors into the task. If the task has owned input
  /// tensors, this function is responsible for releasing them. They cannot be
  /// used after this point.
  /// @param ctx The context for the push operation.
  /// @return The result of the push operation.
  [[nodiscard]] virtual OpResult try_push(AsyncPushCtx &ctx) = 0;

  /// Try to pop output tensors from the task. If the task has owned output
  /// tensors, this function is responsible for acquiring them and updating the
  /// output views in the context.
  /// @param ctx The context for the pop operation.
  /// @return The result of the pop operation.
  [[nodiscard]] virtual OpResult try_pop(AsyncPopCtx &ctx) = 0;
};

/// Describes an in-place link between an input and output tensor.
struct InPlace {
  int in_idx;  ///< Index of the input tensor
  int out_idx; ///< Index of the output tensor
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
};

} // namespace holoflow::core