# Task Model

Holoflow tasks are executable graph vertices. A task is not discovered by reflection or by templates; it is always driven by a factory and a static inference contract.

## The key interfaces

The task system revolves around four families of types:

- execution contexts
  - `SyncCtx`
  - `AsyncPushCtx`
  - `AsyncPopCtx`
- task interfaces
  - `ITask`
  - `ISyncTask`
  - `IAsyncTask`
- factory interfaces
  - `ITaskFactory`
  - `ISyncTaskFactory`
  - `IAsyncTaskFactory`
- metadata
  - `InferResult`
  - `InPlace`
  - `TaskKind`
  - `OpResult`

## Static inference contract

Before a task instance exists, the compiler asks the factory to infer the task contract:

```cpp
InferResult infer(std::span<const TDesc> input_descs,
                  const nlohmann::json& jsettings) const;
```

The returned `InferResult` defines:

- expected inputs
- produced outputs
- whether inputs and outputs alias in place
- which slots are task-owned
- whether the node is sync or async

This is what lets the compiler allocate memory and validate the graph without executing anything.

## `InferResult`

`InferResult` contains:

- `input_descs`
- `output_descs`
- `in_place`
- `owned_inputs`
- `owned_outputs`
- `kind`

Each field is semantically important.

### `input_descs` and `output_descs`

These are full `TDesc` values, not just shapes. They define:

- dimensionality
- dtype
- memory location
- layout

### `in_place`

An entry `{in_idx, out_idx}` means:

> output slot `out_idx` reuses the storage of input slot `in_idx`

The compiler encodes this by giving the output tensor the same storage ID as the input tensor.

### `owned_inputs` and `owned_outputs`

These flags tell the runtime whether the scheduler or the task owns the storage lifecycle for that slot.

### `kind`

This decides whether the scheduler will call:

- `execute(...)`
- or the split `try_push(...)` / `try_pop(...)`

## Synchronous tasks

A synchronous task implements:

```cpp
class ISyncTask : public ITask {
public:
  virtual OpResult execute(SyncCtx& ctx) = 0;
};
```

Characteristics:

- one blocking call
- inputs and outputs are visible together
- ideal for transforms, readers, writers, and most kernels

`SyncCtx` contains:

- `inputs`
- `outputs`
- `cancelled`
- `event_writer`
- `event_reader`

## Asynchronous tasks

An asynchronous task implements:

```cpp
class IAsyncTask : public ITask {
public:
  virtual OpResult try_push(AsyncPushCtx& ctx) = 0;
  virtual OpResult try_pop(AsyncPopCtx& ctx) = 0;
};
```

Characteristics:

- decoupled producer and consumer phases
- internal state usually spans multiple scheduler iterations
- suited to queues, staging buffers, and batching

`AsyncPushCtx` contains inputs. `AsyncPopCtx` contains outputs.

## `OpResult`

`OpResult` is not an error transport. It is a control-flow signal:

- `Ok`
- `NotReady`
- `Cancelled`
- `Eof`

Interpretation:

- sync tasks should normally return `Ok`, `Cancelled`, or `Eof`
- async tasks may also return `NotReady`
- fatal problems should throw exceptions rather than being encoded as `OpResult`

## Ownership model

Ownership is the main advanced feature of the task system.

### Borrowed slots

If a slot is not owned:

- the scheduler provides the `TView`
- the task borrows it
- the task must not outlive or retain that view beyond the call boundary

### Owned input slots

If `owned_inputs[i] == true`, the scheduler asks the task for a writable view before running:

```cpp
std::optional<TView> acquire_input(int index);
```

### Owned output slots

If `owned_outputs[i] == true`, the task writes its own `TView` into the output slot and later receives:

```cpp
void release_output(int index);
```

This allows explicit buffer-pool and queue semantics.

## `ITask` services

`ITask` also exposes two bound runtime services:

- a logger via `bind_logger(...)`
- storage access via `bind_storage_access(...)`

The storage accessor is primarily relevant for tasks that own inputs or outputs and need to resolve the underlying storage records.

## Factory creation contexts

Creation is separated from inference. The factory receives runtime creation contexts:

### `SyncCreateCtx`

- `stream`

### `AsyncCreateCtx`

- `producer_stream`
- `consumer_stream`

This allows tasks to initialize stream-bound CUDA resources during creation or update.

## Update behavior

Factories may override:

- `ISyncTaskFactory::update(...)`
- `IAsyncTaskFactory::update(...)`

The default implementation simply recreates the task by delegating to `create(...)`.

Override `update(...)` when preserving state is worth the added complexity.

## Pseudo algorithm: scheduler interaction with a sync task

```text
if task owns some inputs:
  acquire_input for each owned input slot

prepare SyncCtx with input/output views
result = task.execute(ctx)

if task owns some outputs:
  after downstream use, call release_output for each owned output slot
```

## Pseudo algorithm: scheduler interaction with an async task

```text
producer side:
  acquire owned inputs if needed
  repeat try_push until Ok or terminal status

consumer side:
  repeat try_pop until Ok or terminal status
  release owned outputs after downstream use
```

## Practical rules of thumb

- Put all structural validation in `infer(...)`.
- Use sync tasks by default.
- Reach for async tasks only when you truly need decoupled state.
- Use ownership only when the scheduler-managed buffer model is insufficient.
- Use in-place aliasing only when the implementation is genuinely safe with overwritten input storage.
