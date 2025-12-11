## Tasks

Holoflow tasks are the executable vertices of a graph. Each task consumes a fixed
set of tensor views and produces another fixed set of tensor views. The runtime
uses the metadata inferred by the task factory to allocate buffers, reuse
storage, and orchestrate execution.

### Signatures Are Static

Task factories describe the full signature of a task before an instance is
constructed. The inference step returns an `InferResult` (see
`src/holoflow/include/holoflow/core/tasks.hh`) containing:

- `input_descs` / `output_descs`: tensor descriptors (`TDesc`) that lock shape,
  dtype, memory location, and strides.
- `in_place`: pairs of indices that alias the storage of an input and an output.
- `owned_inputs` / `owned_outputs`: whether the scheduler or the task owns the
  lifetime of each tensor slot (see below).
- `kind`: synchronous (`ISyncTask`) or asynchronous (`IAsyncTask`).

Once a task instance is created the parity, descriptors, and ownership flags do
not change. If the upstream topology changes the compiler asks the factory to
`update` (or recreate) the task.

### Runtime Contexts

At runtime the scheduler supplies spans of `TView` into the task context defined
in `src/holoflow/include/holoflow/core/tasks.hh`:

 - `SyncCtx` exposes `inputs`, `outputs`, a `cancelled` flag and two event
  handles: `event_writer` and `event_reader`. These allow a synchronous task
  to emit runtime events (UI telemetry, diagnostics) and to read events
  posted by other components. A synchronous task performs its work inside a
  single `execute` call and returns an
  `OpResult`.
- `AsyncPushCtx` provides the input span to the producer side of an asynchronous
  task (`try_push`).
- `AsyncPopCtx` provides the output span to the consumer side (`try_pop`).

The scheduler refreshes the spans before every call so the task always sees
up-to-date views. Long running work should periodically check `cancelled` and
return `OpResult::Cancelled` when requested.

### OpResult values

Tasks use `OpResult` to indicate control flow status rather than errors. At
this level you'll typically handle these cases returned by task operations:

- `OpResult::Ok` — operation completed and output(s) are available.
- `OpResult::NotReady` — no work available now; caller may retry later.
- `OpResult::Cancelled` — cooperative cancellation; caller should stop using
  the task.
- `OpResult::Eof` — end-of-stream for push/pop style tasks.

### In-place Reuse

In-place reuse means an output tensor shares the storage of one of the inputs.
Factories express this by filling `InferResult::in_place` with `(input, output)`
pairs. During execution the scheduler ensures both indices point to the same
`TView` instance.

```cpp
return InferResult{
    .input_descs   = {idesc},
    .output_descs  = {idesc},
    .in_place      = {{0, 0}},
    .owned_inputs  = {false},
    .owned_outputs = {false},
    .kind          = TaskKind::Sync,
};
```

The Fresnel diffraction task factory (`src/holovibes/src/tasks/syncs/fresnel_diffraction.cu:173`)
uses this contract to run a CUFFT plan while overwriting the input buffer. The
task implementation simply reads from `ctx.inputs[0]` and writes to
`ctx.outputs[0]`, both pointing to the same memory. In-place reuse avoids extra
allocations but requires the task to tolerate the overwritten input.

### Tensor Ownership

Ownership controls who is responsible for allocating and recycling the storage
referenced by a tensor view.

- **Non-owned slots (`false`)**: the scheduler provides the `TView`. The task
  must treat it as borrowed memory.
- **Owned inputs (`true`)**: the task instance owns the underlying memory. Before
  each execution the scheduler calls `ITask::acquire_input(index)` to request a
  writable view. If the task temporarily cannot provide a buffer it returns
  `std::nullopt` and the scheduler retries (see
  `src/holoflow/src/runtime/graph_exec.cc:276`).
- **Owned outputs (`true`)**: after a successful run the task writes its own view
  into `ctx.outputs[index]`. Downstream consumers borrow the view until the
  scheduler notifies completion via `ITask::release_output(index)`
  (`graph_exec.cc:296`).

The asynchronous batch queue (`src/holovibes/src/tasks/asyncs/batch_queue.cc`) is
an example of full ownership. Its inference result sets both ownership masks to
`true`, and the implementation manages a ring buffer: `acquire_input` hands out
the next writable slot, `try_push` advances the writer index, `try_pop` installs
the view for the reader, and `release_output` frees the consumed range.

Ownership is orthogonal to in-place reuse. A task may own the storage and still
declare an in-place mapping to recycle buffers across its own input/output
interface.

### Writing Synchronous Tasks

Subclass `ISyncTask` and implement `OpResult execute(SyncCtx &ctx)`. Inputs and
outputs are exposed as `std::span<TView>` so index into them directly. Validation
should happen in the factory; by the time `execute` runs the spans match the
inferred descriptors.

```cpp
class SqrtTask final : public ISyncTask {
public:
  [[nodiscard]] OpResult execute(SyncCtx &ctx) override {
    const auto elements = ctx.inputs[0].desc.num_elements();
    const auto *in      = reinterpret_cast<const float *>(ctx.inputs[0].data);
    auto       *out     = reinterpret_cast<float *>(ctx.outputs[0].data);

    for (size_t i = 0; i < elements; ++i) {
      out[i] = std::sqrt(in[i]);
    }
    return OpResult::Ok;
  }
};
```

Synchronous tasks always run on the caller thread. When heavy GPU work is
involved, the factory receives a `SyncCreateCtx` that carries the CUDA stream to
use (see the Fresnel diffraction factory).

### Writing Asynchronous Tasks

Asynchronous tasks split the producer (`try_push`) and consumer (`try_pop`) sides
in a single object implementing `IAsyncTask`. The scheduler alternates between
pushing inputs and popping outputs based on `OpResult` (typically retrying on
`NotReady`). The batch queue task demonstrates how ownership lets the task keep
stateful buffers between pushes and pops without scheduler involvement.

### Factories

Factories are responsible for three operations:

- `infer(...)`: validate settings and inputs, then populate an `InferResult` with
  descriptors, in-place relationships, ownership masks, and the task kind.
- `create(...)`: instantiate the task and allocate any internal resources. The
  create context (`SyncCreateCtx` or `AsyncCreateCtx`) carries runtime handles
  such as CUDA streams.
- `update(old_task, ...)`: optionally refresh an existing task by reusing
  expensive allocations. The default implementation destroys `old_task` and
  calls `create` again.

By following the inference contract and ownership hooks, tasks integrate
seamlessly with the Holoflow scheduler while retaining full control over their
internal memory management strategies.
