# Authoring Tasks

This guide explains how to implement a new Holoflow task so that it compiles, executes, and updates correctly.

## Step 1: choose the task kind

Use a synchronous task when the operator is naturally a single call:

- pure transforms
- file readers that produce immediately
- GPU kernels launched and completed as one step

Use an asynchronous task when the operator has decoupled write/read phases:

- internal queues
- buffering and batching stages
- producer/consumer adapters

## Step 2: define the factory contract

Every task is created through a factory. The factory is the compiler-facing object; the task instance is the scheduler-facing object.

### Factory responsibilities

- validate `settings`
- validate the incoming tensor descriptors
- return the full `InferResult`
- create the concrete task instance
- optionally update a previous task instance

## Step 3: implement `infer(...)`

`infer(...)` is the most important method because it defines the static contract of the node.

It must fill:

- `input_descs`
- `output_descs`
- `in_place`
- `owned_inputs`
- `owned_outputs`
- `kind`

### Example: simple pass-through sync operator

```cpp
holoflow::core::InferResult MyFactory::infer(
    std::span<const holoflow::core::TDesc> input_descs,
    const nlohmann::json& settings) const {

  if (input_descs.size() != 1) {
    throw std::invalid_argument("MyTask expects exactly one input");
  }

  (void)settings;

  return {
      .input_descs   = {input_descs[0]},
      .output_descs  = {input_descs[0]},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}
```

## Step 4: implement the task body

### Synchronous task

```cpp
class MyTask final : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx& ctx) override {
    const auto* in  = reinterpret_cast<const float*>(ctx.inputs[0].data());
    auto* out       = reinterpret_cast<float*>(ctx.outputs[0].data());
    const auto n    = ctx.inputs[0].desc.num_elements();

    for (size_t i = 0; i < n; ++i) {
      out[i] = in[i];
    }
    return holoflow::core::OpResult::Ok;
  }
};
```

### Asynchronous task

An async task maintains internal state across two methods:

- `try_push(...)`: accept or stage input
- `try_pop(...)`: publish output when ready

## Step 5: implement `create(...)`

The create context carries runtime handles:

- sync task: one CUDA stream
- async task: producer and consumer streams

Typical uses:

- creating CUDA plans
- allocating task-local scratch memory
- opening device/file handles
- storing the stream in the task instance

## Step 6: optionally implement `update(...)`

The default `update(...)` implementation simply calls `create(...)` again. Override it only when preserving expensive state matters:

- FFT plans
- persistent queues
- large internal buffers
- external device handles

The compiler will call `update(...)` only when node name and node kind still match between the previous and current graph.

## Ownership patterns

### Borrowed I/O

This is the common case:

- `owned_inputs[i] = false`
- `owned_outputs[i] = false`

The scheduler provides all views and owns the backing storage.

### Owned input

Use owned inputs when the task wants to hand the scheduler a writable buffer from its own pool before execution.

Implementation sketch:

```cpp
std::optional<holoflow::core::TView> acquire_input(int index) override;
```

Typical uses:

- staging into a preallocated ring buffer
- task-local DMA or pinned-memory pools

### Owned output

Use owned outputs when the task publishes output views that it owns and later recycles.

Implementation sketch:

```cpp
void release_output(int index) override;
```

Typical uses:

- async queues
- batchers
- shared scratch pools with task-defined reuse

## In-place tasks

If an output may alias an input, express that in `InferResult::in_place`.

Example:

```cpp
return {
    .input_descs   = {desc},
    .output_descs  = {desc},
    .in_place      = {{0, 0}},
    .owned_inputs  = {false},
    .owned_outputs = {false},
    .kind          = holoflow::core::TaskKind::Sync,
};
```

This tells the compiler to map output slot `0` onto the same storage as input slot `0`.

## Event-aware tasks

Synchronous tasks can use:

- `ctx.event_writer`
- `ctx.event_reader`

This is useful for:

- reporting measurements
- consuming UI control events
- emitting debugging annotations

## Pseudo algorithm: factory inference

```text
infer(inputs, settings):
  validate number of inputs
  validate shapes / dtypes / memory locations
  derive output descriptors
  choose sync vs async
  choose borrowed vs owned slots
  choose in-place aliases if safe
  return InferResult
```

## Pseudo algorithm: owned async queue

```text
state:
  ring buffer of N slots
  write index
  read index
  occupancy count

acquire_input(i):
  if buffer full: return nullopt
  return writable slot at write index

try_push(ctx):
  commit write index
  occupancy += 1
  return Ok

try_pop(ctx):
  if occupancy == 0: return NotReady
  publish readable slot at read index into ctx.outputs[0]
  return Ok

release_output(i):
  advance read index
  occupancy -= 1
```

## Common mistakes

- Doing validation in `execute(...)` instead of in `infer(...)`
- Returning `NotReady` from a synchronous task
- Declaring in-place aliasing when the implementation cannot tolerate overwritten input
- Marking several tasks as owners of the same tensor
- Forgetting that acquired owned inputs are transient
- Assuming `update(...)` is always called on graph changes

## Registration

Once implemented, register the factory with a unique kind string:

```cpp
holoflow::core::Registry registry;
registry.register_sync("MyTask", std::make_unique<MyTaskFactory>());
```

or

```cpp
registry.register_async("MyQueue", std::make_unique<MyQueueFactory>());
```

Kinds are unique across both sync and async registries.
