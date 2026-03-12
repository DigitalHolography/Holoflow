# Scheduler

`holoflow::runtime::Scheduler` executes a compiled graph. It does not compile or mutate the plan; it consumes the immutable `GraphPlan` and the mutable runtime resources produced by the compiler.

## Public API

```cpp
holoflow::runtime::Scheduler scheduler(graph, sections, resources);

scheduler.start();
scheduler.request_stop();
scheduler.wait();
```

Additional runtime APIs:

- `is_running()`
- `stop_requested()`
- `set_metrics_interval(...)`
- `metrics()`
- `ui_try_send(node_id, json)`
- `ui_try_receive()`

## Threading model

When started, the scheduler launches:

- one thread per `Section`
- one extra router thread for event dispatch
- one metrics thread when metrics are enabled

Each section thread repeatedly executes its own local scheduling loop.

## Section loop

The current runtime executes each section in six phases:

1. acquire owned inputs for sync nodes and async producers
2. execute async consumers
3. execute synchronous nodes in topological order
4. synchronize the section CUDA stream
5. execute async producers
6. release owned outputs for sync nodes and async consumers

This ordering is the core of the runtime.

## Pseudo algorithm: section execution

```text
while not stop_requested:
  for sync node in section.sync_topo:
    acquire all owned inputs

  for async node in section.async_prod:
    acquire all owned inputs

  for async node in section.async_cons:
    repeatedly try_pop until Ok / Cancelled / Eof / stop requested

  for sync node in section.sync_topo:
    execute once

  cudaStreamSynchronize(section.stream)

  for async node in section.async_prod:
    repeatedly try_push until Ok / Cancelled / Eof / stop requested

  for sync node in section.sync_topo:
    release owned outputs

  for async node in section.async_cons:
    release owned outputs
```

## Synchronous node semantics

For `ISyncTask`:

- the scheduler calls `execute(SyncCtx&)`
- `OpResult::Ok` means normal completion
- `OpResult::Cancelled` requests global stop
- `OpResult::Eof` requests global stop
- `OpResult::NotReady` is considered invalid for synchronous tasks and causes stop

That last point is important: `NotReady` is reserved for async retry loops.

## Asynchronous node semantics

For `IAsyncTask` the scheduler splits execution:

- consumer side: `try_pop(AsyncPopCtx&)`
- producer side: `try_push(AsyncPushCtx&)`

If the task returns `OpResult::NotReady`, the scheduler retries until:

- the task becomes ready
- stop is requested

This is effectively a polling protocol.

## Cancellation model

Cancellation is cooperative.

- Every runtime context receives `std::atomic<bool>* cancelled`
- `request_stop()` sets the scheduler stop flag
- tasks should periodically check the flag and return `OpResult::Cancelled` when appropriate

The scheduler also stops itself if a task returns:

- `Cancelled`
- `Eof`

## Metrics

The scheduler records per-node samples and periodically aggregates them into:

- `average_duration_ms`
- `runs_per_second`
- `host_throughput_bytes_per_second`
- `device_throughput_bytes_per_second`
- `sample_count`

Metrics are sampled from the sizes of input and output `TView`s and the observed wall-clock time around task calls.

## UI event path

The scheduler owns a `holoflow_event::Router`.

- each node receives named router handles at initialization
- synchronous tasks can emit and receive JSON events through `SyncCtx`
- the application can inject events with `ui_try_send(...)`
- the application can read node events with `ui_try_receive()`

This is useful for:

- debugging
- UI controls
- task-local telemetry
- runtime tuning messages

## Ownership handling

Ownership changes how the scheduler treats I/O slots.

### Owned inputs

Before running a node, the scheduler asks the task for owned inputs:

```cpp
std::optional<TView> acquire_input(int index);
```

If the task cannot provide a buffer yet, it returns `std::nullopt`, and the scheduler retries until stop is requested.

### Owned outputs

After downstream consumers are done, the scheduler tells the task it may recycle the produced output:

```cpp
void release_output(int index);
```

This is how queue-like or pool-backed tasks can manage their own memory lifecycle.

## Scheduler invariants

Several assumptions shape the current implementation:

- the compiled graph is already valid
- section-local sync nodes are already topologically ordered
- async tasks use a single-producer/single-consumer model
- section stream synchronization happens before async producers run
- the compiled output outlives the scheduler

## Common operational patterns

### Long-running application loop

```cpp
scheduler.start();

while (app_running) {
  auto metrics = scheduler.metrics();
  auto event   = scheduler.ui_try_receive();
  /* UI / monitoring work */
}

scheduler.request_stop();
scheduler.wait();
```

### Recompile and restart

Because the scheduler borrows the compiled artifacts, a graph update typically follows this pattern:

```cpp
scheduler.request_stop();
scheduler.wait();

compiled = compiler.compile(updated_graph, std::move(compiled));

holoflow::runtime::Scheduler next(compiled->graph, compiled->sections, compiled->resources);
next.start();
```

## Practical caveats

- `start()` is non-blocking.
- `request_stop()` is non-blocking.
- `wait()` is the synchronization point that joins section threads.
- If a sync task returns `NotReady`, that is treated as runtime misuse.
- If an async task never becomes ready and never observes cancellation, the section thread will spin.
