# Section CUDA graphs

Holoflow can replace a section's synchronous task loop with one `cudaGraphLaunch`.
Acquisition, asynchronous consumers/producers, barriers and output release still run on the host.
SlidingAverage's producer kernel remains outside the graph.

## Configuration

`Compiler::Config::max_section_cuda_graphs` defaults to **128 per section**. Set it to **0** to
disable section graphs. Holovibes reads `HOLOFLOW_MAX_SECTION_CUDA_GRAPHS` when compiling a pipeline
to override this value. Queue capacities and processing windows are never changed automatically.

All synchronous tasks must opt in, and their storage must have finite known pointer domains.
Unsupported tasks, unknown domains and oversized products select ordinary execution for the
entire section. Other sections remain eligible. Task-local graphs remain available in fallback.

Eligible sections eagerly instantiate the entire Cartesian product during compilation: there is
no first-use capture, eviction or per-launch parameter update. Distinct storage IDs contribute
dimensions; tensor aliases share a dimension. Compiler-owned storage contributes one pointer.
Only storage referenced by synchronous tasks contributes, excluding unused async output ports.

Compilation logs report counts, construction time and fallback reasons. With `log_dir` configured,
`section_cuda_graphs.json` also records the enumerated domain sizes. The cap limits graph count,
not memory bytes; memory depends on task complexity, and counts add across sections.

## Optional task interfaces

Existing tasks need no changes. Graph-compatible synchronous tasks override
`supports_cuda_graph() const noexcept` and `record_cuda_graph(CudaGraphCtx&)`.
Recording appends the same GPU work as ordinary execution, using the provided stream which is
already capturing into the top-level graph. Do not launch cached executables, start nested capture,
or insert child-graph nodes. Conditional nodes and their required body graphs are supported.
Child-graph nodes are rejected recursively, including inside conditional bodies.

For explicit nodes, use `CudaGraphCtx::dependencies` to obtain the capture frontier and edge data,
append nodes with those dependencies, and call `set_dependencies` with the new frontier.

Recording receives private storage bindings and must not retain views, alter live storage,
execute a frame, acquire/publish/release buffers, consume events or advance host-side state.
Initialize plans and workspaces during construction/update. All referenced resources must remain
valid until graph destruction. Repeated recording must leave ordinary execution behavior intact.

For fixed bindings and configuration, replay must work without a host task call. GPU-maintained
state and device-selected conditional branches are allowed. Host-selected internal variants,
per-iteration host updates, host control results, and synchronous owned-output publication are
not supported by section graphs in this version.

## Owned pointer domains

Factories may fill `InferResult::owned_input_pointer_counts` and `owned_output_pointer_counts`.
A supplied vector has one entry per corresponding port. Empty vectors or `nullopt` entries mean
unknown. After construction/binding, `ITask::owned_input_pointers(index)` and
`owned_output_pointers(index)` return optional vectors of **storage-base addresses**, before
`TDesc::offset`. Enumeration never acquires buffers or changes state.

Values must be unique, non-null, match their declared count, and form a complete stable set
throughout the compiled lifetime. Malformed declarations fail compilation; missing enumeration
causes fallback. Null pointers between ownership phases are not domain members.

BatchQueue enumerates aligned write/read positions in the actual padded allocation.
DualReaderBatchQueue additionally includes its delayed reader's startup scratch buffer when
the delay is nonzero. SlidingAverage enumerates its full ring.

## Replay, lifetime and metrics

After acquisition and async consumption, the scheduler maps current addresses to a variant.
An unexpected pointer disables that section's graphs before submission and resumes ordinary
execution. Recording/instantiation failures discard partial sets; a poisoned CUDA context cannot
use fallback. Launch errors never trigger duplicate ordinary execution.

Stop/resume retains variants. Recompilation drains prior streams and destroys executables before
updating tasks, modules, workspaces or allocations. Callers must stop/wait before recompiling and
keep `CompilerOutput` alive throughout scheduler use. Cancellation drains submitted graph work
before releasing its buffers even when the usual producer barrier is skipped.

Node metrics retain counts and throughput, with `individual_timing_available=false` for intervals
containing graph launches. Individual durations are unavailable, not measured zero cost.
`Scheduler::section_graph_metrics()` reports graph submission counts and host launch duration,
not GPU execution time. Aggregation derives node counts from section counters, avoiding a per-node
loop during graph replay.

## Reference validation

The portable fixture `test/holotask/fixtures/section_cuda_graph_reference.json` preserves the
supplied compute pipeline and 64-frame windows, substitutes deterministic test endpoints, and
reduces the final queue capacity from 64 to 32. With a **4,096** cap:

| Section | Variants |
| --- | ---: |
| Conversion → Reshape | 792 |
| PCA → Filter2D | 21 |
| Fresnel → Mean → FFTShift → Flatfield → Mean | 4,032 |
| PctClip → Conversion | 2,376 |

The regular test uses smaller images, queues and windows to exercise wraparound below the 128 cap.
Run the full-size equivalence/performance comparison separately:

```powershell
ctest --test-dir build/msvc-multi -C Release -R 'SectionGraphReference.PerformanceFullReference' -V
```

It compares output images with ordinary execution and prints compilation time, allocated device
memory and steady output throughput. Memory includes task buffers and library resources, so
differences approximate graph overhead rather than measuring it in isolation.
