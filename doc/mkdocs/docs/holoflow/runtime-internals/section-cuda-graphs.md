# Section CUDA graphs

Holoflow can replace a section's synchronous task loop with one `cudaGraphLaunch`.
Acquisition, asynchronous consumers/producers, barriers and output release still run on the host.
SlidingAverage's producer kernel remains outside the graph.

## Configuration

`Compiler::Config::max_section_cuda_graphs` defaults to **4,096 per section**. Set it to **0** to
disable section graphs. Holovibes reads `HOLOFLOW_MAX_SECTION_CUDA_GRAPHS` when compiling a pipeline
to override this value. Queue capacities and processing windows are never changed automatically.

All synchronous tasks must opt in, and their storage must have finite known pointer domains.
Unsupported tasks, unknown domains and oversized products select ordinary execution for the
entire section. Other sections remain eligible. Task-local graphs remain available in fallback.

Compilation inspects all sections but does not instantiate executables. Every scheduler start,
including resume, refreshes domains and sequences from paused task state and eagerly prepares all
eligible sections before creating worker threads. Matching executables are reused within the same
compiled resource generation; obsolete variants are discarded before missing ones are created.
There is no first-use capture or per-launch parameter update. Distinct storage IDs contribute
dimensions; tensor aliases share a dimension. Compiler-owned storage contributes one pointer.
Only storage referenced by synchronous tasks contributes, excluding unused async output ports.

The cap applies after sequence pruning. It limits graph count, not memory bytes; memory depends
on task complexity, and counts add across sections.

## Diagnostics

Inspection gathers every referenced storage and every task blocker, even for disabled or
incompatible sections. Each storage reports its owner task/port, tensor aliases, declared count,
enumerated count when available, and optional sequence. Skipped enumeration and unknown counts
are explicit; they are not represented as zero. Oversized or incompatible sections do not enumerate
potentially enormous pointer domains just to obtain diagnostics.

With `log_dir` configured, `section_cuda_graphs.json` is written at compilation, each start and
shutdown. Reports survive graph fallback and are written before graph-preparation errors propagate.
They include raw Cartesian and pruned counts (with unknown/overflow/limit status), blockers,
failure stage and recording task, inspection/construction/preparation time, nodes, and executable
reuse/create/discard counts. `planned` at compilation does not mean executables have been built;
`ready` is reported only after successful start-time preparation.

Normal logs summarize sections, storage counts, transitions and shutdown counters. Raw addresses
are debug-only. `Scheduler::section_graph_diagnostics()` returns thread-safe JSON snapshots with
graph launches, ordinary iterations, pointer/tuple misses and refresh counts. Counters are cumulative
for the compiled graph lifetime, including resumes. There is no per-frame diagnostic logging.

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

## Optional pointer order

Owners can additionally override `owned_input_pointer_sequence(index)` and
`owned_output_pointer_sequence(index)`. The default is `nullopt`: an unordered domain whose
Cartesian combinations remain necessary. A `PointerSequence` contains pointer-list indices:
`prefix` is used once, followed by the nonempty `cycle` forever. Repeated indices are allowed.
Indices must lie within the declared domain. Missing metadata is supported; invalid metadata
is an error.

The sequence describes bindings starting at the next successful use from the task's current paused
state. Queries must be read-only. Failed acquisition/pop attempts do not advance its logical clock.
Each complete section iteration uses each referenced owning phase once; ordering is combined only
within that section, not between independent producer and consumer sections.

The planner enumerates the longest startup prefix and the least common multiple of cycle lengths,
then deduplicates pointer tuples. Unspecified domains remain independent Cartesian dimensions.
For example, cycles of length 4 and 6 produce 12 reachable pairs rather than 24. Initial phases
matter. Checked arithmetic and a one-million-step planning budget bound sequence simulation;
if analysis exceeds that budget, the planner conservatively uses Cartesian domains and reports why.

BatchQueue declares aligned cyclic writer/reader positions. DualReaderBatchQueue also describes
its delayed-reader scratch prefix. SlidingAverage declares its output cycle and a fixed-discard
input prefix when no validity input exists. With data-dependent validity, its input order remains
unspecified. None of these declarations change queue execution behavior.

## Replay, lifetime and metrics

After acquisition and async consumption, the scheduler maps current addresses to a variant.
An unexpected pointer or a tuple outside the prepared combinations disables that section's graphs
before submission and resumes ordinary execution. Diagnostics identify the storages and domain
indices involved. Recording/instantiation failures discard partial sets; a poisoned CUDA context cannot
use fallback. Launch errors never trigger duplicate ordinary execution.

Every start replans from current queue phases, so partial advancement during cooperative shutdown
does not by itself cause fallback after resume. Matching variants are retained. Recompilation drains
prior streams and destroys executables before
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

| Section | Cartesian | Pruned |
| --- | ---: | ---: |
| Conversion → Reshape | 792 | 264 |
| PCA → Filter2D | 21 | 21 |
| Fresnel → Mean → FFTShift → Flatfield → Mean | 4,032 | 504 |
| PctClip → Conversion | 2,376 | 792 |
| Total | 7,221 | 1,581 |

The regular test uses smaller images, queues and windows with an explicit 128 cap.
Run the full-size equivalence/performance comparison separately:

```powershell
ctest --test-dir build/msvc-multi -C Release -R 'SectionGraphReference.PerformanceFullReference' -V
```

It compares output images with ordinary execution and prints compilation/start time, allocated device
memory and steady output throughput. Memory includes task buffers and library resources, so
differences approximate graph overhead rather than measuring it in isolation.
