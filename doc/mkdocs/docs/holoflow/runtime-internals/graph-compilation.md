# Graph Compilation

Graph compilation turns a declarative `GraphSpec` into the graph, resources, and execution
sections consumed by the scheduler. This page describes the compiler as a sequence of passes. Each
pass establishes invariants on which later passes rely.

Compilation prepares tasks but does not execute the pipeline. After compilation, the application
must keep the returned `CompilerOutput` alive for as long as its scheduler uses the contained graph
and resources.

## Inputs and output

The compiler receives:

- a `GraphSpec` containing node names, task kinds, settings, and slot-to-slot edges;
- a `Registry` that maps task kinds to factories; and
- optionally, the previous `CompilerOutput` when recompiling an updated graph.

It produces a `CompilerOutput` with three parts:

| Part | Contents |
| --- | --- |
| `graph` | A `GraphPlan` enriched with inference results, tensor descriptors, and tensor IDs |
| `sections` | Independently scheduled regions and their synchronous and asynchronous work |
| `resources` | Storage objects, memory blocks, CUDA streams, task instances, and storage adapters |

The compiler uses several related identities:

- A **node name** identifies a task across compilation and graph updates.
- A **tensor ID** (TID) identifies a logical node output and all edges that consume it.
- A **storage ID** (SID) identifies the backing storage. Several tensors may share one SID through
  an in-place mapping.
- A **section ID** identifies work run by one scheduler thread on one CUDA stream.

For an output slot `(v, i)`, every edge that consumes that slot receives the same tensor ID:

\[
\operatorname{tid}(e) = \operatorname{out\_tid}(v, i)
\quad\text{when}\quad
e = (v, *, \operatorname{out\_idx}=i).
\]

For an in-place mapping from output `o` to input `i`:

\[
\operatorname{sid}(\operatorname{out\_tid}[o])
=
\operatorname{sid}(\operatorname{in\_tid}[i]).
\]

## Pass pipeline

Passes run in this fixed order:

| Phase | Passes | Principal result |
| --- | --- | --- |
| Structure | Validate Spec, Build Graph Plan | A valid mutable graph skeleton |
| Semantics | Type Inference | Task contracts and tensor descriptors |
| Storage | Tensor IDs, Storage Mapping, Buffer Consistency, Buffer Allocation, Storage Adapters | Addressable logical tensors and their storage |
| Execution | Section Partitioning, Stream Assignment | Independently scheduled work and CUDA streams |
| Materialization | Task Instantiation, Task Binding | Runtime tasks with their services bound |

Profiling output and Graphviz dumps surround this pipeline but do not change its semantic result.

## Validate Spec

**Purpose.** Reject structural errors that can be diagnosed without invoking task factories.

**Preconditions.** The input `GraphSpec` is available and the registry has been populated.

**Operation.** The pass verifies that node names are unique, every node kind is registered, and no
two edges target the same `(node, input index)` pair.

**Postconditions.** Names can safely key task and adapter maps, every node has a factory, and each
connected input slot has one producer at most.

**Rejected input.** Duplicate node names, unknown node kinds, and multiple edges targeting one
input slot are rejected here. Cycles and invalid slot indices are deliberately deferred to type
inference, where the compiler has the graph plan and factory contracts needed to diagnose them.

## Build Graph Plan

**Purpose.** Create the mutable representation enriched by the remaining passes.

**Preconditions.** The specification passed structural validation.

**Operation.** Each specification node becomes a `NodePlan` containing a copy of its `NodeSpec`.
Each specification edge becomes an `EdgePlan` containing a copy of its `EdgeSpec`. The inference,
descriptor, and ID fields are populated later.

**Postconditions.** `CompilerOutput::graph` has the same topology and specifications as the input
graph. No runtime resources exist yet.

## Type Inference

**Purpose.** Ask every factory for its task contract and propagate tensor descriptions downstream.

**Preconditions.** The graph plan mirrors the specification and all node kinds have registered
factories.

**Operation.** The compiler obtains a topological order, then visits producers before consumers.
For each node it places incoming edge descriptors at their declared input indices, calls the
factory's `infer(...)`, and copies inferred output descriptors to outgoing edges by output index.

```text
order = topological_sort(graph)          # throws if graph is cyclic

for node in producer_to_consumer(order):
    inputs = array(in_degree(node))

    for edge in incoming_edges(node):
        require edge.in_idx < inputs.size
        inputs[edge.in_idx] = edge.desc

    node.infer = registry[node.kind].infer(inputs, node.settings)

    for edge in outgoing_edges(node):
        require edge.out_idx < node.infer.output_descs.size
        edge.desc = node.infer.output_descs[edge.out_idx]
```

**Postconditions.** Every node has an `InferResult`. Every edge has the descriptor of the source
output it carries. The result also records task kind, owned slots, in-place mappings, and whether an
asynchronous task synchronizes its producer stream.

**Rejected input.** Cyclic graphs and edges whose input or output indices are out of bounds are
rejected. Factory-specific validation errors also surface from `infer(...)`.

!!! note "Input arity"
    The input descriptor array is sized from the node's incoming edge count. Consequently, connected
    input indices must form the range expected by the factory; a gap can make a higher input index
    out of bounds.

## Tensor IDs

**Purpose.** Give each logical output a stable identity inside the compiled result.

**Preconditions.** Type inference completed, so every node exposes its input and output descriptor
vectors and every edge carries a descriptor.

**Operation.** Nodes are visited producer-first. Each output slot receives a fresh TID, including
outputs with no consumers. That TID is copied to every outgoing edge selecting the output slot.
Incoming edge TIDs are then used to populate each consumer's indexed input TIDs. Tensor descriptors
are recorded in `resources.tensor_descs`.

**Postconditions.** Each node has complete `in_tids` and `out_tids` vectors. All consumers of one
logical output agree on its TID, and every assigned TID has a tensor descriptor.

## Storage Mapping

**Purpose.** Decide which logical tensors share backing storage.

**Preconditions.** Tensor IDs are assigned, and inference has declared every in-place mapping.

**Operation.** Nodes are visited producer-first. An ordinary output receives a fresh SID. An output
declared in-place reuses the SID already assigned to the mapped input.

```text
for node in producer_to_consumer(topological_sort(graph)):
    for output_index, output_tid in node.out_tids:
        mapping = in_place_mapping_for(output_index)

        if mapping exists:
            input_tid = node.in_tids[mapping.input_index]
            require input_tid already has a storage ID
            sid = tid_to_sid[input_tid]
        else:
            sid = next_fresh_storage_id()

        tid_to_sid[output_tid] = sid
```

**Postconditions.** Every output TID maps to one SID. In-place output tensors alias their mapped
inputs; all other outputs have distinct storage.

**Rejected input.** An in-place output is rejected if its mapped input has no assigned SID. Valid
in-place indices and compatible descriptors are part of the factory's inference contract.

## Buffer Consistency

**Purpose.** Ensure one task at most controls the lifetime of any storage object.

**Preconditions.** Owned input and output slots are known from inference, and TIDs map to SIDs.

**Operation.** The pass groups all task-owned input and output declarations by SID.

**Postconditions.** Every SID has zero or one declared owner.

**Rejected input.** Compilation fails when multiple owned slots resolve to the same SID. This can
happen when ownership and in-place aliasing combine into conflicting lifetime authorities. See
[Storage Ownership](../concepts/storage-ownership.md) for the runtime ownership protocol.

## Buffer Allocation

**Purpose.** Create stable storage objects and allocate compiler-owned memory.

**Preconditions.** Tensor descriptors and the TID-to-SID mapping are complete, and storage ownership
is unambiguous.

**Operation.** The compiler first identifies SIDs controlled by tasks and chooses a representative
TID for every SID. If a previous compiler output was supplied, its compiler-owned memory blocks form
a reuse pool keyed by `(memory location, byte size)`.

```text
owned_sids = storage_ids_of_all_owned_slots()
representative_tid = choose_one_tensor_per_sid()
reuse_pool = move_previous_blocks_grouped_by(memory_location, byte_size)

for sid, tid in representative_tid:
    desc = tensor_descs[tid]
    storage = Storage(desc.mem_loc, desc.num_bytes, null)

    if sid not in owned_sids:
        block = reuse_pool.take_exact_match(desc.mem_loc, desc.num_bytes)
                or allocate(desc.mem_loc, desc.num_bytes)
        storage.ptr = block.data
        memory_blocks[sid] = move(block)

    storages[sid] = move(storage)
```

Host descriptors receive host allocations and device descriptors receive device allocations. A
task-owned SID still receives a stable `Storage` object, but its pointer initially remains null; the
owning task publishes its storage during execution.

**Postconditions.** Every SID has a stable `Storage`. Every compiler-owned SID also has a matching
`MemoryBlock`, and its storage points at that block. Reused blocks have exactly the requested memory
location and byte size. Old blocks left in the reuse pool are released.

## Storage Adapters

**Purpose.** Give each task indexed access to the storage objects corresponding to its inputs and
outputs.

**Preconditions.** Node TID vectors and the resource maps from TID to SID to `Storage` are complete.

**Operation.** The compiler creates one `TaskStorageAdapter` per node. The adapter retains the
node's input and output TIDs and resolves them through the compiled resources.

**Postconditions.** `resources.node_storage_adapters` contains an adapter keyed by every node name.
The adapters are not exposed to tasks until the final binding pass.

## Section Partitioning

**Purpose.** Divide the graph into regions that the scheduler can run independently.

Synchronous nodes connected without crossing an asynchronous node belong to one section. An
asynchronous node forms a boundary: it participates as a producer in upstream sections and as a
consumer in downstream sections, rather than belonging to a section's synchronous sequence.

**Preconditions.** Every node's inferred task kind is known.

**Operation.** The pass uses disjoint sets to build synchronous components:

```text
reject every Async -> Async edge
create one disjoint set per graph node

for each Sync -> Sync edge:
    unite(source, target)

for each Async node:
    unite all of its synchronous predecessors
    unite all of its synchronous successors

for each Sync node in topological order:
    append node to the section representing its set

for each Async node:
    add node as consumer to each downstream synchronous section
    add node as producer to each upstream synchronous section

for each section:
    stably move producers that synchronize their producer stream first
```

Merging all synchronous predecessors of an asynchronous node ensures its producer side is invoked
by one section. The corresponding successor merge does the same for its consumer side. Each
section's `sync_topo` remains producer-to-consumer ordered.

For example:

```text
Source -> Upload -> Queue -> Compute -> Download -> Sink
        [ section 0 ]       [          section 1          ]
                       ^   ^
                       |   +-- Queue is an async consumer of section 1
                       +------ Queue is an async producer of section 0
```

**Postconditions.** Every synchronous node belongs to one section. Each asynchronous node is listed
in the relevant upstream `async_prod` and downstream `async_cons` collections. Synchronizing
producers precede ordinary producers, and `has_synchronizing_async_producer` records whether the
section contains one.

**Rejected input.** Direct edges between two asynchronous nodes are not currently supported.

## Stream Assignment

**Purpose.** Give each execution section the CUDA stream used by its synchronous work.

**Preconditions.** Section partitioning has produced the complete section list.

**Operation.** A fresh compilation creates one stream per section. During recompilation, streams
from the previous result are moved into new sections in map iteration order. If the new graph has
more sections, the compiler creates the additional streams.

**Postconditions.** Every section has a valid stream handle, and the owning `CudaStream` is stored
under that section's ID in `resources.streams`.

!!! note "Stream reuse is positional"
    Current recompilation reuses available streams by order; it does not match sections by name or
    graph identity. Tasks must receive their streams from the new creation context.

## Task Instantiation

**Purpose.** Create runtime tasks, or update reusable tasks from a previous compiled graph.

**Preconditions.** Factories, inferred contracts, sections, and section streams are available.

**Operation.** A synchronous task receives its section stream in `SyncCreateCtx`. An asynchronous
task receives the stream of its first synchronous predecessor and first synchronous successor as
its producer and consumer streams, respectively; either side may be null at a graph boundary.

When a previous output is available, reuse follows this decision:

```text
previous = previous_tasks.find(node.name)

if previous is absent:
    factory.create(...)
else if no previous graph node has the same name and kind:
    factory.create(...)
else if previous task cannot be cast to the required task interface:
    factory.create(...)
else:
    factory.update(move(previous), ...)

synchronize every non-null stream supplied in the creation context
```

The factory decides what state an update preserves. Synchronizing after `create(...)` or
`update(...)` ensures any initialization submitted to the supplied streams has completed before
compilation returns.

**Postconditions.** `resources.tasks` contains one task keyed by each node name. Each task was
created or updated against the newly inferred input descriptors, settings, and stream context.

## Task Binding

**Purpose.** Inject runtime services that tasks must not use during construction.

**Preconditions.** Tasks and per-node storage adapters have been created.

**Operation.** Every task is bound to its node's storage adapter and to a logger named from the task
kind and node name.

**Postconditions.** Compiled tasks can access logging and storage services during runtime methods.
Constructors and factory update methods must not assume these services are already bound. See the
[Holoflow Task Model](../concepts/task-model.md#the-common-task-interface) for the task-facing
contract.

## Incremental recompilation

Passing the previous `CompilerOutput` transfers ownership of reusable resources into compilation.
The new graph is still validated, inferred, and planned from scratch; reuse is an optimization, not
a shortcut around compiler passes.

Three resource classes can be reused:

- memory blocks with an exact memory-location and byte-size match;
- CUDA streams, assigned to new sections in order; and
- task objects with the same node name, task kind, and compatible sync or async interface.

Resources that cannot be reused are destroyed normally. Because reuse moves objects out of the
previous result, callers must not retain runtime references into that result while recompiling.

## Failure and diagnostics

Any pass may throw. The compiler logs the failing exception and can write a Graphviz snapshot to
`compilation_failure.dot` when a log directory and graph dumping are enabled. It then synchronizes
the CUDA device, clears the last CUDA error, flushes the compiler log, and rethrows the exception.

On success, the same graph-dump setting produces `compilation_success.dot`. When profiling is
enabled, the compiler records pass and detailed timings, logs a summary, and writes Chrome trace
events when a log directory is configured. These diagnostic stages observe the compiler state but
do not establish scheduler invariants.

## Final compiler invariants

Before a `CompilerOutput` is handed to the scheduler:

- the graph is acyclic and each node has a registered factory and inferred contract;
- every logical output has a tensor descriptor, TID, and SID;
- each SID has at most one task owner and a stable `Storage` object;
- compiler-owned storage points to a correctly located and sized memory block;
- each synchronous node belongs to one topologically ordered section;
- asynchronous nodes appear on the producer and consumer sides of their adjacent sections;
- every section owns one CUDA stream;
- every node has a task, storage adapter, and logger; and
- task initialization submitted to compiler-provided streams has completed.

The scheduler can therefore concentrate on execution: constructing tensor views, running section
threads, enforcing asynchronous boundaries, and applying the ownership protocol.

## Implementation map

The pass driver and implementations are in `src/holoflow/src/runtime/compiler.cc`. Public compiler
configuration and output types are declared in `src/holoflow/include/holoflow/runtime/compiler.hh`;
`GraphPlan`, `ExecResouces`, and `Section` are declared in
`src/holoflow/include/holoflow/runtime/graph_exec.hh`.

The most direct behavioral tests are `test/holoflow/compiler_test.cc`,
`test/holoflow/compiler_additional_test.cc`, and the compiler-related cases in
`test/holoflow/scheduler_functional_test.cc`.
