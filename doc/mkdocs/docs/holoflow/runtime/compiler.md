# Compiler

The compiler translates a user-authored `GraphSpec` into an executable runtime package:

- `GraphPlan`
- `std::vector<Section>`
- `ExecResouces`

Its public entry point is:

```cpp
holoflow::runtime::Compiler compiler(registry, config);
auto compiled = compiler.compile(graph_spec);
```

The same API also supports incremental updates:

```cpp
auto next = compiler.compile(updated_graph_spec, std::move(previous_output));
```

## `Compiler::Config`

The current public configuration fields are:

- `log_dir`
- `dump_dot_on_failure`
- `verbose_tracing`
- `enable_profiling`
- `trace_filename`

When `log_dir` is set, the compiler can emit:

- `compiler.log`
- graphviz dumps for success/failure
- Chrome trace JSON when profiling is enabled

## What the compiler guarantees

After successful compilation, the scheduler receives:

- a DAG with compiled node metadata
- validated input/output tensor descriptors for every node
- a tensor-to-storage mapping that respects in-place aliasing
- allocated or user-managed storage records
- instantiated tasks with bound loggers and storage adapters
- execution sections and CUDA streams

## Pass pipeline

The implementation is organized as a sequence of passes.

### 1. Validate spec

Checks performed up front:

- node names must be unique
- every node `kind` must exist in the registry
- a destination input slot may have only one incoming edge

The current compiler also rejects cycles later during topological sorting.

### 2. Build graph plan

The `GraphSpec` structure is copied into a `GraphPlan` skeleton:

- `NodeSpec` becomes `NodePlan::spec`
- `EdgeSpec` becomes `EdgePlan::spec`

At this stage there are still no inferred descriptors or tensor IDs.

### 3. Type inference

The compiler topologically sorts the graph and visits nodes in dependency order. For each node it:

1. collects input descriptors from incoming edges
2. calls `registry.get(kind).infer(input_descs, settings)`
3. stores the returned `InferResult` on the node
4. writes output descriptors back to outgoing edges

This is where most semantic validation happens. If a task factory rejects the settings or the upstream descriptors, compilation fails here.

### 4. Assign tensor IDs

Each node output receives a fresh tensor ID.

- `NodePlan::out_tids` stores those new IDs
- incoming edges propagate their producer tensor IDs into `NodePlan::in_tids`
- `ExecResouces::tensor_descs` records the descriptor for each tensor ID

### 5. Assign storage IDs

The compiler decides which logical tensors share physical memory.

- default case: each output gets a new storage ID
- in-place case: an output reuses the storage ID of the designated input

This uses `InferResult::in_place`.

### 6. Verify buffer consistency

The compiler checks that a tensor is not multiply owned. If several owned input/output declarations point at the same tensor ID, compilation fails.

This protects the runtime from ambiguous lifetime responsibility.

### 7. Allocate buffers

For each storage ID, the compiler either:

- allocates a fresh host/device block
- reuses a matching block scavenged from the previous compilation
- skips allocation entirely when the storage is user-managed through task ownership

Reused blocks are matched by:

- `MemLoc`
- exact byte size

### 8. Create storage adapters

Each node gets an internal `IOStorageAccess` adapter bound to its input and output tensor IDs. Tasks use this only when they implement owned I/O behavior.

### 9. Partition sections

The compiler groups nodes into execution sections:

- sync-to-sync connectivity merges nodes into the same section
- async nodes act as boundaries
- async producer and consumer roles are attached to neighboring sections

### 10. Assign streams

Each section gets a CUDA stream:

- from the previous compilation when possible
- otherwise from a fresh `curaii::CudaStream`

### 11. Instantiate tasks

For each node:

- sync nodes receive a `SyncCreateCtx { stream }`
- async nodes receive an `AsyncCreateCtx { producer_stream, consumer_stream }`

The compiler prefers `update(...)` over `create(...)` only when:

- a previous node with the same `name` exists
- the previous node has the same `kind`
- the previous task instance can be cast to the expected interface

Otherwise it falls back to `create(...)`.

### 12. Bind tasks

Each instantiated task is then bound to:

- its storage adapter
- a per-node logger

## Compiler output structures

### `NodePlan`

Each compiled node stores:

- `spec`
- `infer`
- `in_tids`
- `out_tids`

### `EdgePlan`

Each compiled edge stores:

- `spec`
- `desc`
- `tid`

### `ExecResouces`

The current runtime resource bundle includes:

- `memory_blocks`
- `storages`
- `tensor_descs`
- `tid_to_sid`
- `node_storage_adapters`
- `streams`
- `tasks`

`ExecResouces` is intentionally low-level. It is the physical backing store of the runtime.

## Pseudo algorithm: compiler main loop

```text
compile(graph_spec, prev_output):
  validate_spec(graph_spec, registry)
  graph_plan = copy_structure(graph_spec)

  topo = topological_sort(graph_plan)
  for node in topo:
    inputs = descriptors_from_in_edges(node)
    infer  = registry[node.kind].infer(inputs, node.settings)
    store infer on node
    propagate output descriptors to out edges

  assign_tensor_ids(graph_plan)
  assign_storage_ids(graph_plan, in_place_rules)
  verify_buffer_consistency(graph_plan)

  resources = allocate_or_reuse_storage(prev_output, graph_plan)
  sections  = partition_sections(graph_plan)
  assign_streams(sections, prev_output)
  instantiate_tasks(graph_plan, sections, resources, prev_output)
  bind_tasks(graph_plan, resources)

  return CompilerOutput{graph_plan, sections, resources}
```

## Pseudo algorithm: section partitioning

The current sectioning logic is well approximated by:

```text
parent[v] = v

for each edge u -> v:
  if u is sync and v is sync:
    union(u, v)

for each async node a:
  union all sync predecessors of a together
  union all sync successors of a together

create one section per union-find root that contains sync nodes

for each async node a:
  attach a as async_prod to each predecessor sync section
  attach a as async_cons to each successor sync section
```

This gives the scheduler explicit handoff points across async boundaries.

## Update semantics

Compilation with a previous output is not a diff engine over arbitrary graph changes. The effective reuse rules are pragmatic:

- same node name + same node kind: try `update(...)`
- name missing in previous graph: create fresh task
- same name but different kind: create fresh task
- reused memory blocks must match exact memory location and byte size
- streams are reused in section order as long as old streams are still available

That design keeps update logic simple and predictable.

## Failure modes

Compilation can fail because of:

- unknown task kind
- duplicate node names
- multiple edges targeting the same input slot
- cycles in the graph
- task-specific inference rejection
- invalid in-place aliasing
- conflicting ownership declarations
- allocation failure or CUDA failure

When `dump_dot_on_failure` is enabled and `log_dir` is configured, the compiler emits a DOT dump that is often the fastest way to inspect the failing plan.
