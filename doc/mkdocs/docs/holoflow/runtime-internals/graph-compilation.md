# Graph Compilation

Graph compilation transforms a declarative graph into the execution plan and resources consumed by
the scheduler. It prepares the pipeline but does not execute it.

This page specifies compilation as a sequence of partial transformations. Each pass either enriches
the compiler state, establishes a property required by later passes, or rejects compilation.
Equations describe the intended compiler contract; implementation restrictions and trusted
assumptions are called out explicitly.

!!! warning "CompilerOutput lifetime"
    The returned `CompilerOutput` owns objects referenced by the scheduler. It must remain alive for
    the entire scheduler lifetime.

## Compilation pipeline

Compilation proceeds through twelve passes:

| $j$ | Pass $P_j$ | State added or property established |
| ---: | --- | --- |
| 1 | Validate Spec | Names, registry keys, and destination slots are structurally valid |
| 2 | Build Graph Plan | The mutable graph preserves the specification topology and ports |
| 3 | Type Inference | DAG order, task contracts, and tensor descriptors |
| 4 | Tensor IDs | Output slots and their consumers share logical tensor identities |
| 5 | Storage Mapping | TIDs map to SIDs; in-place outputs alias their inputs |
| 6 | Buffer Consistency | Every SID has at most one task owner |
| 7 | Buffer Allocation | Stable `Storage` objects and compiler-owned memory blocks |
| 8 | Storage Adapters | Node slots resolve to their `Storage` objects |
| 9 | Section Partitioning | Synchronous nodes form independently scheduled sections |
| 10 | Stream Assignment | Every section owns a CUDA stream |
| 11 | Task Instantiation | Factories create or update one task per node |
| 12 | Task Binding | Tasks receive storage access and logging services |

Profiling and Graphviz output observe this process but do not modify its semantic result.

## Formal model

Let the graph specification be a finite, directed, port-labeled multigraph

$$
G_{\mathrm{spec}}=(V_{\mathrm{spec}},E_{\mathrm{spec}}).
$$

An edge is written

$$
e=(u,o,v,i),
$$

where $u$ is the producer, $o$ is its output slot, $v$ is the consumer, and $i$ is its input slot.

| Notation | Definition |
| --- | --- |
| $\operatorname{src}(e)$ | Producer node $u$ |
| $\operatorname{out}(e)$ | Producer output slot $o$ |
| $\operatorname{dst}(e)$ | Consumer node $v$ |
| $\operatorname{in}(e)$ | Consumer input slot $i$ |
| $n(v)\in\mathrm{String}$ | Instance name of node $v$ |
| $q(v)\in\mathrm{RegistryKey}$ | Registry key used to look up the node's factory |
| $\kappa(v)\in\{\mathrm{Sync},\mathrm{Async}\}$ | Execution kind established during type inference |

Let $\Gamma$ denote the registry. The registry key $q(v)$ and execution kind $\kappa(v)$ are
distinct: for example, `Fft` may be a registry key whose factory infers a synchronous task.

### Compiler state

Compilation progressively establishes the following objects and mappings:

| Symbol | Runtime representation | Meaning |
| --- | --- | --- |
| $G=(V,E)$ | `GraphPlan` | Mutable graph enriched by later passes |
| $F_v$ | node `InferResult` | Inference result for node $v$ |
| $D_E$ | edge descriptors | Tensor descriptor carried by an edge |
| $D_T$ | `tensor_descs` | Tensor descriptor associated with a TID |
| $\tau$ | TIDs | Logical tensor identity |
| $\sigma$ | `tid_to_sid` | Logical-to-physical storage mapping |
| $\operatorname{Own}$ | owned input/output flags | Storage lifetime authority |
| $\mathcal S$ | `sections` | Independently scheduled regions |
| $c$ | `streams` | CUDA stream assigned to each section |
| $T_v$ | `tasks` | Materialized runtime task for node $v$ |
| $A_v$ | `node_storage_adapters` | Storage adapter for node $v$ |

Let $\mathcal R$ denote the resource bundle stored in `CompilerOutput::resources`, comprising tensor
descriptors, the TID-to-SID mapping, storage objects and memory blocks, CUDA streams, task instances,
and storage adapters.

When compilation succeeds, we write

$$
O=
\operatorname{Compile}_{\Gamma}(G_{\mathrm{spec}},O_{\mathrm{prev}}?),
$$

where $O_{\mathrm{prev}}$ is an optional previous compilation result and

$$
O=(G,\mathcal S,\mathcal R)
$$

is the resulting `CompilerOutput`.

`Compile` is partial: invalid specifications or factory results may be rejected, and allocation or
CUDA operations may fail.

### Compilation as pass composition

Let $X_0$ contain the compilation inputs and let $X_j$ be the compiler state after pass $P_j$.
Each pass is a partial transformation

$$
P_j:X_{j-1}\rightharpoonup X_j.
$$

For a fixed registry $\Gamma$, the complete compiler is

$$
\operatorname{Compile}_{\Gamma}
=
P_{12}\circ P_{11}\circ\cdots\circ P_1.
$$

## Structure and inference

### Structural validity

Validation establishes the predicate

$$
P_{\mathrm{struct}}(G_{\mathrm{spec}},\Gamma),
$$

defined by three constraints.

Node names are unique:

$$
\forall u,v\in V_{\mathrm{spec}},
\quad
u\ne v\Rightarrow n(u)\ne n(v).
$$

Every declared registry key is registered:

$$
\forall v\in V_{\mathrm{spec}},
\quad
q(v)\in\operatorname{dom}(\Gamma).
$$

Every connected input slot has at most one producer:

$$
\forall e_1,e_2\in E_{\mathrm{spec}},
\quad
e_1\ne e_2
\Rightarrow
(\operatorname{dst}(e_1),\operatorname{in}(e_1))
\ne
(\operatorname{dst}(e_2),\operatorname{in}(e_2)).
$$

### Graph construction

The build pass constructs $G=(V,E)$ and bijections

$$
\phi_V:V_{\mathrm{spec}}\rightarrow V,
\qquad
\phi_E:E_{\mathrm{spec}}\rightarrow E
$$

that preserve node specifications, edge specifications, endpoints, and port indices.

Therefore

$$
G\cong G_{\mathrm{spec}},
$$

where $\cong$ denotes this port- and specification-preserving graph isomorphism.

Only the representation changes. Inference results and runtime resources have not yet been
established.

### Type inference

The compiler computes a topological order

$$
\pi=(v_1,\ldots,v_{|V|})
$$

such that

$$
(u,o,v,i)\in E
\Rightarrow
\operatorname{pos}_{\pi}(u)<\operatorname{pos}_{\pi}(v).
$$

Failure to construct $\pi$ rejects cyclic graphs. Nodes are then visited producer-first.

For each node $v$, incoming edge descriptors form an indexed vector $I_v$:

$$
I_v[\operatorname{in}(e)]=D_E(e)
\qquad
\forall e\in E:\operatorname{dst}(e)=v.
$$

The registered factory computes

$$
F_v
=
\Gamma[q(v)].\operatorname{infer}(I_v,\operatorname{settings}(v)).
$$

Its output descriptors propagate to outgoing edges:

$$
D_E(e)
=
F_{\operatorname{src}(e)}.\operatorname{output\_descs}[\operatorname{out}(e)]
\qquad
\forall e\in E.
$$

After inference, every node has an `InferResult`, every edge carries the descriptor of its selected
producer output, and $G$ is known to be a DAG. Factory validation failures and out-of-range graph
ports reject compilation.

!!! warning "Connected input indices must be dense"
    The implementation sizes $I_v$ from the node's incoming edge count. Connected input indices must
    consequently form $0,\ldots,\deg^-(v)-1$. Sparse connected slots are not represented as explicit
    missing inputs.

!!! warning "Factory result consistency is trusted"
    The intended factory contract requires `input_descs`, `owned_inputs`, and the later `in_tids`
    vector to describe the same input slots; likewise for outputs. It also requires valid in-place
    indices and agreement between the registered factory interface and $\kappa(v)$. The compiler
    does not currently validate all of these relationships immediately after `infer(...)`. A
    malformed factory result may therefore fail in a later pass rather than at the contract
    boundary.

## Logical tensors and physical storage

### Tensor identity

Define the set of inferred output slots

$$
\mathcal O
=
\{(v,o)\mid v\in V,\ 0\le o<|F_v.\operatorname{output\_descs}|\}.
$$

Let $\mathcal T$ be the set of generated tensor IDs. The tensor-ID pass constructs a bijection

$$
\tau_{\mathrm{out}}:\mathcal O\rightarrow\mathcal T.
$$

Every output receives a TID, including an unconnected output. Each edge inherits the identity of the
output it carries:

$$
\tau(e)
=
\tau_{\mathrm{out}}(\operatorname{src}(e),\operatorname{out}(e)).
$$

The consumer observes that same identity:

$$
\tau_{\mathrm{in}}(\operatorname{dst}(e),\operatorname{in}(e))
=
\tau(e).
$$

Thus fan-out duplicates references, not logical tensors:

$$
e_1,e_2\text{ consume }(v,o)
\Rightarrow
\tau(e_1)=\tau(e_2)=\tau_{\mathrm{out}}(v,o).
$$

The TID descriptor map is total over generated TIDs:

$$
D_T(\tau_{\mathrm{out}}(v,o))
=
F_v.\operatorname{output\_descs}[o].
$$

### Storage identity and in-place aliasing

TIDs name logical values; SIDs name physical storage identities.

Let $\mathcal U$ be the set of generated SIDs. The storage-mapping pass constructs a surjection

$$
\sigma:\mathcal T\rightarrow\mathcal U.
$$

An ordinary output receives a fresh SID. If node $v$ declares output $o$ in place with input $i$,
then

$$
\sigma(\tau_{\mathrm{out}}(v,o))
=
\sigma(\tau_{\mathrm{in}}(v,i)).
$$

Different TIDs can therefore describe different logical tensors backed by the same storage.

!!! warning "In-place safety is a factory obligation"
    Every tensor sharing an SID must agree on memory location and fit within the selected allocation.
    Overwriting an in-place input must also be safe for every other consumer. The compiler currently
    trusts the factory on these points: it neither compares all aliased descriptors nor performs
    liveness analysis. See [Holoflow Task Model](../concepts/task-model.md#in-place-mappings).

### Unique ownership

For $d\in\{\mathrm{in},\mathrm{out}\}$, let $\tau_d(v,j)$ denote the TID of slot $j$ on side $d$,
and let $\operatorname{owned}_d(v,j)$ denote the corresponding ownership flag from $F_v$.

For each SID $s\in\mathcal U$, define its declared ownership set

$$
\operatorname{Owners}(s)
=
\{(v,d,j)\mid
d\in\{\mathrm{in},\mathrm{out}\},
\operatorname{owned}_d(v,j),
\sigma(\tau_d(v,j))=s\}.
$$

Buffer consistency validates

$$
|\operatorname{Owners}(s)|\le 1
\qquad
\forall s\in\mathcal U.
$$

This makes storage lifetime authority unambiguous. Multiple owned slots that resolve to one SID are
rejected, including conflicts introduced through in-place aliasing. Runtime ownership behavior is
defined in [Storage Ownership](../concepts/storage-ownership.md).

### Storage materialization

For every SID $s\in\mathcal U$, the compiler selects a representative TID

$$
r(s)\in\sigma^{-1}(\{s\})
$$

and creates one stable `Storage` object with

$$
\operatorname{Storage}[s].\operatorname{mem\_loc}
=
D_T(r(s)).\operatorname{mem\_loc}
$$

and

$$
\operatorname{Storage}[s].\operatorname{bytes}
=
D_T(r(s)).\operatorname{num\_bytes}().
$$

Let

$$
\mathcal U_T
=
\{s\in\mathcal U\mid |\operatorname{Owners}(s)|=1\}
$$

be the task-owned SIDs.

For $s\notin\mathcal U_T$, the compiler allocates a `MemoryBlock` or moves an exact match from the
previous output:

$$
(\operatorname{mem\_loc},\operatorname{bytes})_{\mathrm{old}}
=
(\operatorname{mem\_loc},\operatorname{bytes})_{\mathrm{new}}.
$$

It then establishes

$$
\operatorname{Storage}[s].ptr
=
\operatorname{MemoryBlock}[s].data.
$$

For $s\in\mathcal U_T$, no backing block is allocated and

$$
\operatorname{Storage}[s].ptr=\mathrm{null}
$$

until the owning task publishes a pointer during execution. Previous blocks that are not reused are
released when the reuse pool is destroyed.

Finally, every node receives an adapter $A_v$ that resolves slots through the chain

$$
\mathrm{slot}\xrightarrow{\tau}\mathrm{TID}
\xrightarrow{\sigma}\mathrm{SID}
\longrightarrow\mathrm{Storage}.
$$

For example,

$$
A_v.\operatorname{input}(i)
=
\operatorname{Storage}[\sigma(\tau_{\mathrm{in}}(v,i))].
$$

The adapters exist at this stage but are bound to tasks only after task instantiation.

## From a DAG to execution sections

An asynchronous task splits input acceptance from output production. It is a scheduling boundary,
not a member of a section's synchronous sequence.

Let

$$
V_S=\{v\in V\mid\kappa(v)=\mathrm{Sync}\},
\qquad
V_A=\{v\in V\mid\kappa(v)=\mathrm{Async}\}.
$$

For an asynchronous node $a\in V_A$, define its synchronous predecessor and successor sets:

$$
\operatorname{Pred}_S(a)
=
\{p\in V_S\mid
\exists o,i:\;(p,o,a,i)\in E\},
$$

$$
\operatorname{Succ}_S(a)
=
\{q\in V_S\mid
\exists o,i:\;(a,o,q,i)\in E\}.
$$

### The section equivalence relation

The compiler constructs the smallest equivalence relation $\sim$ over $V_S$ satisfying three
generating rules.

First, directly connected synchronous nodes share a section:

$$
(u,o,v,i)\in E\land u,v\in V_S
\Rightarrow
u\sim v.
$$

Second, all synchronous predecessors of one asynchronous task share a section:

$$
p,q\in\operatorname{Pred}_S(a)
\Rightarrow
p\sim q
\qquad
\forall a\in V_A.
$$

Third, all synchronous successors of one asynchronous task share a section:

$$
p,q\in\operatorname{Succ}_S(a)
\Rightarrow
p\sim q
\qquad
\forall a\in V_A.
$$

The execution sections are the equivalence classes

$$
\mathcal S=V_S/{\sim}.
$$

Consequently, they partition the synchronous nodes:

$$
\bigsqcup_{S\in\mathcal S}S=V_S.
$$

This quotient induces the section map

$$
\operatorname{sec}:V_S\rightarrow\mathcal S,
\qquad
\operatorname{sec}(v)=[v]_{\sim}.
$$

The equations above define section membership. The current implementation computes the equivalence
classes with a disjoint-set structure and stores each class in producer-to-consumer topological
order.

### Attaching asynchronous boundaries

When an asynchronous node $a$ has synchronous predecessors, define

$$
\operatorname{up}(a)=\operatorname{sec}(p),
\qquad
p\in\operatorname{Pred}_S(a).
$$

This is well-defined because all synchronous predecessors of $a$ are equivalent. The task is added
to

$$
\operatorname{async\_prod}[\operatorname{up}(a)].
$$

Similarly, when $a$ has synchronous successors,

$$
\operatorname{down}(a)=\operatorname{sec}(q),
\qquad
q\in\operatorname{Succ}_S(a),
$$

and the task is added to

$$
\operatorname{async\_cons}[\operatorname{down}(a)].
$$

Either attachment may be absent at a graph boundary.

Within each section, asynchronous producers whose inference contract promises producer-stream
synchronization are stably ordered before ordinary producers. This allows their barrier to cover
preceding synchronous work before an ordinary queue publishes GPU-backed input.

For example:

```text
Source -> Upload -> Queue -> Compute -> Download -> Sink
        [ section 0 ]       [          section 1          ]
                       ^   ^
                       |   +-- Queue consumes on section 1
                       +------ Queue produces from section 0
```

!!! warning "Async-to-Async edges are unsupported"
    The current section model rejects every edge $e$ for which both
    $\operatorname{src}(e)\in V_A$ and $\operatorname{dst}(e)\in V_A$.

### Stream assignment

The compiler assigns one owned CUDA stream to every section:

$$
c:\mathcal S\rightarrow\mathrm{CUDAStream}.
$$

Fresh compilation creates a new stream for every section. Recompilation moves available streams
from the previous output into new sections and creates additional streams when necessary. Each
section stores the raw handle $c(S)$, while `resources.streams[section.id]` owns the corresponding
`CudaStream`.

!!! warning "Stream reuse is positional"
    Previous streams are matched to new sections by iteration order, not by section identity or
    graph structure. Reused tasks must obtain their stream handles from the new creation context.

## Task materialization

Every synchronous node $v\in V_S$ receives its section stream:

$$
\operatorname{ctx}(v).\operatorname{stream}
=
c(\operatorname{sec}(v)).
$$

An asynchronous node receives the streams on its two defined sides:

$$
\operatorname{ctx}(a).\operatorname{producer\_stream}
=
\begin{cases}
c(\operatorname{up}(a)), & \operatorname{Pred}_S(a)\ne\varnothing,\\
\mathrm{null}, & \text{otherwise},
\end{cases}
$$

$$
\operatorname{ctx}(a).\operatorname{consumer\_stream}
=
\begin{cases}
c(\operatorname{down}(a)), & \operatorname{Succ}_S(a)\ne\varnothing,\\
\mathrm{null}, & \text{otherwise}.
\end{cases}
$$

The factory creates a new task unless the previous output contains a reusable task with:

$$
\text{the same node name},
\qquad
\text{the same registry key},
\qquad
\text{a compatible Sync/Async interface}.
$$

When all three conditions hold, the previous task is moved into `factory.update(...)`. The factory
decides which internal state survives.

After `create(...)` or `update(...)`, the compiler synchronizes every non-null stream in the creation
context so initialization has completed before compilation returns.

The final binding pass associates each task $T_v$ with its storage adapter and a logger identified by
the registry key and node name:

$$
T_v.\operatorname{storage\_access}\leftarrow A_v.
$$

Constructors and factory update methods must not use these services because binding happens
afterward. See [Holoflow Task Model](../concepts/task-model.md#the-common-task-interface).

## Recompilation semantics

Supplying $O_{\mathrm{prev}}$ transfers ownership of reusable resources to the compiler. The new
specification still passes through every compiler pass.

Therefore

$$
\operatorname{Compile}_{\Gamma}(G_{\mathrm{spec}},O_{\mathrm{prev}})
$$

must satisfy the same compiler postconditions as fresh compilation. Reuse changes resource identity
and construction cost, but not the graph derived from the new specification or the invariants
established by the compiler passes.

The implementation may reuse:

| Resource | Reuse key |
| --- | --- |
| Memory block | Exact memory location and byte size |
| CUDA stream | Position in the previous stream map |
| Task | Node name, registry key, and compatible task interface |

Resources that cannot be reused are destroyed normally. Because reuse moves objects out of the
previous result, callers must not retain scheduler or task references into that result while
recompiling.

## Compiler guarantees and trusted assumptions

### Compiler-enforced postconditions

If compilation succeeds, the output satisfies:

$$
\begin{aligned}
&G\cong G_{\mathrm{spec}}\text{ and }G\text{ is a DAG},\\
&\tau_{\mathrm{out}}:\mathcal O\rightarrow\mathcal T
  \text{ identifies every logical output},\\
&\sigma:\mathcal T\rightarrow\mathcal U
  \text{ assigns every TID one SID},\\
&|\operatorname{Owners}(s)|\le1
  \quad\forall s\in\mathcal U,\\
&\bigsqcup_{S\in\mathcal S}S=V_S,\\
&c(S)\text{ is defined for every }S\in\mathcal S,\\
&T_v\text{ and }A_v\text{ exist for every }v\in V.
\end{aligned}
$$

In addition:

- every SID has a stable `Storage` object;
- compiler-owned SIDs have correctly located and sized backing blocks;
- declared in-place mappings share an SID;
- every section's synchronous sequence is topologically ordered;
- asynchronous tasks are attached to each defined producer and consumer side;
- every task has its runtime storage and logging services bound; and
- initialization submitted to compiler-provided streams during task creation or update has completed.

The scheduler can therefore concentrate on execution: constructing tensor views, running section
threads, enforcing asynchronous boundaries, and applying the storage-ownership protocol.

### Required factory invariants

Successful compilation does not independently verify every property required from task factories.
The factory contract additionally requires that:

- `input_descs`, ownership flags, and input-slot metadata describe compatible slot sets;
- the corresponding output-side vectors are mutually consistent;
- declared in-place indices are valid;
- every tensor sharing an SID is storage-compatible with the selected allocation;
- in-place mutation is safe with respect to every other consumer of the aliased storage; and
- the concrete factory interface agrees with the inferred execution kind $\kappa(v)$.

Violating these obligations may cause a later compiler pass or runtime operation to fail. They form
part of the trusted task/factory boundary rather than compiler-enforced postconditions.

## Failure and diagnostics

Any partial transformation may fail. The compiler logs the exception, optionally writes
`compilation_failure.dot`, synchronizes the CUDA device, clears the last CUDA error, flushes its log,
and rethrows. With graph dumping enabled, success produces `compilation_success.dot`.

When profiling is enabled, pass timings and selected detailed operations may be emitted as Chrome
trace events. These diagnostics observe compiler state but establish no scheduler invariants.

## Implementation map

The pass driver and implementations are in
`src/holoflow/src/runtime/compiler.cc`. Public configuration and output types are declared in
`src/holoflow/include/holoflow/runtime/compiler.hh`; `GraphPlan`, `ExecResouces`, and `Section` are
declared in `src/holoflow/include/holoflow/runtime/graph_exec.hh`.

The most direct behavioral tests are `test/holoflow/compiler_test.cc`,
`test/holoflow/compiler_additional_test.cc`, and the compiler-related cases in
`test/holoflow/scheduler_functional_test.cc`.
