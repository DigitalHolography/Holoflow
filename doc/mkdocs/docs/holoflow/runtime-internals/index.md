# Runtime Internals

!!! note "Documentation in progress"
    This section will document compiler passes, execution-section partitioning, scheduler behavior, CPU and CUDA concurrency, storage ownership, events, and graph updates for contributors working on the Holoflow runtime.

The [Holoflow Task Model](../concepts/task-model.md) introduces the public contracts on which these internals operate.

## Contents

- [Graph Compilation](graph-compilation.md) explains how a `GraphSpec` becomes an executable
  graph, including the contract established by each compiler pass.
