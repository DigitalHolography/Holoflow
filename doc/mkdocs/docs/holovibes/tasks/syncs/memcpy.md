# Memcpy Sync Task
The **Memcpy** sync task performs a raw byte-for-byte copy of its single input tensor into the memory space selected by the `target` setting. It preserves the tensor's shape and dtype and does not apply any numerical transformation.

Depending on the source and target locations, the task chooses an appropriate transfer path:

- Host -> Host copies use a small OpenMP-based worker pool to split the buffer and accelerate CPU transfers.
- Host <-> Device and Device -> Device copies run through `cudaMemcpyAsync` on the task's CUDA stream, allowing the copy to overlap with other GPU work scheduled on the same stream.

## Inputs
This task has a single input tensor of arbitrary shape, dtype, and memory location.

## Outputs
This task has a single output tensor whose shape and dtype exactly match the input tensor. The output memory location is determined by the `target` setting:

- `Host`: the tensor is produced in host memory.
- `Device`: the tensor is produced in device memory.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\memcpy_settings.md"
