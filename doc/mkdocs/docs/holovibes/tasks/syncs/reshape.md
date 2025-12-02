# Reshape Sync Task
The **Reshape** sync task changes the logical shape of a tensor on the GPU while preserving element order and dtype. It writes the input buffer into a new tensor whose extents are provided by the `shape` setting, effectively reinterpreting the linear data with different dimensions.

!!! note
    Reshape only remaps contiguous data; it neither transposes nor reorders samples. The linear memory layout remains identical to the source tensor.

!!! warning
    The configured `shape` must contain only positive dimensions and its product must exactly match the number of elements in the input tensor. If either check fails, task creation is rejected during inference.

## Inputs
This task consumes a single input tensor of arbitrary rank, shape, dtype, and memory location.

## Outputs
This task produces one output tensor located in the same memory space as the input tensor. The output tensor has the same dtype as the input tensor but a shape defined by the configured `shape` setting.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\syncs\reshape_settings.md"
