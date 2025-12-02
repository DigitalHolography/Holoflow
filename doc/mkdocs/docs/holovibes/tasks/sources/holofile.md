# Holofile Source Task

This task reads data from a `.holo` file and outputs it as a tensor. When the end of the file is reached, it loops back to the beginning. The task can be configured to load data in different ways, such as loading all frames into memory at once or loading frames on-the-fly. The file specified must exist and be a valid `.holo` file during the task's validation phase and execution.

## Inputs
This task does not have any inputs.

## Outputs
This task has a single output of shape `(B, H, W)`, where:

- `B` is the batch size specified in the settings.
- `H` and `W` are the height and width of the frames in the `.holo` file.

The dtype of the output tensor is unsigned `N`-bit integer, where `N` is the bits per pixel specified in the file header.
If the file bits per pixel is not 8, 16, or 32, the output tensor will have dtype `uint8`.

The memory location of the output tensor is determined by the `load_kind` setting. If `load_kind` is set to `Live` or `CPUCached`, the output tensor will be in host memory. If `load_kind` is set to `GPUCached`, the output tensor will be in device memory.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\sources\holofile_settings.md"