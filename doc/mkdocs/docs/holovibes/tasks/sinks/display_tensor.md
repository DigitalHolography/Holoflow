# Display Tensor Sink Task
This task displays a tensor as an image in a window. The window to display the image to is passed to the factory.

## Inputs
This task has a single input of shape `(H, W)` or `(1, H, W)`, where:

- `H` and `W` are the height and width of the image to display.

The dtype of the input tensor must be unsigned 8-bit integer (`uint8`).

The memory location of the input tensor can be either host memory or device memory.

## Outputs
This task does not have any outputs.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\sinks\display_tensor_settings.md"