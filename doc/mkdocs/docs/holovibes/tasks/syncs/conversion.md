# Conversion Sync Task
The **Conversion** sync task changes the dtype of a tensor on the GPU. The configured **target** dtype and **strategy** determine how samples are interpreted or rescaled before being written to the output tensor.

Supported combinations are:

- `uint8` -> `complex32` with the `Real` strategy (real part copied, imaginary part set to zero)
- `uint16` -> `complex32` with the `Real` strategy
- `float32` -> `uint8` with the `Scaled` strategy
- `float32` -> `uint16` with the `Scaled` strategy
- `complex32` -> `float32` with the `Modulus` strategy
- `complex32` -> `float32` with the `Argument` strategy

The `Scaled` strategy computes the minimum and maximum value of the input tensor, then linearly maps the range to fill the target integer interval by rescaling the (min, max) interval to the min and max of the target interval. The `Modulus` and `Argument` strategies operate element-wise on complex tensors, returning magnitude or phase respectively.

!!! warning
    When using the `Scaled` strategy, constant inputs (where `min == max`) lead to an undefined scale factor. Make sure the source tensor spans a non-zero range.

## Inputs
This task consumes a single input tensor of any shape.

The dtype of the input tensor must be one of:

- `uint8`
- `uint16`
- `float32`
- `complex32`

The memory location of the input tensor must be device memory.

## Outputs
This task produces one output tensor with the same shape as the input.

The dtype of the output tensor is determined by the configured **target** dtype.

The memory location of the output tensor is device memory.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\conversion_settings.md"
