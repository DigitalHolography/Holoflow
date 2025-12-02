# Rotation Sync Task

The **Rotation** task rotates 2D tensors (images) by multiples of 90 degrees on the GPU. It operates on tensors stored in device memory and uses a CUDA kernel to efficiently remap pixel coordinates.

Rotation is performed around the selected axis, but only rotations around the **Z axis** are currently valid for 2D images. Supported angles are:

* **90°**
* **180°**
* **270°**

Other angles are rejected at inference time.

## Inputs

This task accepts a single input tensor of shape `(B, H, W)` or `(H, W)`, where:

* `B`: batch size (optional)
* `H`: image height
* `W`: image width

Requirements:

* The tensor must reside in **device (GPU) memory**.
* The tensor's dtype must match the expected internal type (`float32` or complex types depending on the pipeline).

## Outputs

The task produces a single output tensor. For 90° or 270° rotations, spatial dimensions are swapped:

* Input shape: `(H, W)` → Output shape: `(W, H)`

Batch dimension (if present) is preserved.

The output tensor:

* Uses the same dtype as the input.
* Is stored in **device memory**.

## Inplace

This task **does not** support inplace execution.

## Ownership

The Rotation task does not own its input or output tensors.

## Settings
--8<-- "docs/schemas\syncs/rotation_settings.md"