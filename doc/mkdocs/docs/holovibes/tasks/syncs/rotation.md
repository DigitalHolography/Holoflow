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

## Rotation Behavior

Rotation is executed by a CUDA kernel that computes the remapped output coordinates for each pixel. For example, a 90° rotation is implemented by mapping:

```
newX = sizeX - 1 - y
newY = x
```

Each output element is directly computed from the input via index transformation, ensuring efficient and parallel GPU execution.

Unsupported rotations (non‑multiples of 90°) or rotations equivalent to 0°/360° are rejected during inference.

## Settings

The rotation settings must be provided as JSON, with the following structure:

```json
{
  "angle": 90,              // Must be 90, 180, or 270
  "axis": "Z"              // Only "Z" is valid for 2D images
}
```

### Fields

* **angle**: The rotation angle in degrees. Must satisfy:

  * `angle % 90 == 0`
  * `angle % 360 != 0`

* **axis**: Rotation axis. Options:

  * `"X"`
  * `"Y"`
  * `"Z"`

  For 2D tensors, only `"Z"` is allowed.

The settings structure is validated in `RotationFactory::infer` before kernel execution.
