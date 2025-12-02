# Percentile Clip Sync Task
The **Percentile Clip** sync task clamps a 3D float tensor on the GPU to the value range defined by two percentiles estimated inside an elliptical region of interest (ROI). Every element of the input tensor is clipped to the closed interval `[p_min, p_max]`, where `p_min` and `p_max` are the selected percentile values`.

!!! note
    The ROI is an ellipse expressed in normalized image coordinates: `cx`/`cy` give the center in `[0, 1]`, `rx`/`ry` the radii in `[0, 1]`, and `angle` the rotation in degrees. 

!!! warning
    If the ROI does not cover any pixels, execution fails with `No pixels in ROI`. Ensure your ROI parameters select at least one sample.

## Inputs
This task expects a single input tensor of shape `(B, H, W)`:

- `B`: number of frames 
- `H`: height 
- `W`: width 

The dtype must be `float32` and the tensor must reside in device memory.

## Outputs
This task emits one output tensor with the same shape `(B, H, W)`, dtype `float32`, and device memory location as the input. Values are the clipped version of the input tensor.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\syncs\pct_clip_settings.md"
