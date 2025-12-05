# Registration Sync Task
The **Registration** sync task computes a translational alignment between frames using frequency-domain cross-correlation and applies the computed subpixel shift to reproject the input frame. The task uses a masked, mean-centered cross-correlation pipeline implemented with FFTs:

1. center the mean of the input (optionally restricted by an ellipse mask derived from `radius`),  
2. compute forward FFTs (R2C) of the input and stored reference,  
3. form the normalized cross-power spectrum (conjugate × product, normalize magnitude),  
4. inverse FFT to obtain the cross-correlation surface,  
5. find the integer peak and refine it with a 3×3 subpixel estimator,  
6. apply the computed subpixel shifts to the input using bilinear interpolation (wrap-around).

Typical uses include translational registration of single-frame images, subpixel alignment prior to averaging, and motion correction pipelines.

!!! warning
    The `radius` setting must be within `[0, 1]`. Values at the extremes may produce degenerate masks (e.g. `radius == 0` selects nothing, `radius == 1` selects the full frame). Choose a radius appropriate to the region of interest for robust cross-correlation.

## Inputs
This task consumes a single input tensor.

The input requirements (as enforced by `RegistrationFactory::infer` and the implementation) are:

- The tensor **must** have at least 2 dimensions (height and width).
- The dtype **must** be `float32` (F32).
- The tensor **must** reside in device (GPU) memory.
- The task expects a single frame (batch size 1) for proper operation; non-unit batches are not processed by the algorithm.

## Outputs
This task produces one output tensor with the same shape and dtype as the input (the pipeline computes shifts and writes the shifted frame).

The memory location of the output tensor is device memory.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\syncs\registration_settings.md"
