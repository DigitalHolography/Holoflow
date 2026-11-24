# Spatial Contrast Sync Task

The **SpatialContrast** sync task calculates laser speckle contrast from intensity samples. For each output pixel, it divides the population standard deviation by the mean over a spatial neighborhood, a temporal window, or both:

$$
K = \frac{\sigma_I}{\mu_I}, \qquad
\sigma_I = \sqrt{\max\!\left(\frac{1}{N}\sum_{i=1}^{N} I_i^2 - \mu_I^2,\,0\right)}.
$$

The kernel and FFT methods calculate the same box-window statistic. Spatial neighborhoods are clipped to the image at borders, so their sample count is smaller there. A non-positive local mean produces zero contrast. The input tensor is not modified.

## Modes and tensor shapes

| Mode | `window_size` | `temporal_window_size` | Input | Output |
| --- | ---: | ---: | --- | --- |
| Spatial (2D) | Odd, 3–255 | 1 | `(..., H, W)` | Same shape |
| Temporal only | 1 | At least 3 | `(T, H, W)`, with `T = temporal_window_size` | `(1, H, W)` |
| Pooled spatial and temporal (3D) | Odd, 3–255 | At least 3 | `(T, H, W)`, with `T = temporal_window_size` | `(1, H, W)` |

The task accepts one GPU-resident, C-contiguous tensor of dtype U8, U16, or F32 and returns F32 values. In 2D mode, leading dimensions are independent frames. In temporal modes, the first dimension is one complete frame window, and the task emits one contrast map for that window. A temporal size of 2 is not accepted.

The direct `kernel` method uses CUDA kernels for local sums. The `fft` method uses box convolution for spatial windows; temporal-only contrast uses the direct kernel even if `fft` is selected. The FFT method does not support CUDA graph capture for spatial windows.

## Holovibes processing

For **LSCI (2D)**, choose it under **Space Transform**. Holovibes calculates a contrast map for every input frame; **Time Window** and **Time Stride** do not set its estimator support.

For **LSCI (3D)**, choose it under **Time Transform**. **LSCI Window (px)** sets the spatial side length, **Time Window** sets the number of consecutive frames pooled into each map, and **Time Stride** sets how many frames to advance. Set stride equal to window length for nonoverlapping blocks; a smaller stride creates overlapping windows. The two LSCI choices are mutually exclusive. **LSCI method** selects Kernel or FFT.

Both modes use the acquired intensity frames directly, without hologram propagation or temporal Fourier analysis. **Accumulation** can average successive contrast maps. If enabled, **Flatfield** is applied to the calculated map before accumulation, and percentile clipping is applied after accumulation. These operations change the processed result; disable them when reproducing an uncorrected $K$ map.

The calculated map is F32 before display conversion. **Display max K** maps the range from 0 to the selected value onto U8 grayscale and clips larger values for display. **Invert LSCI** reverses that grayscale. Holovibes uses this same U8 image for the processed preview and processed recording; neither control changes the estimator's F32 output. Raw viewing and raw recording use the original intensity stream.

For a worked configuration based on high-speed retinal imaging, see [Retinal laser speckle contrast imaging](../../../applications/retinal-lsci.md).

## Settings

--8<-- "docs/schemas/syncs/spatial_contrast_settings.md"
