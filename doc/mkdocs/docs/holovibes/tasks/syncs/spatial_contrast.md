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

Choose **PCA** under **Time Transform**, then choose **Local PCA** under **PCA type**.
**Global PCA** also appears there when a propagation Space Transform is selected.
Local PCA-LSCI is retained for older settings but hidden from the selector.

**Holovibe Local-PCA** is a separate Time Transform based on the Holovibes
`local_pca` branch. It applies the selected spatial transform to each temporal
frame first, then gathers wrapped complex kernels (32×32 by default, step 16).
For each kernel it diagonalizes the uncentered Gram matrix $H^*H$, computes
$e=\exp(-\sum_i p_i\log p_i)$ with $p_i=\max(\lambda_i,0)/\sum_j\max(\lambda_j,0)$,
and removes the strongest $\min(T,\lceil f e\rceil)$ modes, where $f$ is the
**Rank factor** (default 1). The residual $H V_{\rm keep}V_{\rm keep}^*$ is
accumulated at every kernel pixel. Overlapping residuals are averaged by their
actual contribution counts. The rectangular checkbox exposes independent kernel
widths, heights, and steps; each step must be positive and no larger than its
kernel side. Time blocks are disjoint. Z and Width select residual time samples
for display rather than PCA components. With no Space Transform, real input is
treated as complex input with zero imaginary part.

**Local PCA** computes a temporal PCA of the uncentered intensity stack in each spatial kernel.
Like Global PCA, it outputs the selected component projections, with **Z** and **Width** selecting
an ascending-energy component interval `[Z, Z + Width)`. The projections have one image plane per
selected component. Fresnel Diffraction or Angular Spectrum is applied to the component images
after PCA, as in Global PCA; with None, it displays the projections directly. Local eigenvector signs can vary
between neighboring kernels. A full-frame kernel and step use one basis for the entire frame;
component magnitudes then agree with Global PCA when both receive the same input.

Local PCA supports 2–64 frames.

For Local PCA, the square kernel can be odd-sized up to the smaller frame dimension
and moves one pixel at a time. Enable **Rectangular kernel** to set width and height separately,
up to the corresponding frame dimensions, and set a horizontal and vertical step. Each step
must fit its kernel dimension and match its parity. Kernels wrap at image borders and write
only their central step-width × step-height output tile; edge tiles are clipped. Neighboring
kernels can overlap, but output tiles do not.

**Local PCA-LSCI** uses a sliding PCA with each frame's kernel mean subtracted to reconstruct a static time series at
each kernel center. Automatic selection reconstructs modes at or above the local mean eigenvalue;
manual selection reconstructs the complement of the Z/Width interval. The **Static (Local PCA then
LSCI 3D)** map is 3D LSCI on the locally reconstructed intensities. Select **LSCI (3D)**
to view the input contrast by itself. The
PCA kernel wraps at the image border; the subsequent
LSCI spatial window follows the existing clipped-border rule. The task stores temporal moments
for each reconstructed pixel rather than a second full frame stack. Local PCA-LSCI uses stride 1.
Processed preview and recording convert the selected F32 map to U8 using the configured
**Display max K** range.

Local PCA-LSCI uses the acquired intensity frames directly, without hologram propagation or temporal Fourier analysis. Local PCA can propagate its component images after PCA. **Accumulation** can average successive maps. If enabled, **Flatfield** is applied to the calculated map before accumulation, and percentile clipping is applied after accumulation. These operations change the processed result; disable them when reproducing an uncorrected $K$ map.

The calculated map is F32 before display conversion. With fixed scaling, **Display max K** maps the range from 0 to the selected value onto U8 grayscale and clips larger values for display. **Invert LSCI** reverses that grayscale. Holovibes uses this same U8 image for the processed preview and processed recording; neither control changes the estimator's F32 output. Raw viewing and raw recording use the original intensity stream.

For a worked configuration based on high-speed retinal imaging, see [Retinal laser speckle contrast imaging](../../../applications/retinal-lsci.md).

## Settings

--8<-- "docs/schemas/syncs/spatial_contrast_settings.md"
