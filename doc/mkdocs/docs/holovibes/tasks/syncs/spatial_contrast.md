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

Choose **PCA** under **Time Transform**, then choose **Local PCA**, **Advanced Local PCA**, or
**Local PCA-LSCI** under **PCA type**. **Global PCA** also appears there when
a propagation Space Transform is selected. PCA-LSCI is currently hidden in the UI. It computes a single temporal PCA over the entire
frame block after subtracting each frame's spatial mean. Eigenvectors whose nonnegative
eigenvalues are at least the average eigenvalue form a static reconstruction. The **PCA-LSCI map**
selector displays **Static (PCA then LSCI 3D)**, **Dynamic** (input LSCI 3D minus Static, the
default), or **PCA frame (center)** (the reconstructed center frame before LSCI 3D). Select
**LSCI (3D)** to view the input contrast alone. Dynamic is a signed
difference of contrast maps, not contrast of the PCA residual. PCA-LSCI accepts 2–512 frames
and an odd 1–63 pixel spatial window. It uses the Kernel LSCI border rule, clipping windows at
image edges. The PCA projection is tiled, and no second full reconstructed frame stack is stored.
The PCA frame preview scales its own intensity range to the display; the contrast maps use the
configured LSCI contrast range.

**Local PCA** computes a temporal PCA of the uncentered intensity stack in each spatial kernel.
Like Global PCA, it outputs the selected component projections, with **Z** and **Width** selecting
an ascending-energy component interval `[Z, Z + Width)`. The projections have one image plane per
selected component. Fresnel Diffraction or Angular Spectrum is applied to the component images
after PCA, as in Global PCA; with None, it displays the projections directly. Local eigenvector signs can vary
between neighboring kernels. A full-frame kernel and step use one basis for the entire frame;
component magnitudes then agree with Global PCA when both receive the same input.

**Advanced Local PCA** uses an uncentered local decomposition of the input intensity frames.
When a Space Transform is selected, it is applied to each frame before this decomposition.
**Modes to remove** selects
`r = 0`, `1`, or `2` highest-energy modes (default `1`). It subtracts those modes from each
pixel's original time series and outputs its temporal residual variance divided by the square
of that pixel's original temporal mean intensity. For `r = 0`, this is the original temporal
variance divided by the original mean squared. Pixels with zero original mean output zero.
This is a relative activity index, not a calibrated flow speed or an LSCI calculation.
Both Local PCA modes support 2–64 frames.

For either Local PCA mode, the square kernel can be odd-sized up to the smaller frame dimension
and moves one pixel at a time. Enable **Rectangular kernel** to set width and height separately,
up to the corresponding frame dimensions, and set a horizontal and vertical step. Each step
must fit its kernel dimension and match its parity. Kernels wrap at image borders and write
only their central step-width × step-height output tile; edge tiles are clipped. Neighboring
kernels can overlap, but output tiles do not.

**Local PCA-LSCI** uses a sliding PCA with each frame's kernel mean subtracted to reconstruct a static time series at
each kernel center. Automatic selection reconstructs modes at or above the local mean eigenvalue;
manual selection reconstructs the complement of the Z/Width interval. The **Static (Local PCA then
LSCI 3D)** map is 3D LSCI on the locally reconstructed intensities. **Dynamic** subtracts Static
from 3D LSCI on the input frames; select **LSCI (3D)** to view that input contrast by itself. The
PCA kernel wraps at the image border; the subsequent
LSCI spatial window follows the existing clipped-border rule. The task stores temporal moments
for each reconstructed pixel rather than a second full frame stack. Local PCA-LSCI uses stride 1.
Processed preview and recording convert the selected F32 map to U8. For Dynamic, **Auto scale
Dynamic display** maps each output's minimum and maximum to black and white by default, so
small differences remain visible. A constant map displays black. Turn it off to use the fixed
**Display max K** range; that range clips negative Dynamic values. Auto scaling changes the
displayed U8 image, not the calculated F32 map, and brightness is not comparable across frames.

PCA-LSCI and Local PCA-LSCI use the acquired intensity frames directly, without hologram propagation or temporal Fourier analysis. Local PCA can propagate its component images after PCA; Advanced Local PCA can propagate input frames before PCA. **Accumulation** can average successive maps. If enabled, **Flatfield** is applied to the calculated map before accumulation, and percentile clipping is applied after accumulation. These operations change the processed result; disable them when reproducing an uncorrected $K$ map.

The calculated map is F32 before display conversion. With fixed scaling, **Display max K** maps the range from 0 to the selected value onto U8 grayscale and clips larger values for display. **Invert LSCI** reverses that grayscale. Holovibes uses this same U8 image for the processed preview and processed recording; neither control changes the estimator's F32 output. Raw viewing and raw recording use the original intensity stream.

For a worked configuration based on high-speed retinal imaging, see [Retinal laser speckle contrast imaging](../../../applications/retinal-lsci.md).

## Settings

--8<-- "docs/schemas/syncs/spatial_contrast_settings.md"
