# Retinal laser speckle contrast imaging

Laser speckle contrast imaging (LSCI) summarizes fluctuations in camera intensity. The [retinal demonstration by Auray et al.](../references.md#auray-2026) used a pooled spatial and temporal estimator, $K = \sigma_I / \mu_I$, over each 5×5-pixel neighborhood and 16 consecutive frames. The standard deviation uses the population variance. Lower $K$ can indicate faster speckle decorrelation under stable acquisition conditions, but $K$ is not calibrated blood-flow velocity.

## Reported operating point

The following values describe the four recordings in the manuscript. They are an experimental example, not hardware requirements or a benchmark for every Holoflow installation.

| Quantity | Reported value |
| --- | --- |
| Camera input | 3840×2160 pixels, U8, 1000 frames/s, 988 µs exposure |
| Processing hardware | Two Euresys Coaxlink QSFP+ frame grabbers, Intel Xeon host, NVIDIA RTX 5090 |
| Contrast neighborhood | 5×5 pixels across 16 frames; 400 intensity samples at interior pixels |
| Temporal step | 16 frames; nonoverlapping windows |
| Additional processing | No flatfield, percentile clipping, motion correction, or contrast-map averaging |
| Nominal rates | 8.29 Gpixel/s raw input, 62.5 contrast maps/s, 60-frame/s rendering |
| Recordings | Four sequences of 4096 frames, about 4.10 s each, from one eye |

The 16-frame window spans 16 ms at the 1000-frame/s acquisition rate; with 988 µs exposures, its first-to-last exposure support is 15.99 ms. Each 4096-frame sequence contains 256 complete nonoverlapping windows. These are acquisition and estimator quantities. The paper did not independently measure processing latency, queue occupancy, or frame-loss counters, and did not test longer uninterrupted runs.

## Configure Holovibes

1. Load a raw intensity Holofile or configure the supported camera source. Select **Processed** under **Image**.
2. Select **LSCI (3D)** under **Time Transform**. Set **LSCI Window (px)** to `5`, **Time Window** to `16`, **Time Stride** to `16`, and **LSCI method** to **Kernel**.
3. Set **Accumulation** to `1`. Disable **Flatfield** and percentile clipping to obtain the uncorrected estimator used in the manuscript.
4. Set **Display max K** and **Invert LSCI** for the desired grayscale presentation. These controls affect the U8 processed preview and processed recording, not the calculated F32 contrast map.

The 3D LSCI path consumes intensity frames directly. It does not reconstruct a hologram or perform a temporal Fourier transform. **LSCI (2D)** under **Space Transform** instead produces a spatial contrast map for each frame, so it does not reproduce the 5×5×16 estimator. See the [SpatialContrast task reference](../holovibes/tasks/syncs/spatial_contrast.md) for tensor shapes, settings, border handling, and the FFT option.

Raw recording preserves the input intensity sequence. Processed recording saves the display-converted U8 map. To retain numerical $K$ values for quantitative analysis, use the F32 task output before Holovibes display conversion.

## Interpret the result

The manuscript reports visible principal retinal vessels and cardiac-cycle variations in peripapillary recordings from one volunteer. Its regional traces spatially averaged $K$ in selected regions and then calculated the reciprocal of that mean, $1/\bar K$, in arbitrary units. That regional calculation is an analysis of the maps, not part of the SpatialContrast task or the Holovibes display path.

The 400 samples in an interior 5×5×16 neighborhood need not be statistically independent. Contrast depends on exposure, optical sampling, static scattering, detector response, and motion. The reported observations are descriptive; they do not establish absolute flow, layer-specific retinal flow, clinical performance, or a universal conversion from $K$ to velocity.
