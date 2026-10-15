# Spatial Contrast Sync Task

The **Spatial Contrast** sync task calculates local speckle contrast for each input frame. At each pixel, it computes the population standard deviation divided by the mean inside a square sliding window centered on that pixel.

When enabled in Holovibes, **Spatial LSCI contrast** is a dedicated path for laser-speckle intensity input. It applies this task directly to each acquired frame, then uses the existing **Accumulation** setting to average successive contrast maps. Select **Processed** image mode to display the map; lower contrast can indicate more motion or flow. The **Raw** view and raw input recording remain unchanged. This path bypasses hologram propagation, time transforms, FFT shift, and the existing high-pass Flatfield operation because those are not part of the spatial LSCI estimator.

The window is clipped at image borders, so border pixels use only the in-frame samples in their neighborhood. A non-positive local mean produces an output value of zero. The processed preview keeps the F32 contrast values. **Display max K** sets the fixed grayscale range from 0 (black) to the selected value (white); larger values are clipped. This display-only control does not change the calculated or recorded K values, and its fixed range avoids brightness flicker between frames. The raw input buffer is not modified.

## Inputs

This task consumes one GPU-resident, C-contiguous tensor of rank 2 or higher. The last two dimensions are interpreted as image height and width; all leading dimensions are processed as independent frames. Supported input dtypes are U8, U16, and F32.

## Outputs

The output has the same shape as the input and dtype F32. Each frame is processed independently, and the input tensor is not modified.

## Settings

`window_size` is an odd integer from 3 to 255. The default is 7, giving a 7×7 sliding window. Larger windows improve statistical stability but reduce spatial detail; choose a size that spans several speckle grains while retaining the vessel detail you need.

--8<-- "docs/schemas/syncs/spatial_contrast_settings.md"
