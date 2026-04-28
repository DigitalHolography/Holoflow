# Filter 2D Sync Task
The **Filter 2D** sync task constructs and applies a radial frequency-domain filter to real (F32) or complex (CF32) images on the GPU. The task builds a per-frame frequency mask from the provided radii and smoothing parameters, and uses an FFT-based pipeline (forward FFT → multiply by filter in a CUFFT callback → inverse FFT) to filter each input frame. Real input uses an R2C → mask → C2R path with a compact half-spectrum mask, so its output remains real.

Supported filter behaviors (depending on how `r_inner`, `r_outer`, `s_inner`, and `s_outer` are selected):

- **Low-pass**: choose `r_outer` small and `r_inner == 0` to preserve low spatial frequencies and attenuate high frequencies.
- **High-pass**: choose `r_inner` large and `r_outer` small (or zero) so low frequencies are suppressed and high frequencies preserved.
- **Band-pass**: choose `r_inner < r_outer` to keep a ring of spatial frequencies between the inner and outer radii.
- **Band-stop**: choose `r_inner < r_outer` and invert interpretation (the implemented filter produces a ring-shaped pass region; parameter choices can emulate stop behavior by post-processing).

The filter value at each frequency location is computed from `r_inner`, `r_outer` and their corresponding smoothing widths `s_inner`/`s_outer`. The smoothing parameters produce cosine transitions over the smoothing width so the filter transitions smoothly between 0 and 1 rather than abruptly.

!!! warning
    Very large radius or smoothing values relative to the image size may produce degenerate filters. Ensure your radii/smoothing parameters are chosen with respect to the frame dimensions to obtain the intended frequency response.

## Inputs
This task consumes a single input tensor.

The input requirements (as enforced by `Filter2DFactory::infer`) are:

- The tensor **must** have rank 2 or higher. The filter is applied over the last two dimensions.
- The dtype **must** be `float32` (F32) or `complex32` (CF32).
- The tensor **must** reside in device (GPU) memory.

## Outputs
This task produces one output tensor with the same shape as the input. Complex input keeps dtype `complex32` and supports in-place execution; real input keeps dtype `float32`.

The memory location of the output tensor is device memory.

## Inplace
This task supports in-place operation for complex input. Real input is not in-place because the R2C/C2R implementation uses an internal half-spectrum work buffer.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\syncs\filter2d_settings.md"
