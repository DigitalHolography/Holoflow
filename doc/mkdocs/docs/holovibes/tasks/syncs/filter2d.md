# Filter 2D Sync Task
The **Filter 2D** sync task constructs and applies a radial frequency-domain filter to complex (CF32) images on the GPU. The task builds a per-frame complex filter (CUFFT callback) from the provided radii and smoothing parameters, shifts corners (FFT quadrant swap), and uses an FFT-based pipeline (forward FFT → multiply by filter → inverse FFT) to filter each input frame.

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

- The tensor **must** be 3D with shape `(B, H, W)` (batch, height, width).
- The dtype **must** be `complex32` (CF32).
- The tensor **must** reside in device (GPU) memory.

## Outputs
This task produces one output tensor with the same shape and dtype as the input (the filter is applied in-place relative to shape/dtype but the implementation may use intermediate buffers).

The memory location of the output tensor is device memory.

## Inplace
This task supports in-place operation for the tensor shapes used (the infer result returns an in-place pair), however the implementation constructs FFT plans and may use device buffers; the net effect is that the output has the same shape and dtype as the input.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\syncs\filter2d_settings.md"
