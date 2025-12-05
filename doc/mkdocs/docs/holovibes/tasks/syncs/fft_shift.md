# FFT Shift Sync Task
The **FFT shift** task rearranges the quadrants of a batched 2D tensor so that the zero-frequency component moves between the image corners and the center. It mirrors the behavior of `numpy.fft.fftshift` and is typically used after a Fast Fourier Transform to center the spectrum for analysis or visualization.

Internally, each frame is split into four equal quadrants: top-left, top-right, bottom-left, and bottom-right, and opposite quadrants are swapped. The operation runs entirely on the GPU and supports real or complex data.

!!! note
    For the zero-frequency component to land exactly at the center, both the width and height of the input frame should be even. When a dimension is odd, the middle row or column stays in place, matching the truncation behavior of `fftshift`.

## Inputs
This task has a single input of shape `(B, H, W)`, where:

- `B`: batch size
- `H`: frame height
- `W`: frame width

Supported dtypes are `uint8`, `uint16`, `float32`, and `complex32`.  
The input tensor must reside in device memory.

## Outputs
This task produces one output of shape `(B, H, W)`, matching the input dimensions.  
The dtype of the output tensor is identical to the input tensor.  
The output tensor is written in device memory.

## Inplace
This task has an inplace relationship between its input and output.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\syncs\fft_shift_settings.md"
