# Convolution Sync Task

The **Convolution** task performs 2D convolution on input images using the Fast Fourier Transform (FFT) method for efficient computation. This approach transforms both the input image and kernel to the frequency domain, performs element-wise multiplication, and then transforms back to the spatial domain.

The convolution operation in the spatial domain:

$$(f * g)(x, y) = \sum_{i=-∞}^{∞} \sum_{j=-∞}^{∞} f(i, j) \cdot g(x-i, y-j)$$

is equivalent to multiplication in the frequency domain:

$$\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$$

where:
- $f$ is the input image
- $g$ is the convolution kernel
- $\mathcal{F}$ denotes the Fourier transform

The task implements this using CUDA's cuFFT library for high-performance FFT computation on GPU.

## Inputs
This task has a single input of shape `(B, H, W)` or `(H, W)`, where:
- `B`: batch size (optional)
- `H`: height of the input image
- `W`: width of the input image

The dtype of the input tensor must be 32-bit float (`float32`).

The memory location of the input tensor must be device memory.

## Outputs
This task has a single output of shape `(B, H, W)` or `(H, W)`, where:
- `B`: batch size (same as input)
- `H`: height of the output image (same as input)
- `W`: width of the output image (same as input)

The dtype of the output tensor is 32-bit float (`float32`).

The memory location of the output tensor is device memory.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

## Kernel Format
The convolution kernel is loaded from a JSON file with the following format:
```json
{
  "kernel": [
    [k11, k12, k13, ...],
    [k21, k22, k23, ...],
    [k31, k32, k33, ...],
    ...
  ]
}
```
All rows must have the same width, and the kernel must be non-empty.

---
## Settings
--8<-- "docs\schemas\syncs\convolution_settings.md"