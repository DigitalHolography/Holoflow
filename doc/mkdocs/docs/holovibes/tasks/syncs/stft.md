# STFT Sync Task
The **Short-Time Fourier Transform (STFT)** sync task computes a forward 1D complex-to-complex FFT along the first axis of its input tensor. For each spatial coordinate `(h, w)` it treats the `B` samples along the first dimension as a temporal window and produces the corresponding frequency spectrum in place on the GPU using cuFFT:

$$
Y[k, h, w] = \sum_{n=0}^{B-1} X[n, h, w] \, e^{-2\pi i \frac{k n}{B}}, \qquad 0 \le k < B
$$

This realizes an STFT with a window length equal to the batch dimension and a hop size of one frame. No windowing or overlap is applied by the task itself.

!!! note
    Apply any desired window function or overlap-add scheme before invoking the task. The current implementation exposes no runtime settings.

## Inputs
This task consumes a single input tensor of shape `(B, H, W)`, where:

- `B`: number of samples in the temporal window
- `H`: image height (rows)
- `W`: image width (columns)

The dtype of the input tensor must be `complex32`. The tensor must reside in device memory.

## Outputs
This task emits one output tensor with shape `(B, H, W)` (identical to the input).

The dtype of the output tensor is `complex32`. The tensor is written to device memory.

## Inplace
This task has an inplace relationship between its input and output.

## Ownership
This task does not own any inputs or outputs.

---
## Settings
--8<-- "docs\schemas\stft_settings.md"
