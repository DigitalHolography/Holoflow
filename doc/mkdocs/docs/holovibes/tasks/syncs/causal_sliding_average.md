# Causal Sliding Average

The **Causal Sliding Average** is a synchronous, stateful GPU task. It accepts any contiguous,
device-resident `float32` tensor and returns a tensor with the same descriptor for every input.

For the first `window_size - 1` inputs, the output is divided by the number of samples observed.
Once the window is full, the task emits the mean of the most recent `window_size` tensors. A GPU
ring buffer and running sum make each execution constant in the window length.

Task creation or pipeline rebuilding clears the stored history.

## Settings

- `window_size`: number of tensors in the causal window; must be positive.
