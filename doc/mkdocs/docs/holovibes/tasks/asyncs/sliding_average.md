# Sliding Average Async Task
The **Sliding Average** async task maintains a rolling mean of the most recent `window_size` frames entirely on the GPU. 

!!! note
    The internal buffer is provisioned with `target_capacity + window_size` slots in device memory. The extra `target_capacity` slots absorb producer/consumer latency.
    
## Inputs
This task consumes a single tensor per push with shape `(1, H, W)`, where:

- `1`: unit batch size (exactly one frame per push)
- `H`, `W`: frame height and width

The tensor must be 32-bit floating point (`float32`) data already located in device memory. `acquire_input` returns a writable view that aliases the internal circular buffer.

## Outputs
After the warm-up period required to enqueue the first `window_size` frames, each pop exposes one tensor of shape `(1, H, W)` with the same dtype and memory location as the input. 

## Inplace
This task does not perform inplace propagation between its input and output tensors.

## Ownership
The task owns both the input and output storage it exposes.

---
## Settings
--8<-- "docs\schemas\asyncs\sliding_average_settings.md"
