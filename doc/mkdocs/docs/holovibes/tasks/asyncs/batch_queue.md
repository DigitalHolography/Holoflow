# Batch Queue Async Task
The **Batch Queue** async task implements a lock-free circular buffer that bridges an upstream
producer and a downstream consumer. Each push writes an entire input tensor into the queue.
When at least `output_stride` batches have accumulated, the consumer can pop a view containing
`output_size` consecutive batches. Releasing that view advances the read cursor by
`output_stride`, which enables decimation or sparse sampling of the buffered stream without extra
copies.

Internally the capacity is rounded up to the least common multiple of the input size and the stride
so that both writers and readers wrap around on aligned boundaries. This guarantees that every
output window is contiguous in memory.

!!! note
    The queue never copies data between host and device. The circular buffer is allocated in the
    same memory space as the input descriptor, so the output tensor views alias the storage that the
    producer filled.

## Inputs
This task accepts a single tensor with rank ≥ 1. Let the input shape be `(N, d1, ..., dk)`:

- `N`: number of batches pushed at once (the leading axis)
- `d1, ..., dk`: remaining spatial or channel dimensions inherited by the queue

All element dtypes are supported because the queue stores opaque bytes. The tensor must already
reside in either host or device memory; the queue allocates its internal buffer in the same memory
space and provides a writable view to the producer.

## Outputs
One output tensor is produced per pop call with shape `(output_size, d1, ..., dk)`. The dtype and
memory location match the input descriptor, and the data is a view into the queued storage starting
at the current read position.

## Inplace
This task does not support inplace operation.

## Ownership
The task owns both its input and output memory.

---
## Settings
--8<-- "docs\schemas\asyncs\batch_queue_settings.md"
