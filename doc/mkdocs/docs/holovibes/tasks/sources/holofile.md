# Holofile Source Task

This task reads data from a `.holo` file and outputs it as a tensor. When the end of the file is reached, it loops back to the beginning. The task can be configured to load data in different ways, such as loading all frames into memory at once or loading frames on-the-fly. The file specified must exist and be a valid `.holo` file during the task's validation phase and execution. An optional `max_fps` setting can be used to throttle playback; if it is omitted, frames are emitted as fast as possible.

## Inputs
This task does not have any inputs.

## Outputs
This task has a single output of shape `(B, H, W)`, where:

- `B` is the batch size specified in the settings.
- `H` and `W` are the height and width of the frames in the `.holo` file.

The dtype of the output tensor is unsigned `N`-bit integer, where `N` is the bits per pixel specified in the file header.
If the file bits per pixel is not 8, 16, or 32, the output tensor will have dtype `uint8`.

The memory location of the output tensor is determined by the `load_kind` setting. If `load_kind` is set to `Live` or `CPUCached`, the output tensor will be in host memory. If `load_kind` is set to `GPUCached`, the output tensor will be in device memory.

## Inplace
This task does not support inplace operation.

## Ownership
Cached modes own their output views into an immutable preloaded buffer. CPU-cached storage is
pinned host memory. Live mode writes into the compiler-provided output buffer and does not own it.

## Iteration pre-queuing

Uncapped cached playback supports two ordinary submissions in flight. In the CPU-cached
`Holofile → Memcpy → BatchQueue` section, the source selects the next cached batch and enqueues
its H2D copy before waiting for the previous iteration. The runtime retains each source view
until its completion event, publishes copied batches in FIFO order, and preserves source state
across pause/resume. Cached pointers are relative to `start_frame`, including after wraparound.

Live disk reads and playback with `max_fps` remain serial. The source does not record CUDA graphs:
advancing its cursor and publishing an output view are host-side operations.

`OrdinaryPrequeue.PerformanceCachedCopy` benchmarks uncapped CPU-cached H2D playback with only
section-0 forced serial versus pre-queued. `OrdinaryPrequeue.PerformanceCachedPcaPipeline` uses
the same source plus conversion and pipelined PCA. Both alternate five paired ten-second runs,
using 32-frame windows of 320×512 U8 pixels and a 128-window warmup. Set
`HOLOFLOW_SOURCE_BENCHMARK_MS` and `HOLOFLOW_SOURCE_BENCHMARK_RUNS` for shorter diagnostic runs;
`HOLOFLOW_SOURCE_BENCHMARK_MODE=1` selects serial only and `2` selects prequeue only (default `0`
compares both). Registered NVTX ranges `Cached playback serial` and `Cached playback prequeued`
enclose execution after compilation for profiling.

On an RTX 5090 (2026-10-27), the five paired ten-second measurements gave:

| Workload | Serial section-0, windows/s | Prequeued section-0, windows/s | Median change |
| --- | ---: | ---: | ---: |
| Cached H2D playback | 9,499 | 9,748 | +2.6% |
| Cached playback with pipelined PCA | 9,321 | 9,330 | +0.1% |

The PCA result is within run-to-run variation: serial runs ranged from 9,181–9,382 windows/s,
and prequeued runs from 9,321–9,346. These synthetic fixtures demonstrate a modest transfer-only
gain, without a clear end-to-end PCA throughput improvement. Local Nsight captures for the H2D
comparison are `build/section0-serial.nsys-rep` and `build/section0-prequeued.nsys-rep`.

The prequeued capture contains adjacent 5 MiB copies where the next `cudaMemcpyAsync` returns
at 500.205462 ms before the preceding GPU copy completes at 500.311059 ms. In the serial sample,
the next API call starts at 500.229111 ms after the preceding copy completes at 500.216692 ms.
These individual timeline examples confirm submission look-ahead; the throughput measurements
above come from unprofiled runs.

A separate trial compared the earlier eager-retirement build with the nonblocking refill
policy and additional backpressure NVTX ranges, with prequeue enabled in both. Three paired
five-second runs on the same GPU and fixture, alternating build order, gave:

| Workload | Eager retirement, median windows/s | Nonblocking refill, median windows/s | Change |
| --- | ---: | ---: | ---: |
| Cached H2D playback | 9,740 | 9,767 | +0.3% |
| Cached playback with pipelined PCA | 9,331 | 9,239 | −1.0% |

The refill regression test confirms that newly available buffers can be used while a remaining
GPU iteration is incomplete; this short throughput trial shows no PCA performance gain.
The comparison includes the new instrumentation and does not isolate its overhead from the
policy change. Copy runs ranged from 9,702–9,744 (eager) and 9,640–9,771 (new); PCA runs from
9,322–9,335 (eager) and 9,225–9,284 (new). See the
[task-model NVTX guide](../../../holoflow/concepts/task-model.md#cuda-iteration-pre-queuing)
for the backpressure labels.

---
## Settings
--8<-- "docs/schemas/sources/holofile_settings.md"
