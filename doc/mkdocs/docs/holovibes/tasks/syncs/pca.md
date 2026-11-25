# PCA Sync Task
The **Principal Component Analysis (PCA)** sync task projects real FP32 data onto a reduced set of orthogonal modes. Given an input tensor $X \in\mathbb{R}^{F \times H \times W}$, the task reshapes it into a sample matrix $X_{(N \times F)}$ with $N = H \cdot W$, forms the symmetric covariance

$$
C = X^\mathrm{H} X,
$$

computes the eigenpairs associated with indices `begin` (inclusive) through `end` (exclusive), and finally produces the principal component responses

$$
Y = X \, V_{[\text{begin}:\text{end}]},
$$

where the columns of $V$ are the selected eigenvectors. 

!!! warning
    The input is not mean-centered before decomposition.

!!! note
    Eigenvalues are returned in ascending order. To capture the most energetic components, choose indices close to `F` (for example, `begin = F - K`, `end = F` to keep the `K` largest modes). 

## Inputs
This task has a single input of shape `(F, H, W)`, where:

- `F`: number of features (e.g., frames or spectral slices)
- `H`: height (rows)
- `W`: width (columns)

The dtype must be real 32-bit (`float32`) and the tensor must reside in device memory. Complex
FP32 PCA is not yet supported by the cuSolverDx path.

## Outputs
This task produces a single output of shape `((end - begin), H, W)`, where:

- `end - begin`: number of retained principal components
- `H`: height (same as input)
- `W`: width (same as input)

The dtype of the output tensor is real 32-bit (`float32`) and it is written to device memory.

## Inplace
This task does not support inplace operation.

## Ownership
This task does not own any inputs or outputs.

## Pipelined execution

`PcaDecompositionQueue` and `PcaProjection` provide a streaming alternative to the synchronous
task. The queue accepts contiguous device FP32 `[1,F,H,W]` windows and retains groups of
`batch_size` windows. Its paired outputs are the original input `[B,F,H,W]` and eigenvectors
`[B,F,F]`. Projection accepts these two tensors and the same `begin`/`end` settings as `Pca`.

Covariance runs on the producer section stream, HEEV runs on a private nonblocking stream, and
projection runs on the consumer section stream. Events enforce covariance-before-HEEV and a pop
only succeeds after HEEV completes. This permits covariance and projection for other groups to
overlap the eigensolver without increasing the number of windows per decomposition.

The queue owns each group's original input through projection. Both outputs must be released
before that slot can be reused; releasing X immediately after covariance would be unsafe.
The application's PCA branch uses `batch_size=4`, `slot_count=3`, retaining at most twelve windows.
Partial final groups
remain unpublished, matching the existing accumulation queue. Both ordinary execution and the
runtime's two-ticket push/pop protocol are supported. Task replacement drains private solver work
and invalidates graphs referring to the old buffers.

The existing `Pca` factory remains available for numerical and performance comparison. A matched
benchmark is available as `PcaPipeline.PerformanceOverlap` in `holotask_test`. It alternates baseline
and pipelined runs, using four-window groups, a 256-window warmup, and five ten-second measurements
per mode. `HOLOFLOW_PCA_BENCHMARK_MS` and `HOLOFLOW_PCA_BENCHMARK_RUNS` can shorten diagnostic runs.
`HOLOFLOW_PCA_BENCHMARK_MODE=1` selects baseline only; `2` selects pipeline only (default `0`
compares both). The `PCA pipeline run` NVTX range encloses execution after compilation for capture.
`HOLOFLOW_PCA_BENCHMARK_HEIGHT` changes image height from the default 320 (width stays 512).
Smaller images make covariance inputs ready sooner and help inspect HEEV/GEMM overlap; larger
images can hide HEEV behind input generation and copies instead.

On an RTX 5090 (2026-10-27), the matched benchmark with `[32,320,512]` windows, four-window groups,
and components `[1,16)` measured:

| Execution | Median windows/s | Range across five runs |
| --- | ---: | ---: |
| Synchronous PCA | 6,824 | 6,795–6,861 |
| Pipelined PCA | 8,016 | 7,931–8,057 |

This is a 17.5% improvement in this synthetic streaming benchmark; acquisition and downstream work
can change application throughput. At these dimensions, retained X occupies 240 MiB in either
implementation (twelve windows). Covariance, eigenvalue, and status buffers total
50,736 bytes for three slots, compared with 16,912 bytes for synchronous PCA. Library workspaces
and graph allocations are additional.

The local `build/pca-pipeline-overlap.nsys-rep` capture uses `[32,80,512]` windows to expose direct
HEEV/projection concurrency. For example, HEEV runs on stream 17 from 500.014718 to 500.095808 ms,
while a projection GEMM runs on stream 33 from 500.017310 to 500.021470 ms. These are individual
Nsight Systems timeline intervals, not hardware utilization measurements. The larger-image
`build/pca-pipeline.nsys-rep` capture primarily shows HEEV overlapping input preparation, with
covariance and projection running concurrently.

---
## Settings
--8<-- "docs/schemas/syncs/pca_settings.md"
