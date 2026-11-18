# PCA precision benchmark

```powershell
cmake --preset msvc-multi -DPCA_BENCH_CUDA_ARCH=120
cmake --build --preset build-Release
./build/msvc-multi/Release/pca_bench.exe
```

Use architecture 86 for SM86 hardware. The default workload is one 32-by-320-by-512 U8 window, selecting eigenvectors [1,16) in ascending eigenvalue order. This matches production's uncentered Gram-matrix PCA; no mean subtraction or normalization of input values occurs.

Three complete pipelines are reported:

* `PCA/U8_to_FP32`: production real-value U8-to-FP32 conversion plus production PCA, FP32 output.
* `PCA/FP32`: production PCA on the same preconverted values, FP32 output. Preconversion is excluded.
* `PCA/U8_Fused_FP16_Autotuned`: fused U8 loading into CUTLASS half-precision tensor-core multiplication, FP32 Gram accumulation/reduction, runtime-compiled cuSolverDx FP32 eigensolve, and fused projection to FP16 output. No full-volume converted input is materialized.

The direct path has lower output precision. Production already permits FP16 tensor-core computation through cuBLAS FAST_16F, despite FP32 storage. Reported ratios include differences in output bandwidth and numerical precision.

The installed cuBLASDx and cuSolverDx releases reject MSVC compilation. This benchmark uses the bundled CUTLASS for GEMMs and NVRTC/nvJitLink for cuSolverDx, matching the production solver approach. Runtime assets are resolved from this build's configured CUDA/MathDx paths; this executable is intended for build-tree use.

FP16 tuning runs once per process across 36 tile/block configurations, rejecting candidates that fail accuracy checks. Five samples of 100 complete graph replays select the lowest median. The selected configuration appears in the label. Tuning, plans, allocations, validation, compilation, transfers, and graph capture are excluded from reported latency. Final Google Benchmark iterations each average 100 graph replays. Read Time as GPU microseconds per window; CPU is host overhead. `windows/s` reports GPU throughput.

Gram matrices are scaled by an exact power of two before the eigensolve to improve numerical range, and eigenvalues are rescaled afterward. Vendor nonconvergence uses a bounded FP32 Jacobi fallback, with a checked convergence result. The zero matrix has an explicit identity-basis solution. The runtime-compiled vendor solver and native fallback are separate launches, both included in the timed graph.

```powershell
./build/msvc-multi/Release/pca_bench.exe --benchmark_list_tests
./build/msvc-multi/Release/pca_bench.exe --benchmark_filter='PCA/FP32'
./build/msvc-multi/Release/pca_bench.exe --benchmark_min_time=0.01s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_out=build/pca-google.json --benchmark_out_format=json
ctest --preset test-Release -R PcaBenchmarkValidation --output-on-failure
```

Listing performs no GPU setup, and baseline-only filtering skips tuning. Execution and validation failures return nonzero. Correctness checks use FP64 cuBLAS/cuSOLVER references, with 1% limits for Gram error, selected-subspace error, aligned projected output, orthogonality, and solver residuals. For degenerate eigenspaces, projection is checked against the actual eigenbasis and normalized by input energy when the selected output approaches zero.

[RTX 5090 measurements](RTX5090.md).

## Production FP16 lower-triangle study

Run on an otherwise idle RTX 5090 from the MSVC development shell:

```powershell
python bench/pca/f16_study.py --nsys 'C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe'
```

The runner builds both values of the default-off `PCA_F16_LOWER_TRIANGLE` option
with the same Release preset. It validates the lower Gram triangle and sign-aligned
FP16 projections across builds, checks graph replay and runs Compute Sanitizer
initcheck. It then measures standalone production Gram and covariance before
measuring complete production PCA and queue throughput. Nsight Systems captures
run separately from the timing runs. Omit `--nsys` for timing and validation only.

Raw logs and CSVs are under `build/pca-f16-study`. GPU timings are microseconds
per window. Each latency sample averages 100 graph replays, so its p95 describes
the spread of those averages. Queue values are completed windows per second.
The runner restores the default-off build configuration when it exits.

### RTX 5090 study results

Median of the ten paired relative changes. Lower is faster for GPU time.

| Stage | Windows | Components | Paired change | Pairs improved |
| --- | ---: | ---: | ---: | ---: |
| Gram | 1 | — | −2.85% | 6/10 |
| Gram | 4 | — | −11.22% | 10/10 |
| Gram + reduction | 1 | — | −4.47% | 10/10 |
| Gram + reduction | 4 | — | −10.75% | 10/10 |
| Full PCA | 1 | 1 | −1.95% | 7/10 |
| Full PCA | 1 | 16 | −1.33% | 7/10 |
| Full PCA | 4 | 1 | −3.17% | 9/10 |
| Full PCA | 4 | 16 | −3.25% | 10/10 |
| Queue throughput | 4 | 16 | +3.73% | 10/10 |

For four windows and 16 components, median full-PCA latency went from 48.12 to
46.53 µs per window and p95 went from 53.16 to 49.32 µs. For one window and one
component, pooled p95 went from 105.64 to 108.05 µs, with a higher p95 in
four of ten pairs. Queue throughput ranged from
3,501–14,469 windows/s at baseline and 9,556–14,825 with the candidate. The
variance limits the throughput estimate.

Both builds passed batched FP16 PCA and queue graph-replay tests. Validation
signatures matched and candidate initcheck reported zero errors. The baseline
produced nonfinite projections on the all-255 input with 16 components, so this
input is used for the Gram comparison only. Projection validation includes an
all-zero window and a rank-deficient window with a known null component.

Both Nsight Systems captures showed HEEV overlapping Gram and projection on
separate CUDA streams. The four-window, 16-component workload is the strongest
case for enabling the variant. The study option remains off by default.
