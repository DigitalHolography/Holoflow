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
