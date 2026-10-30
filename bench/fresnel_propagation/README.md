# Fresnel FP32 versus FP16

Build on RTX 5090:

```powershell
cmake --preset msvc-multi -DFRESNEL_BENCH_CUDA_ARCH=120
cmake --build --preset build-Release
./build/msvc-multi/Release/fresnel_propagation_bench.exe
```

Use architecture 86 for SM86 GPUs. The executable reports two Google Benchmark cases: production FP32 and automatically tuned FP16. Read the Time column as GPU microseconds per volume; CPU time is host overhead. Volumes/s is GPU throughput.

FP16 startup sweeps supported descriptor pairs across strided, transposed, Y shared, and fused-transposed layouts. Tuning runs once per process and can take a few minutes. Candidates use five samples of 100 graph launches; ranked finalists must pass bounded numerical and replay validation. The selected layout and X/Y EPT/FPB appear in its label. Allocation, tuning, validation, transfers, and graph capture are excluded from final timings. Each measured iteration averages 100 graph replays.

The fixed workload is 32 independent 320-by-512 transforms with a cached input lens, no padding or normalization. Both paths receive identical half-quantized input. FP16 arithmetic and output remain half precision; bounded relative L2 must be below 1%. This benchmark does not imply that large-amplitude inputs avoid FP16 overflow.

Standard options:

```powershell
./build/msvc-multi/Release/fresnel_propagation_bench.exe --benchmark_list_tests
./build/msvc-multi/Release/fresnel_propagation_bench.exe --benchmark_filter=FP32
./build/msvc-multi/Release/fresnel_propagation_bench.exe --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_out=build/fresnel.json --benchmark_out_format=json
```

Listing performs no GPU setup. FP32 filtering skips tuning. FP16 filtering still builds the FP32 correctness reference. Failed execution or validation exits nonzero.

[Historical research results](HISTORICAL.md) and [RTX 5090 report](RTX5090.md) describe the previous executable and timing methodology.
