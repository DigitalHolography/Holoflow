# Fresnel propagation precision benchmark

Compares the production cached-lens FP32 cuFFT callback against FP16 cuFFTDx
propagation on an SM86 GPU. Each volume is `(32, 320, 512)` in ZYX order:
32 independent forward 320-by-512 XY FFTs, with no padding or normalization.

## Build and run

From the repository root, with the MSVC/CUDA environment initialized:

```powershell
cmake --preset msvc-multi
cmake --build --preset build-Release

# Sweep and select the fastest validated complete pipeline.
./build/msvc-multi/Release/fresnel_propagation_bench.exe --sweep

# Study both fused layouts against the earlier controls.
./build/msvc-multi/Release/fresnel_propagation_bench.exe --study-fused

# Reproduce the fused-transposed finalist.
./build/msvc-multi/Release/fresnel_propagation_bench.exe --config fused-transposed 7 4

# Reproduce the earlier strided winner.
./build/msvc-multi/Release/fresnel_propagation_bench.exe --config strided 0 12

# Reproduce the fastest transposed configuration from that sweep.
./build/msvc-multi/Release/fresnel_propagation_bench.exe --config transposed 6 5

ctest --preset test-Release -R Fresnel --output-on-failure
```

No argument also runs the sweep. Configuration indices refer to the `STAGE`
rows emitted by the sweep; the selected pair is printed at the end. Positional
arguments `0`, `1`, and `2` remain available for the aligned control, the earlier
X EPT=8 / Y EPT=10 / FPB=4 strided experiment, and its transposed equivalent.
Modes `3` and `4` preserve the initial fixed-descriptor fused experiments.
Named layouts additionally accept `y-shared` and `fused-transposed`.
`--check-config layout X Y` runs validation without timing loops; its timing
fields are zero and do not represent performance. Invalid command arguments
or failed validation return a nonzero exit status.

## Precision and execution

Real input values are quantized to FP16, then widened once for the FP32 path,
so both paths receive the same values. Wavelength is 852 nm, pixel pitch is
20 micrometres, and propagation distance is 0.1 m. Only the input quadratic
phase is applied. Both paths cache one complex FP32 lens plane before timing
and reuse it across all 32 batches.

FP32 uses `FresnelDiffractionFactory` with its fused cuFFT input callback.
FP16 fuses lens multiplication into the X FFT load: multiplication is FP32,
then converted into half FFT registers. FFT arithmetic, intermediate storage,
and output components are FP16. Complex16 occupies four bytes and has
four-byte alignment. Tuned kernels pack halves directly into paired `half2`
registers and load/store aligned complex samples. Pair indexing handles both
explicit FFTs per block and cuFFTDx's implicit two-FFT batching.

The aligned control retains the original descriptors and packing conversions,
but shares the current aligned complex load/store implementation. Historical
executables used different memory access code; their timings are not an
isolated comparison of descriptor settings.

The sweep registers the original descriptors plus all supported configurations
from these sets:

- X elements per thread (EPT): 4, 8, 16.
- Y elements per thread: 10, 20, 40.
- FFTs per block (FPB), independently for each axis: 2, 4, 8, 16, 32.

cuFFTDx support traits, workspace requirements, kernel thread/register limits,
and device shared-memory limits filter configurations before launch. The
current SM86 build registers 15 X and 16 Y configurations per layout. New
FPB=32 entries are appended after the existing indices; X EPT=4 / FPB=32
would require 2048 threads and is excluded before descriptor completion.
The ordinary sweep times all 240 X/Y pairs for each of two layouts (480 pipelines):
strided Y, and contiguous Y with separate tiled transposes before and after it.
Padded 32-by-33 shared tiles restore ordinary ZYX output order.

## Timing and validation

Each measurement warms five times, captures a CUDA Graph, and collects five
samples of 100 graph launches using CUDA events. Full-pipeline timings include
both transpose kernels where applicable. Setup, allocation, input widening,
lens generation, plan/callback compilation, transfers, and validation are
excluded. Stage timings use separate graphs and are diagnostic: their sum
need not equal the full-pipeline result because cache conditions differ.

All pipeline pairs are checked against FP32 on the deterministic sine input.
Each axis configuration is additionally checked in a complete pipeline on
bounded random input, zero, constant input, and impulses at distinct XY
positions and amplitudes in every Z batch. The control and best pipeline of
each layout receive all these cases, graph replay checks, and byte comparisons
verifying unchanged FP16 and FP32 inputs. Replay output must be bit-identical.

Bounded cases require no nonfinite output and relative L2 error below 1%; zero
input must remain exactly zero. FP32 output is independently checked against
an explicit lens-plus-cuFFT reference with relative L2 error below 1e-5.
A 60,000-amplitude constant input deliberately stresses overflow: its errors
and nonfinite count are reported without failing the bounded-input checks.
Nonfinite output makes reported relative and maximum errors infinite.
No automatic scaling or output-precision fallback is applied.

Selection uses the lowest median full-pipeline latency among validated
configurations. Production propagation, public tensor types, and dependencies
are unchanged.

## Results before fused transpose: RTX 3090, CUDA 13.3, Release

Sweep results, five timing samples per pipeline:

| Pipeline | X EPT / FPB | Y EPT / FPB | Min / median / max (ms) | Sine relative L2 |
|---|---|---|---|---|
| Current aligned control | 8 / 2 | 40 / 2 | 0.177606 / 0.182994 / 0.189226 | 0.00405434 |
| Earlier selected strided | 8 / 2 | 40 / 16 | 0.111327 / 0.111777 / 0.116311 | 0.00405434 |
| Best transposed | 8 / 4 | 20 / 2 | 0.202273 / 0.203115 / 0.211591 | 0.00230374 |
| Production FP32 | - | - | 0.190474 / 0.192726 / 0.200977 | reference |

The selected configuration is 1.64x faster than the current control and 1.72x
faster than FP32 in that sweep. Five subsequent independent process runs gave
median FP16 latencies of 0.113300-0.116431 ms, versus 0.194071-0.198973 ms for
FP32 (paired speedups approximately 1.69-1.75x).
Five independent process runs of the control gave median latencies of
0.188065-0.194608 ms; the transposed finalist gave 0.207234-0.211264 ms.

All bounded validations passed, with zero nonfinite samples. The stress case
produced 5,242,880 nonfinite samples in every finalist. All nine existing
Fresnel correctness tests passed; the full Release build succeeded.

Separate stage graph medians in the sweep: control X 0.052760 ms, control Y
0.139120 ms, selected Y 0.062046 ms, forward transpose 0.052929 ms, and reverse
transpose 0.053136 ms. The best transposed pipeline's Y stage was 0.056173 ms.

Fresh Nsight Compute measurements on cached warmup launches:

| Kernel | Threads/block | Registers/thread | Active threads/warp | Achieved occupancy | Duration |
|---|---|---|---|---|---|
| Control X | 64 | 39 | 32 | 63.72% | 49.92 us |
| Control Y | 8 | 151 | 8 | 23.90% | 140.13 us |
| Selected Y | 64 | 152 | 32 | 23.08% | 59.04 us |
| Best transposed Y | 16 | 72 | 16 | 32.70% | 51.97 us |

For Y, reported excess global-memory sectors fell from 88% in the control to
50% in the selected kernel. Neither kernel spilled registers. Occupancy alone
does not explain the improvement: grouping more FFTs into each block fills
warp lanes and improves memory transactions despite similar register-limited
occupancy. Transposing makes Y fast, but its two extra full-volume passes
outweigh that advantage in the complete pipeline. Profiler durations are
separate diagnostic measurements, not substitutes for unprofiled graph timing.

The remaining shared-memory conflicts and excess global sectors are potential
future work. These measurements do not claim the best possible cuFFTDx
implementation; they establish a faster configuration within this sweep.

## Reproduce profiling

Use the installed Nsight Compute CLI (not Nsight Systems) for kernel counters:

```powershell
$ncu = 'C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/ncu.bat'
& $ncu --kernel-name regex:fft_pass --launch-skip 2 --launch-count 2 `
  --section LaunchStats --section Occupancy --section SpeedOfLight `
  --section MemoryWorkloadAnalysis --section WarpStateStats --section SourceCounters `
  --log-file build/fresnel-best-ncu.txt `
  ./build/msvc-multi/Release/fresnel_propagation_bench.exe --config strided 0 12
```

Skipping the first two matching launches excludes the initial validation pair;
the next pair is cached X/Y warmup work. Use `--config strided 0 0` for the
control or `--config transposed 6 5` for the transposed finalist. Do not compare
event timings collected during profiling with independent unprofiled timings.

Local evidence is written under `build/`: `fresnel-sweep.log`,
`fresnel-best-repeat.log`, `fresnel-control-ncu.txt`, `fresnel-best-ncu.txt`,
`fresnel-transposed-ncu.txt`, `fresnel-control-repeat.log`,
`fresnel-transposed-repeat.log`, and `fresnel-tests.log`.

## Completed fused-transpose study

`--study-fused` adds two layouts, each using exactly two FFT kernel launches:

- **y-shared:** standard X output in ZYX order; Y cooperatively loads a padded
  `[320][FPB+1]` shared tile, gathers paired FFT inputs, then stages its output
  through the same tile for coalesced writes.
- **fused-transposed:** X stages its FFT output in `[FPB][513]` and writes a ZXY
  intermediate; Y reads contiguous inputs and stages its output through
  `[320][FPB+1]`, restoring ordinary ZYX output order without another kernel.

A block uses one 16-byte-aligned dynamic shared allocation sized to the larger
of the FFT scratch and transpose tile. Block-wide barriers separate each use.
Both variants retain half arithmetic and four-byte complex16 storage.

The study measures all 240 descriptor pairs in each new layout, plus the
previous strided winner and standalone-transpose finalist: **482 pipelines**.
It checks every fused sine result bit-for-bit against the unfused strided
pipeline with matching FFT descriptors. Every axis configuration additionally
receives all bounded cases, and finalists receive all six inputs, graph replay,
and unchanged-input checks. The same-descriptor identity check also runs in
single-configuration mode. Consequently, accuracy changes between different
EPT settings come from FFT implementation choices, rather than transposition.

### Sweep and independent repetitions

RTX 3090, CUDA 13.3, Release. Each timing sample contains 100 graph launches.

| Pipeline | X EPT / FPB | Y EPT / FPB | Sweep median | Five-process median range | Median of process medians |
|---|---|---|---|---|---|
| Earlier strided winner | 8 / 2 | 40 / 16 | 0.114513 ms | 0.113573-0.119722 ms | 0.115340 ms |
| Standalone transposes | 8 / 4 | 20 / 2 | 0.208691 ms | 0.204438-0.211400 ms | 0.207891 ms |
| Y shared staging | 8 / 4 | 10 / 16 | 0.108845 ms | 0.108209-0.111959 ms | 0.111009 ms |
| Fused transposed intermediate | 8 / 8 | 10 / 16 | 0.107635 ms | 0.108292-0.111383 ms | 0.109878 ms |

The independent runs rotate the four layouts' execution order on each round.
Fused transposition reduces median latency by approximately 4.7% relative to
the earlier strided winner, and gives approximately 1.8x FP32 throughput in
these runs. It and Y shared staging are close; their timing ranges overlap.
The sweep selects the lowest validated median, preserving every variant for
comparison. FPB=32 configurations were validated but did not win.

Both new finalists have sine relative L2 error **0.00170983 (0.171%)**, maximum
absolute error 2.47860, and no nonfinite bounded outputs. The stress input still
produces 5,242,880 nonfinite samples. Scaling and output precision are unchanged.

Same-descriptor diagnostic controls measured 0.258771 ms for strided X=6/Y=4,
0.254199 ms for strided X=7/Y=4, and 0.211406 ms for standalone transposes
X=7/Y=4. Those timings each use five samples in one process and demonstrate
that shared reordering makes the EPT=10 configuration competitive. The headline
comparison above remains against the faster, independently tuned strided winner.

### Profiling and sanitizer evidence

Fresh Nsight Compute profiles of cached warmup kernels:

| Pipeline / kernel | Duration | Registers/thread | Achieved occupancy | DRAM throughput |
|---|---|---|---|---|
| Earlier strided X | 50.66 us | 39 | 64.33% | 76.87% |
| Earlier strided Y | 60.29 us | 152 | 23.30% | 73.62% |
| Y shared staging X | 49.63 us | 39 | 85.33% | 76.49% |
| Y shared staging Y | 60.74 us | 48 | 65.52% | 75.81% |
| Fused transposed X | 58.94 us | 40 | 75.26% | 65.71% |
| Fused transposed Y | 50.56 us | 46 | 64.44% | 90.66% |

No measured kernel spilled registers. Fusing transposition improves Y but adds
X staging work, explaining the modest overall gain. Shared-memory conflicts
remain. Profiler durations are diagnostic and should not be summed as a
replacement for the independent full-pipeline timings.

Compute Sanitizer memcheck, racecheck, and synccheck each checked both initial
FFT launches on the complete volume for four configurations: both finalists,
Y shared staging X=0/Y=15 (FPB=32), and fused transposition X=13/Y=15
(FPB=32, including the 65,664-byte X shared tile). All **12 checks** reported
zero errors; racecheck also reported zero warnings. Kernel filtering excludes
third-party cuFFT kernels and limits instrumentation to the first pair. Other
input cases and graph replays still run normal numerical validation.
The full Release build and all nine existing Fresnel correctness tests pass.

Reproduce a finalist and the largest shared-tile check:

```powershell
$exe = './build/msvc-multi/Release/fresnel_propagation_bench.exe'
& $exe --config y-shared 6 4
& $exe --config fused-transposed 7 4
& $exe --check-config fused-transposed 13 15

foreach ($tool in @('memcheck', 'racecheck', 'synccheck')) {
  compute-sanitizer --tool $tool --kernel-name kns=fft_pass --launch-count 2 `
    --error-exitcode 99 $exe --check-config fused-transposed 13 15
}
```

For profiling a fused mode, use `--launch-skip 4` instead of 2: the first pair
is validation and the second pair is the matched unfused reference. The next
pair is the fused warmup work. For the original layouts, continue using skip 2.

```powershell
$ncu = 'C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/ncu.bat'
& $ncu --kernel-name regex:fft_pass --launch-skip 4 --launch-count 2 `
  --section LaunchStats --section Occupancy --section SpeedOfLight `
  --section MemoryWorkloadAnalysis --section WarpStateStats --section SourceCounters `
  --log-file build/fused-transposed-ncu.txt `
  ./build/msvc-multi/Release/fresnel_propagation_bench.exe --config fused-transposed 7 4
```

To reproduce alternating independent repetitions:

```powershell
$configs = @(
  @('strided', 0, 12), @('transposed', 6, 5),
  @('y-shared', 6, 4), @('fused-transposed', 7, 4)
)
for ($run = 0; $run -lt 5; $run++) {
  for ($offset = 0; $offset -lt 4; $offset++) {
    $c = $configs[($run + $offset) % 4]
    & $exe --config $c[0] $c[1] $c[2]
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark validation failed' }
  }
}
```

Evidence under `build/`: `fresnel-fused-study.log`,
`fresnel-fused-alternating.log`, `fused-matched-controls.log`,
`fused-strided-ncu.txt`, `fused-y-shared-ncu.txt`, `fused-transposed-ncu.txt`,
`fused-{memcheck,racecheck,synccheck}-*.log`, and `fresnel-tests-final.log`.
