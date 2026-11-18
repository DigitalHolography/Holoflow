# Production FP16 PCA and Fresnel

The basic PCA/Fresnel/M0 pipeline enables FP16 through
`src/holovibes/src/pipeline/precision_options.hh`. All four switches default to `true`:

- `enable_f16_pca`: U8 covariance and projection with CUTLASS tensor operations, FP32 accumulation and the existing FP32 eigensolver; F16 projected output.
- `enable_f16_filter2d`: normalized real spatial filtering with F16 input/output and native FFTs on validated devices.
- `enable_f16_fresnel`: runtime-compiled cuFFTDx FFTs with F16 complex intermediates.
- `fuse_fresnel_mean`: combine magnitude and component mean; F32 reduced output.

Set all switches to `false` and rebuild to restore the previous graph. PCA and
Fresnel can also be disabled independently. F16 PCA output is widened before legacy
Fresnel when its switch is disabled.

The initial application scope is U8 windows `[32,320,512]`, PCA selection of 1-16
components, Fresnel propagation and M0, without 3D cuts. Other
configurations retain FP32 PCA. Optimized Fresnel supports contiguous real F16/F32
with trailing 320x512 dimensions on SM86/SM120. Unsupported Fresnel configurations
or runtime setup failures use legacy FP32 computation, widening F16 input and
preserving the requested reduced F32 output shape.

PCA still groups four windows with three queue slots. Both neighboring queues own
storage, so a U8 device copy separates the input queue and PCA queue; the optimized
route has no full U8-to-F32 conversion. Covariance remains the same uncentered,
unnormalized Gram matrix. The eigensolver and component ordering are unchanged.
Autofocus retains the eligible FP16 PCA queue and its input format across
toggles. The Shack–Hartmann Fresnel callback and phase correction load F16
fields and compute in FP32. Corrected fields normally use CF32, including
multi-pass correction. For a single pass on SM120, the default-off
`enable_cf16_correction` option in `precision_options.hh` enables CF16 storage
between phase correction and fused Fresnel. Its fallback widens the field to
CF32 and preserves the requested magnitude output and reduction.

Mixed precision does not promise bitwise equality with FP32.

Fresnel derives power-of-two scales per input plane to bound both FFT stages and
restores magnitudes in FP32. Component reduction accumulates in FP32, avoiding an
overflowing F16 magnitude buffer. There is no application accuracy threshold or
per-frame FP32 reference computation.

Runtime-only CUDA sources are formatted files under `syncs/pca/runtime/` and
`syncs/fresnel_diffraction/runtime/`. CMake embeds them in generated headers,
tracks changes and excludes them from native MSVC compilation. Host helpers and
native PCA kernels live in separate files in the parent folders. Installed NVRTC
assets now include cuFFTDx headers alongside CUDA/cuSolverDx assets.

## Build and verify

```powershell
cmake --preset msvc-multi
cmake --build --preset build-Release
ctest --preset test-Release
```

Run `build/msvc-multi/Release/holovibes.exe`. An installed or already running older
executable does not acquire these changes. The log reports
`[Fresnel] fused F16 + magnitude mean` and
`[PCA] fused U8/F16 covariance and projection; FP32 eigensolver` when created.
In a trace expect `pca_f16::gram`, `pca_f16::project`, `fused_f16_scale`,
`fused_f16_fft_x`, `fused_f16_fft_y_mean` and `fused_f16_finish`.
`pca_heev_batched` remains expected: the solver deliberately stays FP32.

Focused tests cover batched U8 PCA, queue wraparound and CUDA graph replay,
Fresnel high-amplitude finite output, changed-input replay and unsupported-size
fallback. An optional headless recording test mirrors the basic graph:

```powershell
$env:HOLOFLOW_F16_RECORDING_PATH = 'D:/260312_GUJ_1.holo'
$env:HOLOFLOW_F16_RECORDING_MS = '3000'
$env:HOLOFLOW_F16_RECORDING_MODE = '3'
./build/msvc-multi/Release/holotask_test.exe --gtest_filter=ProductionF16.RecordingBasicPipeline
```

Modes: `0` legacy, `1` FP16 PCA only, `2` FP16 Fresnel only, `3` both. The source
caches frames 0-16384, loops 32-frame batches, warms 256 outputs and measures
completed output windows/s. This includes scheduler, queues, transfers and a
synchronous one-byte headless sink; it is not kernel-only timing or GUI FPS.
Without the path environment variable this recording test is skipped.

## RTX 5090 recording results (2026-10-30)

Release/MSVC, `D:/260312_GUJ_1.holo`, 320x512 U8 frames, 32-frame windows,
16 PCA components, four-window/three-slot PCA queue, 64-window averaging.
Each measurement uses a separate process, 256 warm-up outputs and a 3-second
measurement interval. Runs execute sequentially without concurrent GPU tests.

| Mode | Output windows/s | Runs |
| --- | ---: | ---: |
| Legacy FP32 | 4626-4673 | 3 |
| FP16 PCA, legacy Fresnel | 4276-4364 | 2 |
| Legacy PCA, FP16 Fresnel + mean | 4826 | 1 |
| FP16 PCA and Fresnel + mean | 4720-4808 | 2 |

The combined path is about 2-4% faster than the adjacent legacy run in this
headless setup. PCA-only is slower, including the extra F16-to-F32 widening needed
by legacy Fresnel. These short measurements include CPU scheduling and sink
synchronization, so they do not establish a GUI FPS or isolated GPU speedup.
They provide no numerical accuracy comparison.

The initial implementation was slower. Inspection found a one-block-per-plane
scaling scan and individually submitted Fresnel kernels. The final version uses
a split scaling reduction and the existing Fresnel CUDA graph cache for rotating
input/output addresses. External section-graph capture is also supported.

Validation: Release configure/build; full CTest suite (583 tests, zero failures,
optional recording test skipped); separate recording runs for all four modes;
Compute Sanitizer memcheck with zero errors on batched U8 PCA, fused Fresnel,
changed-input graph replay and unsupported-size fallback. The final scaling/graph
changes were checked again with memcheck and the focused FP16 tests.

The final capture is `build/production-f16-final.nsys-rep` (CUDA/NVTX, CUDA graph
nodes traced). Kernel evidence is summarized in `build/production-f16-final-kernels.json`.
Unprofiled rates above come from `build/production-f16-final-mode-*.log`, with
PCA-only runs in `build/production-f16-mode-1.log` and
`build/production-f16-reverse-1.log`. Profiling rates are not used for this table.

Final trace backend checks: 1323 launches each of `fused_f16_scale_partial`,
`fused_f16_scale`, `fused_f16_fft_x`, `fused_f16_fft_y_mean` and
`fused_f16_finish`; 1332 native FP16 PCA projection launches; 1356 Gram/reduction
launches (including setup); 336 FP32 `pca_heev_batched` launches. No
`u8_f32_real_kernel`, legacy cuFFT entrypoints or cuBLAS GEMM kernels occur in
this combined-path capture. The remaining F32 mean and display kernels are expected.
Final focused CTest result: 4/4 passed after the scaling and graph-cache changes.

## F16 spatial filtering

Enabling spatial filtering in the supported PCA/Fresnel/M0 configuration retains F16
storage through Filter2D and into Fresnel. Filter2D uses native cuFFTDx FFTs for
contiguous real F16/F32 tensors with trailing 320x512 dimensions on SM86/SM120.
The native path transposes the real input's half spectrum for contiguous column
access and runs both column FFTs and the mask in one kernel. Per-plane scaling
bounds the forward FFTs, and fixed power-of-two scaling bounds the inverse FFTs.
The inverse row FFT restores these scales and writes the normalized result.

The new Filter2D settings default to `use_fused_f16=false` and `normalize=false`.
The production F16 route enables normalization. Native FFTs are available through
explicit task settings on SM86 and enabled in the production route on SM120. Normalized filtering divides the legacy
output by H*W. Fresnel does not compensate this fixed gain: relative image values
are preserved, but absolute amplitudes differ from the legacy filtered pipeline.
F16 output requires normalization. Ordinary F16 rounding, underflow and output
range limits still apply; there is no frame-dependent normalization or clipping.

Unsupported configurations and native setup failures widen F16 input, perform
FP32 filtering, normalize, and return F16. Existing F32/CF32 settings keep their
legacy behavior. Normalization currently applies only to real input.

The native path is benchmarked against normalized FP32 on three and 64 planes.
On RTX 5090 (SM120), native median times were 19.9 and 97.5 microseconds
for three and 64 planes, versus 30.1 and 158.1 microseconds with normalized
FP32. Native p95 also passed at both sizes. Production uses native FFTs on
SM120 and normalized FP32 computation with F16 storage on unvalidated devices.
Set `enable_f16_filter2d=false` to retain the previous filtered pipeline.
Accuracy tests compare normalized outputs within 1% relative L2 error, including
graph replay after changes to the input and mask, odd plane counts, zero/impulse
inputs and large finite signals. Recording-based validation requires an input recording.
