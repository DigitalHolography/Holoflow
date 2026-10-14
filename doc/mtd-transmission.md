# MTD transmission reconstruction

Holotask provides an offline MTD reconstruction graph that runs in the native
Holoflow compiler and scheduler. It consumes a complete example scan once and
exports a complex refractive-index volume without reducing it to a display image.

## Run the example dataset

Build with the normal repository workflow:

```powershell
cmake --preset msvc-multi
cmake --build --preset build-Release
```

Run the graph (from the repository directory):

```powershell
.\build\msvc-multi\Release\holotask_mtd.exe `
  C:\Users\Novosibirsk\Documents\data_mtd\ACQUIS `
  .\build\mtd\indiceGPS.npy `
  --slice .\build\mtd\slice280.npy --z 280 `
  --graph .\build\mtd\pipeline.json
```

The output includes `indiceGPS.npy.json` with optical settings, volume axes,
voxel pitch in metres, and the mounting-index assumption. Existing outputs are
protected unless `--overwrite` is given. The command accepts `--medium-index n`
when the actual mounting-medium refractive index is known. The sample lacks NM,
so the default uses N0, records that assumption, and prints a notice.

The dataset folder must contain `i*.pgm`, `Image_mask.pgm`,
`config_manip.txt_SAV`, and `recon.txt_SAV`. Saved configuration files take
precedence; active text copies are a fallback. NB_HOLO specifies the complete scan
count, matching the notebook's use of all 600 images rather than the historical
NB_HOLO_RECON partial-scan setting.

Use NumPy to inspect the output:

```python
import numpy as np
volume = np.load("build/mtd/indiceGPS.npy", mmap_mode="r")
xy_slice = volume[280].real
maximum_projection = volume.real.max(axis=0)
```

## Nodes and tensor contracts

| Node | Inputs | Outputs |
| --- | --- | --- |
| PgmSequence | directory, prefix, batch size | Host U8/U16 `[frames,y,x]` |
| MTDFieldRecovery | Host/device U8/U16/F32 holograms | Device CF32 fields and F32 `[frames,2]` carrier coordinates `(x,y)` |
| RytovCorrection | Device fields and host U8 background mask | Device CF32 corrected log-amplitude plus corrected phase |
| MTDLinear3DReconstruction | Rytov fields and carriers | Device CF32 `[z,y,x]` volume and F32 centered Fourier coverage counts |
| GPSRefinement3D | Linear volume and coverage | Device CF32 constrained GPS volume |
| VolumeSlice | Device volume, axis/index/component settings | Device F32 plane for existing display sinks |
| NpyVolumeWriter | Contiguous host/device F32/CF32 plane or volume | NPY file and JSON metadata |

`VolumeSlice` uses axis 0 for XY, axis 1 for XZ, and axis 2 for YZ. Components
are real, imaginary, and magnitude. Carrier coordinates use F32 because the
framework's dtype set has no signed integer type; these small integer coordinates
are exactly represented. Coverage counts are exact for the example scan.

The existing `PgmSequence` defaults to looping for playback compatibility. With
`repeat=false`, the source emits complete batches once and then reports Eof. A
finite scan requires batch_size to divide the frame count. The MTD graph requests
the complete scan in one batch and requires exactly NB_HOLO matching frames and
one matching mask. Extra files are rejected rather than silently dropping images.
Volume, slice, graph export, and metadata output paths must be distinct.

## Library and application integration

`holotask/pipelines/mtd.hh` exposes `load_mtd_dataset`, `build_mtd_pipeline`, and
`register_mtd_tasks`. The application registry registers all new task kinds;
typed adapters are available in `GraphBuilderTasks`. The standalone runner is the
entry point for the assembled offline pipeline; the existing camera/import UI
continues to build its configured holographic processing graph.

To show a reconstructed volume in an application graph, connect `VolumeSlice`
to an existing `DisplayTensor` sink. Selecting a different index selects a
different plane. Music controls from the exploratory notebook are not part of
the scientific reconstruction nodes.

## Numerical behavior and optimization

Field recovery follows the notebook's Tukey windows, Fourier-order crop,
first-maximum carrier detection, recentering, and unnormalized inverse transform.
Linear assembly follows its Ewald mapping, averaging, FFT centering, and physical
scaling. GPS follows its real/imaginary constraints, measured-frequency
replacement, inverse normalization, iteration count, and final constraints.

Phase unwrapping uses the recovered CPU MTD implementation's mirrored spectral
phase-gradient integration with configurable global/local damping and border
suppression. This differs from the notebook's scikit-image approximation. The
polynomial background uses normalized coordinates and a double-precision,
reorthogonalized QR projection. A source PNG maximum projection cannot establish
an exact match to the original 3D MTD volume.

cuFFT plans and per-frame work buffers are allocated once per task. Host input
frames are transferred individually; the complete raw dataset is not copied to
GPU memory. Background projection is prepared once per batch and applied on the
GPU. Ewald accumulation executes illuminations in sequence. Rytov and GPS preserve
their input tensors so branched graphs can also export intermediate results.
GPS transforms its output workspace in place, and FFT centering swaps disjoint voxel
pairs without a second shift buffer. Export uses an 8 MiB pinned transfer buffer.
Cancellation is checked between frames, GPS iterations, and export chunks.

## Validation

`mtd_test.cc` compares the nodes against independent NumPy references, including
mirrored Volkov integration, all three slice axes, GPS on dimensions divisible
and not divisible by four, and complete compiler/scheduler execution. The tests
also cover crop validation, empty masks, invalid bounds, source publication/EOF,
cancellation, and lossless NPY interoperability.
Branched exports, exact scan counts, colliding output paths, and restoration of
previous output files after a failed overwrite are covered by regression tests.

```powershell
ctest --preset test-Release -R MTD
ctest --preset test-Release
```

The native runner was exercised on the supplied 600-frame scan with a 512-cubed
complex64 output and Z=280 slice on a TITAN RTX. A warmed run, including compilation
and file export, took 3.75 seconds. All volume values were finite, and the exported
F32 slice exactly equaled `volume[280].real`. This measures execution and storage
correctness; it does not establish an exact match to the original MTD reconstruction.
The final Release build and all 469 tests in the repository test preset passed.
All added C++/CUDA code passed the repository clang-format configuration.
