# Python development and performance

Use an x64 Visual Studio developer shell with the existing Qt/CUDA prerequisites.
uv manages Python and dependencies; CMake builds the native extension.

```powershell
uv sync --group examples
cmake --preset msvc-multi -DENABLE_PYTHON=ON -DPython3_EXECUTABLE="$PWD/.venv/Scripts/python.exe"
cmake --build --preset build-Release
uv run python tools/setup_python.py
uv run hf-demo --help
ctest --preset test-Release
```

The setup helper registers the CMake Release package with a `.pth` file in this
environment. Subsequent scripts need no `PYTHONPATH`. Repeat setup after moving
the build or recreating the environment. Rebuild native changes with CMake.
The examples dependency group supplies NumPy, Matplotlib, and PyYAML. Base installation
has no Python runtime dependencies. Python 3.14 and Release are supported.

## YAML processing demo

Copy `example/python/holography_demo.yaml` and set `input.path` to a Holofile.
Relative paths are resolved next to the config. Install the examples group
and register the launcher with the setup helper shown above.

```powershell
uv run hf-demo config.yaml
```

The processing order is temporal PCA (or one image per raw frame), optional
Shack–Hartmann aberration correction, then optional Fresnel propagation.
The upload queue assembles complete temporal windows, and PCA processes each
window directly. The propagation and host output handoffs use independent queues.
For the optimized basic pipeline, set `time.precision: fp16`. Use looping,
CPU-cached U8 input at 320�512, 32-frame input and PCA windows, at most 16
selected components, and native Fresnel. Disable spatial filtering and aberration
correction. This mode uses the pipelined PCA queue, FP16 projection, and fused
Fresnel magnitude reduction. It keeps the headless output on the GPU, so its
latency ends at GPU output completion. The default `time.precision: fp32` keeps
the existing host output path.

Older configs can still include `time.batch_windows` and
`queues.preprocessing_windows`. The demo ignores those settings.
Tune queue capacities in `queues`. The output latency probe runs after the
device-to-host transfer and host queue, before NumPy collection or Void.
Set `input.mode` to `finite` for EOF or `looping` for continuous playback.
Use `input.duration_seconds` to stop a looping run automatically. The `drop`
tail policy trims incomplete source batches and temporal windows.

Set `postprocessing.average_window` to average consecutive processed images
and, when CAO is enabled, use the same window for its sensor average. The
default of one keeps each image. The first averaged image is emitted after
a full output window is available. Latency uses the complete contributing
frame range, including both the aberration sensor average and the output
sliding average. The live display shows input frames/s, output images/s, and
oldest and newest frame latency. Set `viewer.enabled: true` for a finite
Matplotlib preview. Use `input.end_frame` to limit the range to
`viewer.max_frames` output images. The metrics include collecting the preview,
and the viewer opens after the final summary.

All demo configs produce scaled 8-bit XY output. Configure wavelength, input
pixel pitch, and the shared propagation distance under `space`. The optional
`spatial_filter` section controls filtering after temporal processing, before
CAO or Fresnel. Set `spatial_filter.enabled: true` to apply it.
`postprocessing.flatfield_cutoff_m` supplies the physical cutoff for both the
CAO sensor and the final XY image. Each stage converts it using its own image
sampling pitch.

The [Basic config](../example/python/holography_basic.yaml) runs PCA and Fresnel.
The [CAO config](../example/python/holography_cao.yaml) adds spatial filtering,
flatfield and percentile clipping, and three separate
coefficient history plots. With `viewer.cao_coefficients_only: true`, the demo
collects coefficient vectors and plots a4, a5, and a6 after the sensor window
fills. XY processing and metrics still run, with the images sent to Void.
The CAO example enables `viewer.cao_compare_modes` to overlay single-reference
(grey) and full-pairwise Laplacian (black) histories. The two runs start from
the same input range. The second updates the first native compilation and
reuses compatible resources. Set `cao_compare_modes: false` and choose
`aberration.slope_mode` for a single run.
To see the full dashboard with XY, the Shack–Hartmann mosaic, and wrapped
phase, disable comparison and coefficients-only mode and set `cao_diagnostics: true`.
The preview limit also applies to the diagnostic updates.
Device queues hand off CAO diagnostics before copying them to host, so those
transfers run in separate sections from the correction CUDA graph.

Set `graph.enabled: true` to save the native compiled DOT and a local HTML
report with a **Copy DOT** button. The demo prints both paths after the run.
Set `graph.open_browser: false` to save without opening a browser. If Graphviz
`dot` is on PATH, the report also displays the compiled graph. The report
uses the compiler's execution plan and does not require Perfetto.

## Benchmark your pipeline

```python
import holoflow as hf

def pipeline(path):
    frames = hf.sources.holofile(path, loop=False, node_id="input")
    return hf.sinks.numpy(frames.to("device"))

report = hf.benchmark(pipeline, args=("input.holo",), warmup_seconds=0.5,
                      log_dir="logs/python")
print(report.input_frames_per_second)
print(report.makespan_seconds, report.timings)
images = report.results
print(report.artifacts.directory)
```

`input_frames_per_second` counts successful emissions at the selected Holofile
source, multiplied by its batch size, divided by native monotonic elapsed time.
It measures source progress under downstream backpressure, including data admitted
to queues. Use long windows relative to queue capacity for representative steady
throughput. Warmup is excluded. Retries, EOF, failed emissions and dropped source
tails are excluded. With multiple sources, select `source_node_id` explicitly;
rates are not summed. This first API supports Holofile sources with fixed batches.

Finite mode finishes at EOF and returns collected results. Its steady window
ends after completion and draining; for small datasets, tail draining can dominate.
If it finishes during warmup, throughput is `None`, rather than a fabricated
steady-state estimate. Choose a sufficiently large input range.

Looping mode requires looping sources and uses `duration_seconds` after warmup,
then stops and drains. It returns no collected results. All Holofile sources must
match the selected mode. Command-driven recording pipelines still need their own
event orchestration; the benchmark does not issue recording commands.

Makespan starts immediately before tracing and ends after `result()` (finite) or
`wait()` (looping). It includes Python graph construction, native compilation,
startup, processing, and CUDA draining. It excludes environment startup, module
imports, optional capture initialization, executor destruction, and final artifact
export. Looping makespan describes a timed session, not a fixed workload.
`timings` separates tracing, compilation, startup, execution, cleanup and artifact
finalization; looping shutdown is a subset of execution. Diagnostics add overhead;
compare runs with the same diagnostic settings. Descriptor serialization and
diagnostic submissions during compilation remain part of the measured work.

## Live throughput in a running pipeline

Insert `frames = hf.throughput(frames, axis=0, interval_ms=1000, node_id="input_fps")`
where you want to measure frame rate. The node passes the tensor through without
copying it. Each successful batch counts the size of the selected axis. Use
`axis=None` to count one tensor per batch.

Read `throughput` events with `run.receive_event()`. Each event has `frames`,
`interval_seconds`, `fps`, and `total_frames`. The first window can include
startup. Each node keeps its latest unread update, so poll regularly to display
live FPS. After `run.wait()` or `run.result()`, read the `throughput_summary`
event for `total_frames`, `elapsed_seconds`, and `completed`. The elapsed time
covers startup through wait and CUDA drain. A completed finite run also reports
`average_fps`. CUDA graph counts reflect successfully submitted work.

## Live source-to-output latency

Place `hf.latency_start(frames, node_id="input")` directly after Holofile.
Pass the result tensor through `hf.latency_end(..., source_node_id="input",
node_id="output_latency", first_frame=0, window_frames=32, stride_frames=32,
interval_ms=500)` immediately before `hf.sinks.void()`. A NumPy sink also works
when a host `BatchQueue` publishes the completed transfer upstream of the
latency probe. Use one pair per graph.
Each end-probe execution corresponds to one logical output. Output index `j`
uses the frame range beginning at `first_frame + j * stride_frames` and
spanning `window_frames`. Sequences continue across file wraps. Supply the
actual mapping for your graph, including warmup offsets. Dynamic drops within
the stream require provenance that this explicit mapping cannot infer.

Read `latency` events from `run.receive_event()` for interval statistics, and
`latency_summary` after `wait()` or `result()` for whole-run statistics.
`run.latency("output_latency")` reads the cumulative snapshot. Statistics
include matched and unmatched counts and the latest, min, mean, and max
milliseconds since the oldest and newest contributing input frames entered
the graph. All frames within a source batch share an ingress timestamp.
With a host queue, the probe measures readiness when it sees the published
buffer. Device-side probes measure readiness after the section's existing
CUDA stream barrier. Input timestamps outside the retained `history_batches` (8,192 by
default) count as unmatched.

## Graphs, compiler logs and Perfetto

For manual execution, use `graph.compile(log_dir="logs/python")` and retain
`run.artifacts`. A unique folder is created for each compile. After the executable
context closes, files are finalized and the paths remain usable:

- `descriptor_json` / `descriptor_dot`: the original traced graph.
- `compiled_dot`: the native compiled graph (`compilation_success.dot`).
- `failure_dot`: the native failure graph, when compilation reached that stage.
- `compiler_log`: native compiler messages.
- `perfetto`: compilation and runtime trace; benchmarks also trace Python graph construction.
- `report`: benchmark measurements as JSON, excluding collected arrays.
- `files()`: existing files, including per-section CUDA graph diagnostics.

Open DOT files with a Graphviz-compatible viewer. Load `.perfetto-trace` files in
[Perfetto UI](https://ui.perfetto.dev/). Native tracing shows host lifecycle and
compiler/scheduler scopes; it is not a CUDA kernel execution timeline.

Diagnostics are off by default. Perfetto requires `ENABLE_PERFETTO=ON`; otherwise
graph/log outputs still work and `report.perfetto_available` is false. Only one
process-wide capture may run at a time; overlapping explicit captures raise an
error. Explicit close reports export errors. Failures preserve available
diagnostics; compilation and benchmark exceptions include the run-directory path.
Native diagnostic background write failures retain the existing warning behavior.

## CUDA runtime compilation cache

Holoflow caches linked PCA CUBINs and fused FP16 Fresnel PTX across launches.
On Windows the default directory is `%LOCALAPPDATA%/Holoflow/NvrtcCache/v1`.
Set `HOLOFLOW_NVRTC_CACHE_DIR` to use another directory or
`HOLOFLOW_NVRTC_CACHE_DISABLE=1` to bypass the disk cache. The key follows the
embedded kernel source, compile options, and CUDA and MathDx toolchain identity.
Use the bypass if you edit external toolkit headers in place. Perfetto shows
disk lookup, hits, and writes alongside compilation and module loading.

## Windows CPU-cached Holofile preload

Windows CPU-cached Holofile input overlaps pinned-memory registration with file
reading. The default chunk target is 64 MiB. Holoflow aligns chunks to output
batches and system pages so each batch stays inside one CUDA registration.
Set `HOLOFLOW_HOLOFILE_PRELOAD_CHUNK_MIB` to another target or set it to `0`
to compare with the sequential preload.
