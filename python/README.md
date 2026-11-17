# Holoflow Python interface

The package traces a Python function into the existing native streaming graph.
Source operations return symbolic tensors. Collection sinks return symbolic
result handles; other sinks return `None`. Python runs once
during tracing. Task execution, storage, queues, and CUDA work remain native.

## Development build

Use an x64 Visual Studio developer shell, the repository's Qt/CUDA prerequisites,
and standard 64-bit CPython 3.14 with development headers. Set `CUDA_PATH` to the
toolkit used by CMake's CUDA compiler; mixing toolkit headers is unsupported.

```powershell
cmake --preset msvc-multi -DENABLE_PYTHON=ON -DPython3_EXECUTABLE=C:/Python314/python.exe
cmake --build --preset build-Release
$env:PYTHONPATH = "$PWD/build/msvc-multi/python/Release"
python example/python/simple_doppler.py input.holo output.holo
ctest --preset test-Release
```

Python support defaults to off. Enabling it acquires pinned pybind11 3.1.0 through
the existing FetchContent mechanism. With fetching disabled, supply that exact
pybind11 CMake package. No pip installation or NumPy dependency is required.

## Local installation

```powershell
cmake --install build/msvc-multi --config Release --component python --prefix build/python-install
$env:PYTHONPATH = "$PWD/build/python-install/python"
python -c "import holoflow; print(holoflow.__version__)"
```

The install includes runtime DLLs and PCA runtime-compilation assets; the NVIDIA
driver and Visual C++/OpenMP runtime must be installed on the machine. This is a
local CMake package, not a wheel. The extension is specific to the interpreter ABI
and build configuration. Release is the initial supported configuration.

MathDx 26.10.0 includes CUTLASS 4.8.0 and resolves the CUDA 13.5 private-header
compilation failure. Small-depth PCA uses cuSolverDx, including from an installed
package. The conventional cuSOLVER singleton path is not capturable on CUDA 13.5;
PCA reserves a private zero-matrix scratch slot to use the capturable batched
path. Public tensor shapes and the real input's numerical results are unchanged.

## Tracing and execution

`hf.trace(fn, *args, **kwargs)` executes `fn` in a context-local graph builder.
Native factory inference validates each operation. File sources read metadata
during tracing, but tensor storage and task instances are created at compilation.
`tensor.shape`, `dtype`, `device`, byte `strides`, and `offset` are static metadata.
Tensor value extraction, truth conversion, and array conversion are unsupported.
Ordinary Python side effects run once. Nested traces and cross-trace tensors are
rejected. Call ordinary helper functions to compose a pipeline instead.

`graph.to_json()` returns a dictionary; `graph.to_dot()` returns Graphviz text.
`graph.compile()` creates an executable independently of the trace lifetime.
The first implementation exposes the operations used by the PCA Doppler demo.
Source defaults select the whole file and loop over complete acquisition batches;
use `loop=False` for finite playback with downstream draining.

Every operation accepts an optional `node_id`. Explicit IDs are unique and used
unchanged in graph inspection, metrics, and events; omitted IDs are generated.

Executable methods are generic: `start`, `stop`, `wait`, `close`, `metrics`,
`send_event`, and `receive_event`. Use a context manager for deterministic cleanup.
Compile a fresh executable to run again. Concurrent lifecycle mutations on one
executable are rejected; another thread can request `stop` while `wait` is blocked.

`send_event(node_id, data)` accepts a JSON object and returns whether the outgoing
queue accepted it, not whether the task handled it. Unknown IDs raise `ValueError`.
`receive_event()` is nonblocking and returns `Event(node_id, data)` or `None`.
Node event protocols are unchanged, including recording commands and completion.
Only send commands supported by the destination task.

Caught native worker errors surface as `hf.ExecutionError` from event polling or
`wait`; native assertions and unrecoverable driver/process failures remain fatal.
Polling and timeout handling belong in ordinary Python, as shown by the demo.

## Logged Fresnel Doppler example

`example/python/fresnel_doppler.py` reproduces the logged PCA/queue processing
with a device void sink. Both variants use 852 nm, 492 mm, and 20 µm pixels:

```powershell
python example/python/fresnel_doppler_decomposed.py
python example/python/fresnel_doppler.py
```

Set `INPUT_PATH` and `RUN_SECONDS` at the top of either file before running.
The source loops; the runner stops after that duration or Ctrl+C.
The void sink discards processed frames without copying or writing them.

Each example keeps processing parameters at the top of the file. The decomposed
example calls an ordinary Python `fresnel_diffraction` helper inside `pipeline`;
`hf.trace(pipeline)` records the helper operations into the same graph.


## Finite streams and NumPy results

NumPy is needed only when retrieving arrays or running the collection tests.
It is already available in this development environment; install it in other
Python environments before using `result()`.

```python
import holoflow as hf


def pipeline(path):
    frames = hf.sources.holofile(path, batch_size=32, loop=False)
    values = frames.to("device").astype(hf.float32)
    return hf.sinks.numpy(values, axis=0)


with hf.trace(pipeline, "input.holo").compile() as run:
    run.start()
    images = run.result()  # ndarray: (total_frames, height, width)
# images retains its allocation after run closes.
```

File playback still loops by default. Finite sources close after their selected
`[start_frame, end_frame)` range. EOF is an ordered runtime boundary: queues drain
all valid outputs before forwarding closure. Execution completes automatically
when all sections have drained, including unreturned side branches. `result()`
waits for that completion with the GIL released; `wait()` still returns `None`.
A failure raises `ExecutionError`; an explicit stop before completion makes
`result()` raise `CancelledError`. No partial arrays are returned.

Return a collector handle, or a tuple/list/dictionary with string keys containing
handles (and optional `None`), from the traced function. `result()` reconstructs
that structure with NumPy arrays. Returning symbolic tensors or arbitrary Python
values is an error. Repeated calls reuse the same array objects.

`hf.sinks.numpy(x, axis=0, length=None)` concatenates emissions along `axis`,
including negative axes. Other dimensions and dtype are preserved. `length` is
the final array's size along that axis, not the number of emissions. When omitted,
it is inferred from source and operator counts; if inference is unavailable,
provide it explicitly. Too much data fails immediately; too little fails at EOF.
Filling the allocation does not itself terminate the stream.

The collector allocates one contiguous final host buffer. Private contiguous host
producers can write into its slices directly. Device inputs are transferred to
host; shared producers and strided layouts copy into final slices. Conversion to
NumPy does not copy the completed buffer. Zero outputs produce a typed empty array.

Source and batch-queue incomplete tails default to `tail="error"`. Explicit
`tail="drop"` keeps only complete source batches or queue stride groups. Queue
stride skipping is intentional and does not count as an incomplete tail.
Sliding averages produce only complete windows: N accepted inputs with width W
produce `max(0, N-W+1)` outputs, after configured initial discards. They never pad
or manufacture final outputs. Multi-input operations pair changing streams in
order and reject unmatched data; constants remain reusable.

V1 collection requires finite changing sources throughout the graph and fixed
tensor descriptors. Finite execution supports the Python math/processing
operators, batch queues, sliding averages, dual-reader queues, void sinks, and
NumPy collectors. Operators without a finite closure contract (including the
command-driven HolofileWriter) are rejected before compilation. Existing looping
recording examples retain their explicit recording-event workflow.

See `example/python/finite_numpy.py` for a runnable entry point.


## Finite Fresnel Doppler viewer

With NumPy and Matplotlib available in Python 3.14, run from the repository root:

```powershell
$env:PYTHONPATH = "$PWD/build/msvc-multi/python/Release"
& C:\Python314\python.exe example/python/fresnel_numpy.py "D:\260312_GUJ_1.holo"
```

The default requests 1,024 scaled uint8 output frames, reading the first 34,784
holograms on demand. Use `--output-frames 2` for a short run, or choose
`--start-frame 320 --end-frame 35104` for an explicit exclusive-end range.
`--end-frame` and `--output-frames` are mutually exclusive. Range lengths must
be divisible by 32. Each output averages 64 complete 32-frame processing windows.
The viewer opens after processing and executor cleanup. Drag its frame slider or
use left/right arrow keys. A single frame needs no slider; empty results are
reported without opening a window. Import `collect_frames` to obtain the array
without displaying it, or `create_viewer` to construct a figure separately.


### Spatial filtering and flatfield

`hf.filter2d(x, r_inner, r_outer, s_inner=0, s_outer=1)` applies the native radial
spatial-frequency filter. Radii and smoothing widths are frequency bins. It
accepts contiguous float32 or complex64 device tensors and preserves native FFT
normalization. The current native complex mask stores its coefficient in both
real and imaginary components (an all-pass mask therefore also multiplies by
`1 + 1j`); the Python wrapper preserves this behavior. `hf.flatfield(x, sigma_y, sigma_x)` subtracts a Gaussian background
from contiguous float32 device tensors; sigmas are in pixels. Both accept
`node_id=` and preserve finite-stream emission counts.

The standalone Fresnel viewer always applies filter2d after PCA and flatfield
after fftshift, before temporal averaging. Its constants use UI numeric defaults:
filter radii 0/1024, smoothing widths 0/1, and a 2000 micrometer flatfield cutoff.
The physical cutoff is converted explicitly inside `pipeline()` using propagated
sampling pitches and `sigma_px = 0.187 * cutoff_m / pitch_m`. Edit those constants
to change processing strength. Neither operation changes the requested frame count.


## Aberration-corrected Fresnel dashboard

```powershell
$env:PYTHONPATH = "$PWD/build/msvc-multi/python/Release"
& C:\Python314\python.exe example/python/fresnel_aberration_numpy.py "D:\260312_GUJ_1.holo" --output-frames 2
```

This separate standalone example keeps every processing stage inside `pipeline()`.
It follows the supplied aberration graph: four-window PCA batches, filter radius 3,
a current/delayed dual reader, short-time Fresnel reconstruction, causal sensor
averaging, full-pairwise Shack-Hartmann slopes, Noll modes 4/5/6, phase correction,
and explicit full-frame Fresnel propagation at 851 nm and 0.49 m. Geometry follows
a 5-by-5 subaperture grid (64-by-102 patches for 320-by-512 input).

Qt display sinks are replaced with four NumPy collectors. `collect_frames()` returns
`xy` (uint8), `shack_hartmann` (uint8 mosaic), `coefficients` (float32 radians), and
`phase` (float32 wrapped phase in radians). The coefficient history panel replaces
the separate current-coefficient/history displays. The wall-clock-throttled
ZernikeDefocusZProp logging sink is omitted; it does not feed the correction graph.
The native analysis operators are also exported through `holoflow`; symbolic tensor
indexing supports integer indices and slices, and `hf.transpose`, `hf.copy`,
`hf.normalize`, and `hf.wrap2pi` expose the corresponding native operations.

Unlike the looping CPU-cached target, this example reads on demand and completes
at EOF. It preserves both the 63-sample initial discard and the 64-sample final
average: T temporal windows produce `max(0, T-126)` XY frames, while all three
diagnostic streams retain T updates, including startup samples. Four-window PCA
batches require input lengths divisible by 128. Count requests round the input
up to complete batches and explicitly trim only the final XY array after collection.
The default therefore reads 36,864 holograms, computes 1,026 XY frames, and returns
the first 1,024. `--end-frame` instead returns all XY frames from the explicit range.

The dashboard's XY frame i selects diagnostic update i+126, the end of that XY
averaging window. This labels processing updates, not identical physical timestamps;
the corrected field uses the dual reader's native 31-update delay. The coefficient
plot excludes the first 63 startup samples from its curves and autoscaling,
while retaining the original update indices. Arrow keys and the slider navigate all
panels together. Startup data and diagnostic tail updates remain available in the
returned arrays; the example neither pads nor synthesizes outputs at EOF.
