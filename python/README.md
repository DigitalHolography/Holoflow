# Holoflow Python interface

The package traces a Python function into the existing native streaming graph.
Source operations return symbolic tensors. Collection sinks return symbolic
result handles; other sinks return `None`. Python runs once
during tracing. Task execution, storage, queues, and CUDA work remain native.

## Development build

For the uv workflow, benchmark API, and graph/Perfetto outputs, see
[Python development and performance](PERFORMANCE.md).

Use an x64 Visual Studio developer shell, the repository's Qt/CUDA prerequisites,
and standard 64-bit CPython 3.14 with development headers. Set `CUDA_PATH` to the
toolkit used by CMake's CUDA compiler; mixing toolkit headers is unsupported.

```powershell
cmake --preset msvc-multi -DENABLE_PYTHON=ON -DPython3_EXECUTABLE=C:/Python314/python.exe
cmake --build --preset build-Release
$env:PYTHONPATH = "$PWD/build/msvc-multi/python/Release"
uv run python tools/setup_python.py
uv run hf-demo --help
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
command-driven HolofileWriter) are rejected before compilation. Looping
recording pipelines use the explicit recording-event workflow.

### Spatial filtering and flatfield

`hf.filter2d(x, r_inner, r_outer, s_inner=0, s_outer=1)` applies the native radial
spatial-frequency filter. Radii and smoothing widths are frequency bins. It
accepts contiguous float32 or complex64 device tensors and preserves native FFT
normalization. The current native complex mask stores its coefficient in both
real and imaginary components (an all-pass mask therefore also multiplies by
`1 + 1j`); the Python wrapper preserves this behavior. `hf.flatfield(x, sigma_y, sigma_x)` subtracts a Gaussian background
from contiguous float32 device tensors; sigmas are in pixels. Both accept
`node_id=` and preserve finite-stream emission counts.

See [YAML processing demo](PERFORMANCE.md#yaml-processing-demo) for the runnable Python demo.
