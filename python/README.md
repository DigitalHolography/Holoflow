# Holoflow Python interface

The package traces a Python function into the existing native streaming graph.
Source operations return symbolic tensors; sinks return `None`. Python runs once
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
there is no new finite-stream or downstream-draining behavior.

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
