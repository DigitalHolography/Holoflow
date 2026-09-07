# Holoflow

## What is Holoflow?

Holoflow is a C++/CUDA library for building high-throughput scientific processing pipelines. An application describes its computation as a graph
whose nodes are computational tasks and edges tensor-dependencies between them.

The library separates the description of a pipeline from its execution. A `GraphSpec` records what should run ("the equations"), the compiler validates that graph and prepares its tensors, resources, and tasks. Finally, the scheduler executes the compiled graph on CPU and GPU resources.

Applications provide the concrete tasks. Holoflow provides the graph model, compilation, memory management, scheduling, and inspection tools needed to connect those tasks into a running pipeline.

## Why Holoflow?

Scientific imaging pipelines combine high data rates, demanding numerical processing, and algorithms that change quickly as experiments evolve. Implementing every pipeline directly as buffer-management and scheduling code makes those changes difficult and requires substantial high-performance-computing expertise.

Many scientists use NumPy[^numpy2020] or GPU-accelerated alternatives such as CuPy,[^cupy2017] JAX,[^jax2018] and PyTorch[^pytorch2019] because they hide implementation details and keep the code close to the underlying equations. However, a focused benchmark of a representative micro-batch laser Doppler holography pipeline found that an optimized C++/CUDA implementation was up to 74–267% faster than the Python-based implementations, depending on the platform.[^guillou2027]

Holoflow aims to combine those two needs: a high-level, declarative description of the computation and predictable execution suitable for real-time workloads, with first class support for Windows.

## Example: a GPU band-pass filter

The following pipeline loads a two-dimensional NumPy array, transfers it to the GPU, applies a frequency-domain band-pass filter, and saves the result.

### Build the graph

Each node has a unique name, a registered task kind, and task-specific JSON settings. Each edge identifies the producer's output slot and the consumer's input slot.

```cpp
#include <boost/graph/adjacency_list.hpp>
#include <holoflow/core/graph_spec.hh>

using holoflow::core::EdgeSpec;
using holoflow::core::GraphSpec;
using holoflow::core::NodeSpec;

GraphSpec graph;

auto load     = boost::add_vertex(NodeSpec{"load", "LoadNpy", {{"path", "input.npy"}}}, graph);
auto upload   = boost::add_vertex(NodeSpec{"upload", "Memcpy", {{"target", "Device"}}}, graph);
auto fft      = boost::add_vertex(NodeSpec{"fft", "RFFT2", {{"axes", {-2, -1}}}}, graph);
auto mask     = boost::add_vertex(NodeSpec{"mask", "BandpassMask",{{"shape", {512, 257}}, {"inner_radius", 0.05}, {"outer_radius", 0.25}}}, graph);
auto multiply = boost::add_vertex(NodeSpec{"multiply", "Multiply", nlohmann::json::object()}, graph);
auto ifft     = boost::add_vertex(NodeSpec{"ifft", "IRFFT2", {{"axes", {-2, -1}}}}, graph);
auto download = boost::add_vertex(NodeSpec{"download", "Memcpy", {{"target", "Host"}}}, graph);
auto save     = boost::add_vertex(NodeSpec{"save", "SaveNpy", {{"path", "filtered.npy"}}}, graph);

boost::add_edge(load, upload, EdgeSpec{0, 0}, graph);
boost::add_edge(upload, fft, EdgeSpec{0, 0}, graph);
boost::add_edge(fft, multiply, EdgeSpec{0, 0}, graph);
boost::add_edge(mask, multiply, EdgeSpec{0, 1}, graph);
boost::add_edge(multiply, ifft, EdgeSpec{0, 0}, graph);
boost::add_edge(ifft, download, EdgeSpec{0, 0}, graph);
boost::add_edge(download, save, EdgeSpec{0, 0}, graph);
```

[![Graph specification for the band-pass filter](../assets/images/holoflow-bandpass-spec.svg)](../assets/images/holoflow-bandpass-spec.svg)

*The graph specification describes task configuration and slot-to-slot dependencies.*

### Compile the graph

The registry connects each task kind to the factory that can infer its tensor contract and create its runtime implementation. Once the application has registered those factories, the compiler turns the specification into an executable graph.

```cpp
#include <holoflow/core/registry.hh>
#include <holoflow/runtime/compiler.hh>

holoflow::core::Registry registry;
register_tasks(registry); // Register the application and processing task factories.

holoflow::runtime::Compiler compiler(registry);
auto compiled = compiler.compile(graph);
```

[![Compiled band-pass filter graph](../assets/images/holoflow-bandpass-compiled.svg)](../assets/images/holoflow-bandpass-compiled.svg)

*The compiled graph includes inferred tensor types and memory locations and groups the tasks into an execution section.*

### Start and stop the graph

Construct a scheduler from the compiled graph and start it. `start()` returns after launching the runtime threads, so the application can continue with its own event or monitoring loop.

```cpp
#include <holoflow/runtime/graph_exec.hh>

holoflow::runtime::Scheduler scheduler(
    compiled->graph,
    compiled->sections,
    compiled->resources);

scheduler.start();

// Application work continues here.

scheduler.request_stop();
scheduler.wait();
```

`request_stop()` asks the tasks to stop cooperatively. `wait()` joins the runtime threads and is the synchronization point for a clean shutdown. Because the scheduler borrows the compiled graph and resources, keep `compiled` alive until after `scheduler.wait()` returns.

[^numpy2020]: C. R. Harris, J. Millman, S. J. van der Walt, et al., “[Array programming with NumPy](https://doi.org/10.1038/s41586-020-2649-2),” *Nature*, vol. 585, no. 7825, pp. 357–362, 2020. See also the [site-wide reference](../references.md#harris-2020).

[^cupy2017]: R. Okuta, Y. Unno, D. Nishino, S. Hido, and C. Loomis, “[CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations](https://github.com/cupy/cupy#reference),” in *Proceedings of the Workshop on Machine Learning Systems at NIPS 2017*, 2017. See also the [site-wide reference](../references.md#okuta-2017).

[^jax2018]: J. Bradbury, R. Frostig, P. Hawkins, et al., “[JAX: composable transformations of Python+NumPy programs](https://github.com/jax-ml/jax#citing-jax),” software, 2018. See also the [site-wide reference](../references.md#bradbury-2018).

[^pytorch2019]: A. Paszke, S. Gross, F. Massa, et al., “[PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://papers.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html),” in *Advances in Neural Information Processing Systems 32*, pp. 8024–8035, 2019. See also the [site-wide reference](../references.md#paszke-2019).

[^guillou2027]: J. Guillou, J. Fabrizio, E. Carlinet, and M. Atlan, “Real-Time Scientific Computing in Python: The Cost of High-Level GPU Abstractions,” unpublished manuscript, 2027. See also the [site-wide reference](../references.md#guillou-2027).
