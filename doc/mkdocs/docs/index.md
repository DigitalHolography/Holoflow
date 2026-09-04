# Holoflow

Holoflow is a graph-based high-performance computing framework for real-time scientific imaging, with first-class support for GPU-accelerated processing and laser Doppler holography.

![Holoflow processing graph from Holofile input through GPU processing tasks to the displayed output](assets/images/holoflow-pipeline.svg){ loading=lazy }
/// caption
Example Holoflow processing graph
///

<div class="grid" markdown>

![Black placeholder for the input image](assets/images/input-placeholder.svg){ width=512 loading=lazy }
/// caption
Input
///

![Black placeholder for the output image](assets/images/output-placeholder.svg){ width=512 loading=lazy }
/// caption
Output
///

</div>

## From processing graph to real-time execution

Describe your processing pipelined as a declarative graph of computational tasks. No implementation details, no manual buffer management, just the maths.
Holoflow takes care of compiling and executing it efficiently!

The runtime manages tasks instantiation, GPU scheduling, memory, tensor lifetimes, synchronization, and data movement, allowing processing code to remain focused on the algorithms themselves.
Pipelines can combine acquisition, signal processing, reconstruction, analysis, and visualization while sustaining the high data rates required by modern scientific imaging systems.

Holoflow was developed for demanding digital holography workloads, but its execution model is designed for general-purpose scientific computing.

## Built as a modular stack

<div class="grid cards" markdown>

-   **Holoflow**

    Core graph runtime for task scheduling, tensor management, memory allocation, and GPU execution.

    [Explore Holoflow](holoflow/index.md)

-   **Holovibes**

    Interactive Qt application for real-time acquisition, holographic reconstruction, analysis, and visualization.

    [Explore Holovibes](holovibes/index.md)

-   **Curaii**

    RAII-based GPU, memory, tensor, and CUDA library abstractions used throughout the stack.

    [Explore Curaii](curaii/index.md)

-   **Holofile**

    Efficient reading and writing of the `.holo` format for high-throughput holographic acquisitions.

    [Explore Holofile](holofile/index.md)

</div>
