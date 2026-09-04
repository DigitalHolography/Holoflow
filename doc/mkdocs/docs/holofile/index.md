# Holofile

Holofile is the file I/O library for the `.holo` format used by high-throughput holographic acquisitions. It provides the reader and writer primitives used to move recorded frame data and associated metadata between storage and Holoflow pipelines.

Holovibes integrates the library through two tasks:

- The [Holofile source](../holovibes/tasks/sources/holofile.md) reads recorded frames into a processing graph.
- The [Holofile sink](../holovibes/tasks/sinks/holofile.md) records pipeline tensors and settings to disk.

The public C++ interface is available in [`src/holofile/include/holofile`](https://github.com/DigitalHolography/Holoflow/tree/main/src/holofile/include/holofile).
