# Curaii

Curaii is the low-level C++ support library for GPU resources used throughout Holoflow. It provides RAII-based ownership and error handling around CUDA runtime facilities and companion libraries so higher-level processing code can manage resources safely.

The library includes wrappers and utilities for:

- CUDA devices, streams, events, and memory
- cuBLAS linear algebra operations
- cuFFT transform plans and execution
- cuSOLVER numerical routines
- NVRTC runtime compilation

Curaii is an infrastructure library rather than an executable application. Its public headers are located in [`src/curaii/include/curaii`](https://github.com/DigitalHolography/Holoflow/tree/main/src/curaii/include/curaii).
