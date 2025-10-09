# Holoflow Project

The **Holoflow Project** is a modular framework for real-time digital holography and high-performance image processing.  
It is composed of several libraries and an executable application:

- **Holoflow (library)**  
  Core runtime and graph execution engine. Provides the abstraction layer for pipelines as graphs of computational tasks.  
  Handles scheduling, tensor management, memory allocation, and GPU acceleration.

- **Holovibes (executable)**  
  Reference application built on top of Holoflow, used for real-time digital holography. Offers a Qt-based graphical 
  interface for visualization and interaction.  
  Demonstrates the full capabilities of the runtime, including live acquisition and processing.

- **Curaii (library)**  
  Low-level utilities for memory and tensor management. Provides smart pointers, allocation helpers, and RAII abstractions.  
  This library provides C++ apis on top of existing libraries like cufft, cublas, etc.

- **Holofile (library)**  
  Low-level file I/O utilities for reading and writing the custom .holo file format.