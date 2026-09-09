# Holovibes

Holovibes is the reference desktop application built on Holoflow. It combines high-throughput image acquisition, GPU-accelerated holographic reconstruction and analysis, and interactive visualization in a Qt interface.

Its processing pipeline is assembled from registered Holoflow tasks:

- **Sources** acquire or load frames, including [camera acquisition](tasks/sources/ametek_s710_euresys_coaxlink_octo.md) and [Holofile playback](tasks/sources/holofile.md).
- **Synchronous tasks** transform tensors inline, such as [Fresnel diffraction](tasks/syncs/fresnel_diffraction.md), [PCA](tasks/syncs/pca.md), and [FFT shift](tasks/syncs/fft_shift.md).
- **Asynchronous tasks** connect independently scheduled pipeline sections through facilities such as the [batch queue](tasks/asyncs/batch_queue.md) and [sliding average](tasks/asyncs/sliding_average.md).
- **Sinks** present or persist results through the [tensor display](tasks/sinks/display_tensor.md) and [Holofile writer](tasks/sinks/holofile.md).

Use the navigation to browse the complete task catalog and the settings accepted by each task.
