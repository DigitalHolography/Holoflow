# Holovibes

Holovibes is the reference desktop application built on Holoflow. It combines high-throughput image acquisition, GPU-accelerated holographic reconstruction and analysis, and interactive visualization in a Qt interface.

Its processing pipeline is assembled from registered Holoflow tasks:

- **Sources** acquire or load frames, including [camera acquisition](tasks/sources/ametek_s710_euresys_coaxlink_octo.md) and [Holofile playback](tasks/sources/holofile.md).
- **Synchronous tasks** transform tensors inline, such as [Fresnel diffraction](tasks/syncs/fresnel_diffraction.md), [PCA](tasks/syncs/pca.md), and [FFT shift](tasks/syncs/fft_shift.md).
- **Asynchronous tasks** connect independently scheduled pipeline sections through facilities such as the [batch queue](tasks/asyncs/batch_queue.md) and [sliding average](tasks/asyncs/sliding_average.md).
- **Sinks** present or persist results through the [tensor display](tasks/sinks/display_tensor.md) and [Holofile writer](tasks/sinks/holofile.md).

Use the navigation to browse the complete task catalog and the settings accepted by each task.

The **Processed Spectrum** visualization is available with temporal **FFT** or **RFFT**.
It plots the mean squared transform magnitude over all pixels and windows in the current batch,
before frequency-band selection or spatial propagation. Power is shown on a linear scale in
arbitrary units, without FFT normalization or one-sided doubling. Frequency spacing is the input
sampling frequency divided by the temporal window size. RFFT shows nonnegative frequencies;
FFT shifts frequencies and their power values together so negative frequencies appear on the
left and zero is centered. Enable the visualization in the workspace to display the graph.
The graph refreshes at up to 100 Hz, subject to the arrival of new transform batches and GUI
availability; queued updates are bounded to prevent a display backlog.

Use the graph's **Log10 power** checkbox to plot `log10(power)` instead of linear power;
zero-power bins are omitted in this mode. **Exclude DC** hides the zero-frequency bin and
automatically fits the Y limits to the remaining bins, in either scale. Both controls update
the current graph immediately without restarting the pipeline.
