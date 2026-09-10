# Learn

This section will build the background needed to understand, adapt, and implement Holoflow experiments. It will connect the equations and physical models to the software that executes them in real time.

## Signal processing

The signal-processing path will progress from sampling and aliasing through Fourier transforms, filtering, convolution, numerical propagation, short-time analysis, PCA, and the interpretation of reconstructed data.

These concepts will link directly to the processing stages used in the [application guides](../applications/index.md) and to their implementations in the [task reference](../holovibes/tasks/syncs/fft_shift.md).

Start with [Optical heterodyne detection](heterodyne-detection.md) to see how detuning moves an optical signal into a measurable frequency band and how a local oscillator changes the detected signal.

## Physics

The physics path will introduce coherent light, interference, diffraction, wavefront propagation, Doppler effects, coherence gating, OCT, and optical aberrations. The emphasis will be on building enough intuition to understand experimental geometry and the assumptions behind reconstruction algorithms.

## Software and HPC

The software path will cover modern C++, CUDA execution, memory and tensor layouts, concurrency, task graphs, scheduling, measurement, and profiling. It will explain how scientific pipelines become predictable real-time systems and connect those ideas to [Holoflow](../holoflow/index.md).

!!! info "Learning material in progress"
    Additional lessons will be added as complete, reviewed units. The [Applications](../applications/index.md) and [Documentation](../documentation/index.md) sections provide the practical and software-oriented entry points.
