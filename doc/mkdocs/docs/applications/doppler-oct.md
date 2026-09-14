# Doppler OCT

!!! warning "Guide in progress"
    This page is an outline, not a validated experimental protocol. Optical specifications, acquisition parameters, and processing settings still need review by the project team.

## Result

Doppler optical coherence tomography combines depth-resolved coherent imaging with motion-sensitive analysis. The intended result is a volumetric or cross-sectional representation in which flow information is localized in depth.

## Physical principles

This section will explain coherence gating, interferometric detection, depth reconstruction, phase or frequency changes caused by motion, and the sampling limits that connect acquisition to measurable velocity.

The future [Learn](../learn/index.md) material will provide the required background in interference, Fourier analysis, OCT, phase, and Doppler processing.

## Optical setup

This section will document the validated interferometer, source and detector requirements, scanning or swept-source geometry where applicable, calibration procedure, and relevant optical-safety constraints.

## Acquisition

This section will specify acquisition dimensions, timing, synchronization, calibration inputs, metadata, and the raw data required for repeatable reconstruction.

## With Holovibes

This section will provide the operator workflow for selecting the acquisition source, configuring depth reconstruction and Doppler analysis, viewing intermediate results, and exporting recordings or measurements.

See the [Holovibes documentation](../holovibes/index.md).

## With Holoflow

This section will describe the corresponding processing graph, tensor shapes, reconstruction and analysis tasks, synchronization requirements, and output handling.

See the [Holoflow documentation](../holoflow/index.md).

## Validate the result

This section will define calibration checks, expected structural and flow views, depth and motion references, and criteria for detecting phase, sampling, or reconstruction artifacts.

## Troubleshooting

This section will cover loss of interference contrast, depth ambiguity, phase instability, synchronization errors, invalid sampling, dropped frames, and insufficient processing throughput.

## References

Validated sources supporting the final protocol will be collected in the [bibliography](../references.md).
