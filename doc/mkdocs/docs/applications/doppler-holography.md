# Doppler Holography

!!! warning "Guide in progress"
    This page is an outline, not a validated experimental protocol. Optical specifications, acquisition parameters, and processing settings still need review by the project team.

## Result

Laser Doppler holography combines interferometric imaging with temporal analysis to reveal motion-induced frequency broadening. In retinal imaging, the processed result can visualize blood flow across retinal and choroidal vessels.

## Physical principles

The experiment records interferograms formed by a reference beam and light backscattered by the sample. Numerical propagation reconstructs the complex optical field, while short-time temporal analysis separates and summarizes Doppler-shifted components.

The future [Learn](../learn/index.md) material will introduce interference, diffraction, sampling, numerical propagation, and Doppler analysis.

## Optical setup

This section will document the validated illumination, interferometer geometry, detection path, camera requirements, alignment procedure, and applicable laser and subject-safety constraints.

## Acquisition

This section will document camera configuration, sampling requirements, acquisition duration, calibration data, and the metadata required to reproduce a measurement.

## With Holovibes

This section will provide the operator workflow: selecting a live camera or Holofile recording, configuring spatial reconstruction, selecting temporal analysis, inspecting the live result, and recording raw or processed data.

See the [Holovibes documentation](../holovibes/index.md) and current [task reference](../holovibes/tasks/syncs/stft.md).

## With Holoflow

This section will describe the equivalent processing graph, its input and output tensors, the tasks used for spatial reconstruction and temporal analysis, and how an application executes and inspects the graph.

See the [Holoflow documentation](../holoflow/index.md).

## Validate the result

This section will define expected intermediate views, quality checks, reference measurements, and criteria for recognizing acquisition or reconstruction artifacts.

## Troubleshooting

This section will cover alignment problems, sampling artifacts, insufficient temporal bandwidth, dropped frames, reconstruction errors, and performance bottlenecks.

## References

The optical and processing protocol will cite validated publications from the [bibliography](../references.md). The homepage demonstration is based on the processing described by [Puyo et al.](../references.md#puyo-2018).
