# Wavefront Analysis

!!! warning "Guide in progress"
    This page is an outline, not a validated experimental protocol. Optical specifications, acquisition parameters, and processing settings still need review by the project team.

## Result

Wavefront analysis estimates optical aberrations from measured changes in an optical field. The intended workflow will cover Shack–Hartmann measurements and the representation of reconstructed aberrations with Zernike modes.

## Physical principles

This section will introduce local wavefront slopes, spot displacement, wavefront reconstruction, pupil geometry, and the interpretation of Zernike coefficients.

The future [Learn](../learn/index.md) material will connect geometrical optics, diffraction, sampling, and numerical estimation to this application.

## Optical setup

This section will document the validated illumination, pupil and lenslet-array geometry, detector requirements, reference measurement, alignment procedure, and applicable optical-safety constraints.

## Acquisition

This section will specify reference and measurement frames, exposure and sampling requirements, calibration data, pupil definition, and metadata needed for repeatable analysis.

## With Holovibes

This section will provide the operator workflow for acquiring a reference, detecting and tracking spots, reconstructing the wavefront, inspecting Zernike modes, and exporting results.

See the [Holovibes documentation](../holovibes/index.md).

## With Holoflow

This section will describe the equivalent task graph, its calibration inputs, spot-analysis and reconstruction stages, output tensors, and integration into a larger experiment.

See the [Holoflow documentation](../holoflow/index.md).

## Validate the result

This section will define reference-wavefront checks, residual and repeatability metrics, known-aberration tests, and criteria for recognizing invalid spot detection or pupil fitting.

## Troubleshooting

This section will cover missing or merged spots, poor reference calibration, incorrect pupil geometry, unstable coefficients, invalid units, and processing-performance problems.

## References

Validated sources supporting the final protocol will be collected in the [bibliography](../references.md).
