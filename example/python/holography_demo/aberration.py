# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Estimate and correct aberrations and collect CAO diagnostics."""

from functools import partial
from math import pi

import holoflow as hf
from holoflow import _native

from . import aberration_ops
from .postprocessing import flatfield
from .queues import host_queue


def _correct_aberration(field, config, *, subaperture_fresnel, zernike_phase, apply_phase, normalize):
    settings = config.aberration
    correction_window = config.correction_window
    height, width = field.shape[-2:]
    subaperture_height = height // settings["subapertures"]
    subaperture_width = width // settings["subapertures"]
    if min(subaperture_height, subaperture_width) < 2:
        raise ValueError(
            "aberration.subapertures must leave at least two pixels per subaperture"
        )
    space = config.space
    wavelength_m, pixel_pitch_m, distance_m = (
        space["wavelength_m"],
        space["pixel_pitch_m"],
        space["distance_m"],
    )
    # ---- Shack-Hartmann images -------------------------------------------------------------------
    current_field, delayed_field, _ = hf.dual_reader_batch_queue(
        field,
        config.queues["propagation_windows"],
        correction_window,
        node_id="correction_input",
    )
    shack_hartmann_images = subaperture_fresnel(
        current_field,
        wavelength_m,
        pixel_pitch_m,
        pixel_pitch_m,
        distance_m,
        win_h=subaperture_height,
        win_w=subaperture_width,
        stride_y=subaperture_height,
        stride_x=subaperture_width,
    )
    shack_hartmann_images = hf.mean(shack_hartmann_images, axis=1)
    shack_hartmann_images = hf.causal_sliding_average(
        shack_hartmann_images, correction_window
    )
    shack_hartmann_images = hf.fft.fftshift(shack_hartmann_images, axes=(-2, -1))
    if (cutoff_m := config.postprocessing["flatfield_cutoff_m"]) is not None:
        shack_hartmann_images = flatfield(
            shack_hartmann_images, cutoff_m, space, propagated=True
        )

    # ---- Zernike phase fitting and correction ----------------------------------------------------
    skip_outside_pupil = settings["skip_outside_pupil"]
    slope_mode = settings["slope_mode"]
    if slope_mode == "single_reference" and any(
        size % 2 == 0 for size in shack_hartmann_images.shape[1:3]
    ):
        raise ValueError(
            "Single-reference CAO requires an odd reconstructed subaperture grid"
        )
    slopes = hf.shack_hartmann_slopes(
        shack_hartmann_images,
        wavelength_m,
        pixel_pitch_m,
        pixel_pitch_m,
        distance_m,
        subaperture_height=subaperture_height,
        subaperture_width=subaperture_width,
        stride_y=subaperture_height,
        stride_x=subaperture_width,
        mode=slope_mode,
        skip_subapertures_outside_pupil=skip_outside_pupil,
    )
    zernike_modes = settings["zernike_modes"]
    coefficients = hf.zernike_from_slopes(
        slopes,
        zernike_modes,
        wavelength_m,
        pixel_pitch_m,
        pixel_pitch_m,
        subaperture_height=subaperture_height,
        subaperture_width=subaperture_width,
        stride_y=subaperture_height,
        stride_x=subaperture_width,
        skip_subapertures_outside_pupil=skip_outside_pupil,
    )[0, 0, :]
    phase = zernike_phase(coefficients, zernike_modes, height, width)
    # The delayed reader aligns the field with the middle of the averaging window
    # by delaying it by (correction_window - 1) // 2 samples.
    corrected = apply_phase(delayed_field, phase)
    corrected = hf.batch_queue(
        corrected, config.queues["corrected_windows"], 1, 1, node_id="corrected_windows"
    )

    # ---- CAO diagnostics -------------------------------------------------------------------------
    diagnostic_tensors = {}
    if config.viewer["cao_diagnostics"] or config.viewer["cao_coefficients_only"]:
        diagnostic_tensors["coefficients"] = coefficients.reshape((1, len(zernike_modes)))
    if config.viewer["cao_diagnostics"]:
        mosaic = normalize(shack_hartmann_images, axes=(-2, -1), lo=0, hi=255)
        _, rows, columns, _, _ = mosaic.shape
        mosaic = hf.transpose(mosaic, (0, 1, 3, 2, 4))
        mosaic = mosaic.reshape(
            (1, rows * subaperture_height, columns * subaperture_width)
        ).astype(hf.uint8, "scaled")
        wrapped = (phase % (2 * pi)).reshape((1, height, width))
        diagnostic_tensors.update(shack_hartmann=mosaic, phase=wrapped)
    diagnostics = {}
    for name, value in diagnostic_tensors.items():
        # Keep host transfers out of the CAO CUDA graph section.
        queued = hf.batch_queue(
            value, config.queues["output_images"], 1, 1, node_id=name + "_queue_device"
        )
        host = host_queue(queued, config, name + "_queue")
        diagnostics[name] = hf.sinks.numpy(host, length=config.temporal_count)
    return corrected, diagnostics


def correct_aberration_native(field, config, *, precision="fp32"):
    output_cf16 = precision == "fp16" and field.dtype == hf.float16 and _native.supports_sm120()
    return _correct_aberration(
        field,
        config,
        subaperture_fresnel=hf.short_time_fresnel_diffraction,
        zernike_phase=hf.zernike_phase,
        apply_phase=partial(hf.correct_phase, output_cf16=output_cf16),
        normalize=hf.normalize,
    )


def correct_aberration_decomposed(field, config):
    return _correct_aberration(
        field,
        config,
        subaperture_fresnel=aberration_ops.subaperture_fresnel,
        zernike_phase=aberration_ops.zernike_phase,
        apply_phase=aberration_ops.apply_phase,
        normalize=aberration_ops.normalize,
    )


def correct_aberration(field, config, *, precision="fp32"):
    if precision == "fp16":
        return correct_aberration_native(field, config, precision=precision)
    return correct_aberration_decomposed(field, config)
