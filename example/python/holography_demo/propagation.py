# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Spatial propagation operators for the demo."""

from math import isfinite, pi

import holoflow as hf
from holoflow import _native


def spatial_filter(field, settings, *, precision):
    half = precision == "fp16" and field.dtype == hf.float16
    return hf.filter2d(
        field, settings["r_inner"], settings["r_outer"], settings["s_inner"], settings["s_outer"],
        use_fused_f16=half and _native.supports_sm120(), normalize=half, node_id="spatial_filter",
    )


def reconstruct_image(field, config, *, precision, fresnel):
    space = config.space
    propagated = space["method"] == "fresnel"
    if propagated:
        pitch = space["pixel_pitch_m"]
        args = (field, space["wavelength_m"], pitch, pitch, space["distance_m"])
        if precision == "fp16" and field.dtype in (hf.float16, hf.DType.complex32):
            widened = field.astype(hf.float32 if field.dtype == hf.float16 else hf.complex64)
            lens = quadratic_lens(*field.shape[-2:], *args[1:])
            magnitude = hf.abs(hf.fft.fft2(widened * lens, prefer_fp16=True))
            reduced = hf.mean(magnitude, axis=-3, keepdims=True)
            return hf.mean(hf.fft.fftshift(reduced, axes=(-2, -1)), axis=0)
        else:
            field = fresnel(*args, output_magnitude=True)
    elif config.aberration["enabled"] or config.time["method"] == "pca":
        field = hf.abs(field)
    image = hf.mean(field, axis=1)
    return hf.fft.fftshift(image, axes=(-2, -1)) if propagated else image


def fresnel_diffraction_decomposed(field, wavelength, dx, dy, z, *, output_magnitude):
    """Return the Fresnel magnitude over the last two axes using graph primitives."""
    if not output_magnitude:
        raise ValueError(
            "Decomposed Fresnel propagation requires output_magnitude=True"
        )
    if field.device != "device" or field.dtype not in (hf.float32, hf.complex64):
        raise ValueError(
            "Decomposed Fresnel propagation requires a float32 or complex64 device field"
        )
    if len(field.shape) < 2:
        raise ValueError("Decomposed Fresnel propagation requires two spatial axes")
    if not isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be finite and positive")
    if not isfinite(dx) or dx <= 0 or dx != dy:
        raise ValueError("dx and dy must be equal, finite, and positive")
    if not isfinite(z) or z == 0:
        raise ValueError("z must be finite and nonzero")

    height, width = field.shape[-2:]
    lens = quadratic_lens(height, width, wavelength, dx, dy, z)
    return hf.abs(hf.fft.fft2(field * lens))


def quadratic_lens(height, width, wavelength, dx, dy, z):
    # Match the fused kernel's coordinates on a centered square grid.
    size = max(height, width)
    start_x = (size - width) // 2 - size / 2
    start_y = (size - height) // 2 - size / 2
    x, y = hf.meshgrid(
        hf.arange(start_x, start_x + width), hf.arange(start_y, start_y + height)
    )
    x_m = x * dx
    y_m = y * dy
    radius_squared = hf.square(x_m) + hf.square(y_m)
    phase = radius_squared * (1j * pi / (wavelength * z))
    return hf.exp(phase)


fresnel_diffraction = fresnel_diffraction_decomposed

__all__ = ["fresnel_diffraction"]
