# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Spatial propagation operators for the demo."""

from math import isfinite, pi

import holoflow as hf


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
    lens = hf.exp(phase)
    return hf.abs(hf.fft.fft2(field * lens))


# Set this to fresnel_diffraction_decomposed to use the primitive graph.
fresnel_diffraction = hf.fresnel_diffraction
fresnel_diffraction = fresnel_diffraction_decomposed

__all__ = ["fresnel_diffraction"]
