# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""NumPy-like graph operations for the demo's aberration correction."""

from math import sqrt

import holoflow as hf

from .propagation import quadratic_lens


def subaperture_fresnel(field, wavelength, dx, dy, z, win_h, win_w, stride_y, stride_x):
    """Propagate non-overlapping tiles with a shared full-field phase reference."""
    if (stride_y, stride_x) != (win_h, win_w):
        raise ValueError("Decomposed subaperture Fresnel requires non-overlapping windows")
    leading = field.shape[:-2]
    height, width = field.shape[-2:]
    rows, columns = height // win_h, width // win_w
    phased = field * quadratic_lens(height, width, wavelength, dx, dy, z)
    if height % win_h or width % win_w:
        crop = (slice(None),) * len(leading) + (
            slice(0, rows * win_h), slice(0, columns * win_w)
        )
        phased = phased[crop]

    tiles = phased.reshape((*leading, rows, win_h, columns, win_w), copy=False)
    n = len(leading)
    tiles = hf.transpose(tiles, (*range(n), n, n + 2, n + 1, n + 3))
    packed_tiles = tiles.reshape(tiles.shape, copy=True)
    return hf.abs(hf.fft.fft2(packed_tiles))


def _zernike_basis(mode, x, y, x_squared, y_squared):
    # Match the native Noll convention, including the modes' orientation.
    match mode:
        case 2:
            return 2 * x
        case 3:
            return 2 * y
        case 4:
            return sqrt(3) * (2 * (x_squared + y_squared) + (-1))
        case 5:
            return (2 * sqrt(6)) * x * y
        case 6:
            return sqrt(6) * (x_squared + (-1) * y_squared)
        case 7:
            return sqrt(8) * y * (3 * x_squared + (-1) * y_squared)
        case 8:
            return sqrt(8) * y * (3 * (x_squared + y_squared) + (-2))
        case 9:
            return sqrt(8) * x * (3 * (x_squared + y_squared) + (-2))
        case 10:
            return sqrt(8) * x * (x_squared + (-3) * y_squared)
        case _:
            raise ValueError("Only Zernike Noll indexes 2 through 10 are supported")


def zernike_phase(coefficients, modes, height, width):
    """Synthesize an unmasked phase map from the fitted coefficients."""
    if not modes or len(set(modes)) != len(modes) or coefficients.shape != (len(modes),):
        raise ValueError("Zernike synthesis requires one coefficient per unique mode")

    x_start = -(width - 1) / 2
    y_start = -(height - 1) / 2
    x, y = hf.meshgrid(
        hf.arange(x_start, x_start + width), hf.arange(y_start, y_start + height)
    )
    inverse_radius = 2 / min(height, width)
    x, y = x * inverse_radius, y * inverse_radius
    x_squared, y_squared = hf.square(x), hf.square(y)

    terms = []
    for index, mode in enumerate(modes):
        basis = _zernike_basis(mode, x, y, x_squared, y_squared)
        terms.append(basis * coefficients[index:index + 1])
    phase = terms[0]
    for term in terms[1:]:
        phase = phase + term
    return phase


def apply_phase(field, phase):
    return field * hf.exp(-1j * phase)


def normalize(image, axes, lo, hi):
    """Normalize finite sensor images independently over their spatial axes."""
    minimum = hf.min(image, axis=axes, keepdims=True)
    maximum = hf.max(image, axis=axes, keepdims=True)
    return ((image - minimum) / (maximum - minimum + 1e-7)) * (hi - lo) + lo
