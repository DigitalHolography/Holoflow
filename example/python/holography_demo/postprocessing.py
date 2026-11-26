# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Apply flat-field correction and prepare reconstruction images."""

from ctypes import c_float
from math import ceil, isfinite, pi

import holoflow as hf


def flatfield(image, cutoff_m, space, propagated):
    height, width = image.shape[-2:]
    pixel_pitch_m = space["pixel_pitch_m"]
    sample_spacing_y_m = sample_spacing_x_m = pixel_pitch_m
    if propagated:
        # Fresnel propagation sets the sample spacing through the output grid size.
        fresnel_sampling_span_m = (
            space["wavelength_m"] * abs(space["distance_m"]) / pixel_pitch_m
        )
        sample_spacing_y_m = fresnel_sampling_span_m / height
        sample_spacing_x_m = fresnel_sampling_span_m / width
    gaussian_width_m = 0.187 * cutoff_m
    return flatfield_decomposed(
        image, gaussian_width_m / sample_spacing_y_m, gaussian_width_m / sample_spacing_x_m
    )


def flatfield_decomposed(image, sigma_y, sigma_x):
    sigma_y, sigma_x = c_float(sigma_y).value, c_float(sigma_x).value
    if not all(isfinite(sigma) and sigma > 0 for sigma in (sigma_y, sigma_x)):
        raise ValueError("flatfield sigmas must be finite and positive")
    height, width = image.shape[-2:]
    radius_y, radius_x = (ceil(c_float(3 * sigma).value) for sigma in (sigma_y, sigma_x))
    if height > 1 and width > 1 and max(radius_y, radius_x) >= 32:
        fx = hf.arange(0, width // 2 + 1) / width
        fy = ((hf.arange(0, height) + height // 2) % height - height // 2) / height
        fx, fy = hf.meshgrid(fx, fy)
        exponent = -2 * pi**2 * (hf.square(sigma_x * fx) + hf.square(sigma_y * fy))
        gain = 1 - hf.exp(exponent)
        return hf.fft.irfft2(hf.fft.rfft2(image) * gain, s=(height, width))

    background = image
    for axis, radius, sigma in ((-1, radius_x, sigma_x), (-2, radius_y, sigma_y)):
        offsets = hf.arange(-radius, radius + 1)
        weights = hf.exp(-0.5 * hf.square(offsets / sigma))
        weights = weights / hf.sum(weights, keepdims=True)
        widths = [(0, 0)] * len(image.shape)
        widths[axis] = (radius, radius)
        padded = hf.pad(background, widths, mode="edge")
        windows = hf.sliding_window_view(padded, 2 * radius + 1, axis=axis)
        background = hf.sum(windows * weights, axis=-1)
    return image - background


def postprocess_image(image, config, propagated):
    postprocessing = config.postprocessing
    if (cutoff_m := postprocessing["flatfield_cutoff_m"]) is not None:
        image = flatfield(image, cutoff_m, config.space, propagated)
    average_window = postprocessing["average_window"]
    if config.correction_window > 1 or average_window > 1:
        image = hf.sliding_average(
            image, 8, average_window, discard_first=config.correction_window - 1
        )
    if (percentile_clip := postprocessing["percentile_clip"]) is not None:
        image = hf.pct_clip(image, **percentile_clip)
    image = image.astype(hf.uint8, "scaled")
    return image, average_window
