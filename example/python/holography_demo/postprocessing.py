# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Apply flat-field correction and prepare reconstruction images."""

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
    return hf.flatfield(
        image, gaussian_width_m / sample_spacing_y_m, gaussian_width_m / sample_spacing_x_m
    )


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
