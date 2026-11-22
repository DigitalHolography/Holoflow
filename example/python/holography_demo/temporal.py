# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Prepare temporal windows and apply optional PCA."""

import holoflow as hf


def process_temporal_windows(frames, temporal_frames, settings):
    frames = frames.astype(hf.float32)
    height, width = frames.shape[-2:]
    field = frames.reshape((1, temporal_frames, height, width), copy=False)
    if settings["method"] == "pca":
        field = hf.pca(
            field, settings["component_begin"], settings["component_end"]
        )
    return field
