# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Prepare temporal windows and apply optional PCA."""

import holoflow as hf


def _reshape_window(frames, temporal_frames):
    return frames.reshape((1, temporal_frames, *frames.shape[-2:]), copy=False)


def process_temporal_windows(frames, temporal_frames, settings, *, precision="fp32"):
    if settings["method"] == "pca":
        return pca(frames, temporal_frames, settings, precision=precision)
    return _reshape_window(frames.astype(hf.float32), temporal_frames)


def pca(frames, temporal_frames, settings, *, precision="fp32"):
    """Project temporal windows using the selected precision."""
    begin, end = settings["component_begin"], settings["component_end"]
    if precision == "fp32":
        windows = _reshape_window(frames.astype(hf.float32), temporal_frames)
        return hf.pca(windows, begin, end)
    if frames.dtype != hf.uint8 or frames.shape != (32, 320, 512):
        raise ValueError("time.precision: fp16 requires U8 input with shape [32, 320, 512]")
    # Both queues own their storage. Separate them with the native device copy.
    windows = _reshape_window(frames.to("device"), temporal_frames)
    grouped, eigenvectors = hf.pca_decomposition_queue(windows, 4, 3, use_fused_f16=True)
    return hf.pca_projection(
        grouped, eigenvectors, begin, end,
        use_fused_f16=True,
    )
