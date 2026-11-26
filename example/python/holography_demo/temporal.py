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
    raise ValueError("FP16 PCA requires the pipelined frame input")


def pipelined_pca(frames, temporal_frames, capacity, settings):
    if frames.dtype != hf.uint8 or (temporal_frames, *frames.shape[-2:]) != (32, 320, 512):
        raise ValueError("time.precision: fp16 requires U8 windows with shape [32, 320, 512]")
    batch = 4
    stride = batch * temporal_frames
    x_cov, x_proj = hf.multi_reader_batch_queue(
        frames, max(capacity, 4 * stride), stride, stride, 2, tail="drop", node_id="uploaded_frames",
    )
    height, width = frames.shape[-2:]
    shape = (batch, temporal_frames, height * width)
    gram_input = x_cov.reshape(shape, copy=False).astype(hf.float32, node_id="pca_gram_cast")
    covariance = gram_input @ gram_input.transpose(0, 2, 1)
    covariance = hf.batch_queue(covariance, 2 * batch, batch, batch)
    _, vectors = hf.linalg.eigh(covariance, UPLO="U")
    vectors = hf.batch_queue(vectors, 2 * batch, batch, batch)
    begin, end = settings["component_begin"], settings["component_end"]
    coefficients = vectors.astype(hf.float16, node_id="pca_vector_cast").transpose(0, 2, 1)[:, begin:end, :]
    samples = x_proj.reshape(shape, copy=False).astype(hf.float16, node_id="pca_input_cast")
    return (coefficients @ samples).reshape((batch, end - begin, height, width), copy=False)
