# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Build the demo processing graph."""

import holoflow as hf

from .aberration import correct_aberration
from .postprocessing import postprocess_image
from .propagation import fresnel_diffraction
from .queues import host_queue
from .temporal import process_temporal_windows


def _fp16_image(frames, config):
    if frames.dtype != hf.uint8 or frames.shape != (32, 320, 512):
        raise ValueError("time.precision: fp16 requires U8 input with shape [32, 320, 512]")
    # Both queues own their storage. Separate them with the native device copy.
    windows = frames.to("device").reshape((1, 32, 320, 512), copy=False)
    grouped, eigenvectors = hf.pca_decomposition_queue(windows, 4, 3, use_fused_f16=True)
    projected = hf.pca_projection(
        grouped, eigenvectors, config.time["component_begin"], config.time["component_end"],
        use_fused_f16=True,
    )
    current, _, _ = hf.dual_reader_batch_queue(
        projected, config.queues["propagation_windows"], config.postprocessing["average_window"]
    )
    space = config.space
    pitch = space["pixel_pitch_m"]
    image = hf.fresnel_diffraction(
        current, space["wavelength_m"], pitch, pitch, space["distance_m"],
        output_magnitude=True, use_fused_f16=True, reduce_magnitude=True,
    )
    return hf.mean(hf.fft.fftshift(image, axes=(-2, -1)), axis=0)


def pipeline(config):
    # ---- Frame input and GPU upload --------------------------------------------------------------
    source = config.input
    interval_ms = config.metrics["interval_ms"]
    temporal_frames = config.temporal_frames
    frames = hf.sources.holofile(
        source["path"],
        batch_size=source["batch_size"],
        load_kind=source["load_kind"],
        start_frame=source["start_frame"],
        end_frame=source["end_frame"],
        max_fps=source["max_fps"],
        loop=source["mode"] == "looping",
        tail=source["tail"],
    )
    frames = hf.latency_start(
        frames, node_id="ingress", history_batches=config.metrics["history_batches"]
    )
    frames = hf.throughput(frames, interval_ms=interval_ms, node_id="input_fps")
    # A cached GPU source owns its output, while BatchQueue owns its input.
    frames = hf.copy(frames) if frames.device == "device" else frames.to("device")
    frames = hf.batch_queue(
        frames,
        config.queues["input_frames"],
        temporal_frames,
        temporal_frames,
        node_id="uploaded_frames",
    )

    if config.time.get("precision") == "fp16":
        image, average_window = postprocess_image(_fp16_image(frames, config), config, True)
        image = hf.batch_queue(image, config.queues["output_images"], 1, 1, node_id="device_output")
        return _output(image, config, interval_ms, average_window)

    # ---- Temporal and spatial preprocessing ------------------------------------------------------
    field = process_temporal_windows(frames, temporal_frames, config.time)
    space = config.space
    spatial_filter = config.spatial_filter
    if spatial_filter["enabled"]:
        field = hf.filter2d(
            field,
            spatial_filter["r_inner"],
            spatial_filter["r_outer"],
            spatial_filter["s_inner"],
            spatial_filter["s_outer"],
            node_id="spatial_filter",
        )
    if config.time["method"] == "pca" and not config.aberration["enabled"]:
        field = hf.batch_queue(
            field,
            config.queues["propagation_windows"],
            1,
            1,
            node_id="propagation_windows",
        )

    # ---- Aberration correction -------------------------------------------------------------------
    diagnostics = {}
    if config.aberration["enabled"]:
        field, diagnostics = correct_aberration(field, config)

    # ---- Image reconstruction --------------------------------------------------------------------
    propagated = space["method"] == "fresnel"
    if propagated:
        pixel_pitch_m = space["pixel_pitch_m"]
        field = fresnel_diffraction(
            field,
            space["wavelength_m"],
            pixel_pitch_m,
            pixel_pitch_m,
            space["distance_m"],
            output_magnitude=True,
        )
    elif config.aberration["enabled"] or config.time["method"] == "pca":
        field = hf.abs(field)
    image = hf.mean(field, axis=1)
    if propagated:
        image = hf.fft.fftshift(image, axes=(-2, -1))

    # ---- Image postprocessing --------------------------------------------------------------------
    image, average_window = postprocess_image(image, config, propagated)

    # ---- Output and metrics ----------------------------------------------------------------------
    image = host_queue(image, config, "host_output")
    return _output(image, config, interval_ms, average_window, diagnostics)


def _output(image, config, interval_ms, average_window, diagnostics=None):
    image = hf.throughput(image, interval_ms=interval_ms, node_id="output_fps")
    measured = hf.latency_end(
        image,
        source_node_id="ingress",
        node_id="latency",
        interval_ms=interval_ms,
        window_frames=(config.correction_window + average_window - 1) * config.temporal_frames,
        stride_frames=config.temporal_frames,
    )
    if config.viewer["enabled"] and not config.viewer["cao_coefficients_only"]:
        xy = hf.sinks.numpy(measured, length=config.output_count, node_id="xy_output")
        return {"xy": xy, **diagnostics} if diagnostics else xy
    hf.sinks.void(measured, node_id="xy_output")
    if diagnostics:
        return diagnostics
