# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Build the demo processing graph."""

import holoflow as hf

from .aberration import correct_aberration
from .postprocessing import postprocess_image
from .propagation import fresnel_diffraction
from .queues import host_queue
from .temporal import process_temporal_windows


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
    image = hf.throughput(image, interval_ms=interval_ms, node_id="output_fps")
    measured = hf.latency_end(
        image,
        source_node_id="ingress",
        node_id="latency",
        interval_ms=interval_ms,
        window_frames=(config.correction_window + average_window - 1) * temporal_frames,
        stride_frames=temporal_frames,
    )
    if config.viewer["enabled"] and not config.viewer["cao_coefficients_only"]:
        xy = hf.sinks.numpy(measured, length=config.output_count, node_id="xy_output")
        return {"xy": xy, **diagnostics} if diagnostics else xy
    hf.sinks.void(measured, node_id="xy_output")
    if diagnostics:
        return diagnostics
