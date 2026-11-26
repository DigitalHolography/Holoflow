# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Build the demo processing graph."""

import holoflow as hf

from .aberration import correct_aberration
from .postprocessing import postprocess_image
from .propagation import fresnel_diffraction, reconstruct_image, spatial_filter
from .queues import output_queue, propagation_queue
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

    precision = config.time.get("precision", "fp32")
    field = process_temporal_windows(frames, temporal_frames, config.time, precision=precision)
    if config.spatial_filter["enabled"]:
        field = spatial_filter(field, config.spatial_filter, precision=precision)

    diagnostics = {}
    if config.aberration["enabled"]:
        field, diagnostics = correct_aberration(field, config, precision=precision)
    elif config.time["method"] == "pca":
        field = propagation_queue(field, config, precision=precision)

    image = reconstruct_image(field, config, precision=precision, fresnel=fresnel_diffraction)
    propagated = config.space["method"] == "fresnel"

    # ---- Image postprocessing --------------------------------------------------------------------
    image, average_window = postprocess_image(image, config, propagated)

    # ---- Output and metrics ----------------------------------------------------------------------
    image = output_queue(image, config, precision=precision)
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
