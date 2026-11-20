# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Correlate source batches with output readiness in deterministic pipelines."""

from operator import index

from ._graph import Tensor, operation


def _integer(value, name, minimum, maximum=(1 << 64) - 1):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = index(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def latency_start(tensor: Tensor, *, node_id: str, history_batches: int = 8192) -> Tensor:
    """Mark the ingress immediately after a fixed-batch Holofile source."""
    return operation("LatencyStart", (tensor,), {
        "history_batches": _integer(history_batches, "history_batches", 1),
    }, node_id)


def latency_end(tensor: Tensor, *, source_node_id: str, node_id: str,
                first_frame: int = 0, window_frames: int = 1, stride_frames: int = 1,
                interval_ms: int = 1000) -> Tensor:
    """Measure output readiness before Void or a NumPy sink fed by a ready host queue."""
    if not isinstance(source_node_id, str) or not source_node_id:
        raise ValueError("source_node_id must identify the start probe")
    return operation("LatencyEnd", (tensor,), {
        "source_node_id": source_node_id,
        "first_frame": _integer(first_frame, "first_frame", 0),
        "window_frames": _integer(window_frames, "window_frames", 1),
        "stride_frames": _integer(stride_frames, "stride_frames", 1),
        "interval_ms": _integer(interval_ms, "interval_ms", 1, (1 << 31) - 1),
    }, node_id)
