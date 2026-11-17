"""Streaming source graph operations."""

from os import fspath
from . import _native
from ._graph import Tensor, operation, _active


def holofile(path, batch_size: int = 32, load_kind: str = "live",
             start_frame: int = 0, end_frame: int | None = None,
             max_fps: int | None = None, *, node_id: str | None = None) -> Tensor:
    if _active.get() is None:
        raise RuntimeError("Holoflow operations require hf.trace()")
    kinds = {"live": "Live", "cpu_cached": "CPUCached", "gpu_cached": "GPUCached"}
    if load_kind not in kinds:
        raise ValueError("load_kind must be live, cpu_cached, or gpu_cached")
    path = fspath(path)
    if end_frame is None:
        end_frame = _native.file_frame_count(path)
    return operation("Holofile", (), {
        "path": path, "batch_size": batch_size, "load_kind": kinds[load_kind],
        "start_frame": start_frame, "end_frame": end_frame,
        "max_fps": max_fps, "keep_cursor": False,
    }, node_id)
