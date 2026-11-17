# Copyright 2026 Digital Holography Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming source graph operations."""

from os import fspath
from . import _native
from ._graph import Tensor, operation, _active


def holofile(path, batch_size: int = 32, load_kind: str = "live",
             start_frame: int = 0, end_frame: int | None = None,
             max_fps: int | None = None, *, loop: bool = True, tail: str = "error", node_id: str | None = None) -> Tensor:
    if _active.get() is None:
        raise RuntimeError("Holoflow operations require hf.trace()")
    kinds = {"live": "Live", "cpu_cached": "CPUCached", "gpu_cached": "GPUCached"}
    if load_kind not in kinds:
        raise ValueError("load_kind must be live, cpu_cached, or gpu_cached")
    if not isinstance(loop, bool):
        raise TypeError("loop must be bool")
    if tail not in ("error", "drop"):
        raise ValueError("tail must be error or drop")
    path = fspath(path)
    if end_frame is None:
        end_frame = _native.file_frame_count(path)
    return operation("Holofile", (), {
        "path": path, "batch_size": batch_size, "load_kind": kinds[load_kind],
        "start_frame": start_frame, "end_frame": end_frame,
        "max_fps": max_fps, "keep_cursor": False, "loop": loop, "tail": tail,
    }, node_id)
