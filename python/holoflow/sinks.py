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

"""Streaming sink graph operations; commands are sent through executable events."""

from os import fspath
from ._graph import Tensor, ArrayResult, operation, _active
import json
import operator


def holofile(image: Tensor, path, *, count: int, node_id: str | None = None) -> None:
    operation("HolofileWriter", (image,), {
        "path": fspath(path), "count": count, "pipeline_settings": {}, "use_buffer": True,
    }, node_id)


def void(x: Tensor, *, node_id: str | None = None) -> None:
    operation("Void", (x,), {}, node_id)


def numpy(x: Tensor, *, axis: int = 0, length: int | None = None,
          node_id: str | None = None) -> ArrayResult:
    """Collect a finite stream by concatenating emissions into one host array.

    length is the final concatenation-axis size, inferred when omitted.
    The returned handle is symbolic; return it from the traced function.
    """
    builder = _active.get()
    if builder is None:
        raise RuntimeError("Holoflow operations require hf.trace()")
    if not isinstance(x, Tensor) or x._builder is not builder:
        raise ValueError("Input must belong to the active trace")
    axis = operator.index(axis)
    if not x.shape or not -len(x.shape) <= axis < len(x.shape):
        raise ValueError("Invalid concatenation axis")
    axis %= len(x.shape)
    if length is not None:
        length = operator.index(length)
        if length < 0:
            raise ValueError("length must be nonnegative")
    if node_id is not None and (not isinstance(node_id, str) or not node_id):
        raise ValueError("node_id must be a nonempty string")
    # A private contiguous transfer can write directly into the final host buffer.
    if x.device == "device" and axis == 0:
        itemsize = {"uint8": 1, "uint16": 2, "float32": 4, "complex64": 8}[x.dtype.name]
        strides = []
        for dim in reversed(x.shape):
            strides.append(itemsize)
            itemsize *= dim
        if x.strides == tuple(reversed(strides)):
            x = x.to("host")
    vertex, _ = builder.add("ArrayCollector", [(x._vertex, x._port)],
                            json.dumps({"axis": axis, "length": length}), node_id)
    return ArrayResult(builder, vertex)
