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

"""Periodic throughput measurements for streaming tensors."""

from operator import index

from ._graph import Tensor, operation


def throughput(tensor: Tensor, *, axis: int | None = 0, interval_ms: int = 1000,
               node_id: str | None = None) -> Tensor:
    """Pass a tensor through and report frames/s through Executable.receive_event().

    Use axis=None to count one frame per tensor. Otherwise count the size of
    the selected axis for every successful batch.
    """
    if not isinstance(tensor, Tensor):
        raise TypeError("throughput expects a symbolic tensor")
    if isinstance(interval_ms, bool):
        raise TypeError("interval_ms must be an integer")
    interval_ms = index(interval_ms)
    if axis is not None:
        axis = index(axis)
        if not -len(tensor.shape) <= axis < len(tensor.shape):
            raise ValueError("Invalid throughput frame axis")
        axis %= len(tensor.shape)
    return operation("Throughput", (tensor,), {"axis": axis, "interval_ms": interval_ms}, node_id)
