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

"""NumPy-style FFT operations on symbolic device tensors."""

from operator import index

from . import _native
from ._graph import operation
from ._ops import fftshift as _fftshift

__all__ = ["fft2", "fftshift", "rfft2", "irfft2"]


def fft2(a, *, axes=(-2, -1), norm=None, prefer_fp16=False, node_id=None):
    """Transform the last two axes, optionally allowing FP16 compute in a fused task."""
    rank = len(a.shape)
    axes = range(rank) if axes is None else axes
    axes = tuple(i + rank if i < 0 else i for i in map(index, axes))
    if rank < 2 or sorted(axes) != [rank - 2, rank - 1]:
        raise ValueError("fft2 supports only the last two axes")
    if norm is None:
        norm = "backward"
    if norm not in ("backward", "forward", "ortho"):
        raise ValueError("norm must be None, 'backward', 'forward', or 'ortho'")
    if a.dtype != _native.DType.complex64:
        a = a.astype(_native.DType.complex64)
    return operation(
        "FFT2", (a,), {"axes": [-2, -1], "inverse": False, "norm": norm,
                        **({"prefer_fp16": True} if prefer_fp16 else {})}, node_id
    )


def fftshift(x, axes=None, *, node_id=None):
    """Shift zero frequency to the center along the selected axes."""
    if axes is None:
        return _fftshift(x, axes=(), node_id=node_id)
    axes = tuple(map(index, axes)) if hasattr(axes, "__iter__") else (index(axes),)
    if not axes:
        return x
    return _fftshift(x, axes=axes, node_id=node_id)

def _real_fft(kind, a, axes, norm, shape, node_id):
    rank = len(a.shape)
    axes = tuple(i + rank if i < 0 else i for i in map(index, axes))
    if rank < 2 or axes != (rank - 2, rank - 1):
        raise ValueError("real 2-D FFTs support only the last two axes")
    norm = "backward" if norm is None else norm
    if norm not in ("backward", "forward", "ortho"):
        raise ValueError("norm must be None, 'backward', 'forward', or 'ortho'")
    params = {"axes": [-2, -1], "norm": norm}
    if shape is not None:
        params["s"] = list(map(index, shape))
        if len(params["s"]) != 2 or min(params["s"]) <= 0:
            raise ValueError("s must contain two positive dimensions")
    return operation(kind, (a,), params, node_id)


def rfft2(a, *, axes=(-2, -1), norm=None, node_id=None):
    """Transform a device float32 tensor to its packed complex spectrum."""
    return _real_fft("RFFT2", a, axes, norm, None, node_id)


def irfft2(a, s=None, *, axes=(-2, -1), norm=None, node_id=None):
    """Invert a packed spectrum, optionally specifying its matching spatial shape."""
    return _real_fft("IRFFT2", a, axes, norm, s, node_id)
