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

"""
Holoflow holonp test oracle.

Usage:
    python oracle.py <manifest.json>

The manifest contains:
    op              : operator name (e.g. "abs")
    inputs          : list of {shape, dtype, strides, offset, payload}
    output_payloads : list of output filenames to write

Input payloads are dense logical bytes (num_bytes() each) starting from the
logical tensor origin (i.e. after the declared offset has been consumed by C++).
Outputs are written as dense C-contiguous arrays.
"""

import sys
import json
import numpy as np
from pathlib import Path


# -------------------------------------------------------------------------------------------------
# Dtype mapping
# -------------------------------------------------------------------------------------------------

_DTYPE_MAP = {
    "U8":   np.uint8,
    "U16":  np.uint16,
    "F32":  np.float32,
    "CF32": np.complex64,
}


# -------------------------------------------------------------------------------------------------
# Tensor I/O helpers
# -------------------------------------------------------------------------------------------------

def _read_input(workdir, desc):
    """Read a dense input payload and return a numpy array."""
    shape = tuple(int(d) for d in desc["shape"])
    np_dtype = _DTYPE_MAP[desc["dtype"]]
    raw = (workdir / desc["payload"]).read_bytes()
    dense_size = int(np.prod(shape)) * np.dtype(np_dtype).itemsize if shape else 0

    # Backward-compatible path: dense logical payload (existing tests).
    if len(raw) == dense_size:
        arr = np.frombuffer(raw, dtype=np_dtype)
        if shape:
            arr = arr.reshape(shape)
        return arr.copy()

    # Strided backing-store path: payload includes row/plane padding.
    strides = tuple(int(s) for s in desc["strides"])
    offset = int(desc.get("offset", 0))
    if not shape:
        return np.array([], dtype=np_dtype)
    return np.ndarray(shape=shape, dtype=np_dtype, buffer=raw, offset=offset, strides=strides).copy()


def _write_output(workdir, fname, arr):
    """Write a C-contiguous numpy array as a raw binary file."""
    out = np.ascontiguousarray(arr)
    (workdir / fname).write_bytes(out.tobytes())


# -------------------------------------------------------------------------------------------------
# Operator implementations
# -------------------------------------------------------------------------------------------------

def _op_abs(inputs, settings):
    x = inputs[0]
    if x.dtype == np.complex64:
        result = np.abs(x).astype(np.float32)
    else:
        result = np.abs(x)
    return [result]


def _op_add(inputs, settings):
    return [inputs[0] + inputs[1]]


def _op_argmax(inputs, settings):
    x = inputs[0]
    raw_axis = settings.get("axis")   # None (JSON null), int, or list
    keepdims = settings.get("keepdims", False)

    if raw_axis is None:
        # Global argmax: return flat index of the maximum element.
        flat_idx = int(np.argmax(x.ravel()))
        if keepdims:
            result = np.full([1] * x.ndim, flat_idx, dtype=np.uint16)
        else:
            result = np.array(flat_idx, dtype=np.uint16)
    elif isinstance(raw_axis, int):
        axis = raw_axis if raw_axis >= 0 else raw_axis + x.ndim
        idx = np.argmax(x, axis=axis)
        if keepdims:
            result = np.expand_dims(idx, axis=axis).astype(np.uint16)
        else:
            result = idx.astype(np.uint16)
    else:
        raise NotImplementedError(f"Multi-axis argmax oracle not supported (axis={raw_axis!r})")

    return [result]


def _arange_len(start, stop, step):
    forward = step > 0.0
    if forward and stop <= start:
        return 0
    if (not forward) and stop >= start:
        return 0
    n = int(np.ceil((stop - start) / step))
    if n <= 0:
        raise ValueError("Arange: invalid resulting length")
    return n


def _op_arange(inputs, settings):
    start = float(settings["start"])
    stop = float(settings["stop"])
    step = float(settings["step"])
    if step == 0.0:
        raise ValueError("Arange: step must be non-zero")

    dtype_name = settings.get("dtype", "F32")
    n = _arange_len(start, stop, step)
    values = start + step * np.arange(n, dtype=np.float64)

    if dtype_name == "CF32":
        result = values.astype(np.float32).astype(np.complex64)
    else:
        result = values.astype(_DTYPE_MAP[dtype_name])
    return [result]


def _op_asarray(inputs, settings):
    value = float(settings["value"])
    return [np.array([value], dtype=np.float32)]


def _op_ascontiguousarray(inputs, settings):
    return [np.ascontiguousarray(inputs[0])]


def _op_conj(inputs, settings):
    return [np.conjugate(inputs[0])]


def _op_concatenate(inputs, settings):
    axis = settings.get("axis", 0)
    return [np.concatenate(inputs, axis=axis)]


def _op_copy(inputs, settings):
    return [np.ascontiguousarray(inputs[0])]


def _op_subtract(inputs, settings):
    return [inputs[0] - inputs[1]]


def _promote_multiply(a_dtype, b_dtype):
    if a_dtype == b_dtype:
        return a_dtype
    if a_dtype == np.complex64 or b_dtype == np.complex64:
        return np.complex64
    if a_dtype == np.float32 or b_dtype == np.float32:
        return np.float32
    if a_dtype == np.uint16 or b_dtype == np.uint16:
        return np.uint16
    return np.uint8


def _op_multiply(inputs, settings):
    a, b = inputs[0], inputs[1]
    out_dtype = _promote_multiply(a.dtype.type, b.dtype.type)
    return [(a * b).astype(out_dtype)]


def _op_divide(inputs, settings):
    a, b = inputs[0], inputs[1]
    if a.dtype == b.dtype:
        if a.dtype == np.uint8 or a.dtype == np.uint16:
            return [np.floor_divide(a, b).astype(a.dtype)]
        return [(a / b).astype(a.dtype)]
    if a.dtype == np.complex64 and (b.dtype == np.float32 or b.dtype == np.uint16 or b.dtype == np.uint8):
        return [(a / b).astype(np.complex64)]
    raise NotImplementedError(f"Unsupported divide dtype combination: {a.dtype} / {b.dtype}")


def _op_equal(inputs, settings):
    return [np.equal(inputs[0], inputs[1]).astype(np.uint8)]


def _op_zeros(inputs, settings):
    shape = tuple(int(d) for d in settings["shape"])
    dtype_name = settings.get("dtype", "F32")
    return [np.zeros(shape, dtype=_DTYPE_MAP[dtype_name])]


def _op_reshape(inputs, settings):
    x = inputs[0]
    shape = [int(d) for d in settings["shape"]]
    return [np.reshape(x, shape)]


def _op_transpose(inputs, settings):
    x = inputs[0]
    axes = settings.get("axes")
    if axes is None:
        return [np.transpose(x)]
    return [np.transpose(x, axes=tuple(int(a) for a in axes))]


def _op_where(inputs, settings):
    cond, x, y = inputs
    return [np.where(cond != 0, x, y)]

def _op_min(inputs, settings):
    x = inputs[0]
    axis = settings.get("axis", None)
    keepdims = settings.get("keepdims", False)
    return [np.min(x, axis=axis, keepdims=keepdims)]


def _op_max(inputs, settings):
    x = inputs[0]
    axis = settings.get("axis", None)
    keepdims = settings.get("keepdims", False)
    return [np.max(x, axis=axis, keepdims=keepdims)]


def _op_mean(inputs, settings):
    x = inputs[0]
    axis = settings.get("axis", None)
    keepdims = settings.get("keepdims", False)
    result = np.mean(x, axis=axis, keepdims=keepdims)
    if x.dtype == np.complex64:
        return [result.astype(np.complex64)]
    return [result.astype(np.float32)]


def _op_meshgrid(inputs, settings):
    indexing = settings.get("indexing", "xy")
    copy = settings.get("copy", True)
    sparse = settings.get("sparse", False)
    return list(np.meshgrid(*inputs, indexing=indexing, copy=copy, sparse=sparse))


def _op_slice(inputs, settings):
    x = inputs[0]
    slices = []
    for item in settings["slices"]:
        if isinstance(item, int):
            slices.append(item)
        else:
            start = item.get("start", None)
            stop = item.get("stop", None)
            step = item.get("step", 1)
            slices.append(slice(start, stop, step))
    return [x[tuple(slices)]]

def _numpy_norm(settings):
    norm = settings.get("norm", "backward")
    if isinstance(norm, str):
        return norm
    # Defensive fallback in case enums arrive as integers.
    if norm == 0:
        return "backward"
    if norm == 1:
        return "forward"
    if norm == 2:
        return "ortho"
    return "backward"


def _op_fft(inputs, settings):
    x = inputs[0]
    axis = int(settings.get("axis", -1))
    norm = _numpy_norm(settings)
    return [np.fft.fft(x, axis=axis, norm=norm).astype(np.complex64)]


def _op_fft2(inputs, settings):
    x = inputs[0]
    axes = settings.get("axes", None)
    axes = None if axes is None else tuple(int(a) for a in axes)
    norm = _numpy_norm(settings)
    return [np.fft.fft2(x, axes=axes, norm=norm).astype(np.complex64)]


def _op_rfft(inputs, settings):
    x = inputs[0]
    axis = int(settings.get("axis", -1))
    norm = _numpy_norm(settings)
    return [np.fft.rfft(x, axis=axis, norm=norm).astype(np.complex64)]


def _op_rfft2(inputs, settings):
    x = inputs[0]
    axes = settings.get("axes", None)
    axes = None if axes is None else tuple(int(a) for a in axes)
    norm = _numpy_norm(settings)
    return [np.fft.rfft2(x, axes=axes, norm=norm).astype(np.complex64)]


def _op_irfft2(inputs, settings):
    x = inputs[0]
    axes = settings.get("axes", None)
    axes = None if axes is None else tuple(int(a) for a in axes)
    norm = _numpy_norm(settings)
    return [np.fft.irfft2(x, axes=axes, norm=norm).astype(np.float32)]


def _op_fftshift(inputs, settings):
    x = inputs[0]
    axes = settings.get("axes", None)
    axes = None if axes is None else tuple(int(a) for a in axes)
    return [np.fft.fftshift(x, axes=axes)]


_DISPATCH = {
    "abs":    _op_abs,
    "add":    _op_add,
    "argmax": _op_argmax,
    "arange": _op_arange,
    "asarray": _op_asarray,
    "ascontiguousarray": _op_ascontiguousarray,
    "conj": _op_conj,
    "concatenate": _op_concatenate,
    "copy": _op_copy,
    "subtract": _op_subtract,
    "multiply": _op_multiply,
    "divide": _op_divide,
    "equal": _op_equal,
    "zeros": _op_zeros,
    "reshape": _op_reshape,
    "transpose": _op_transpose,
    "where": _op_where,
    "min": _op_min,
    "max": _op_max,
    "mean": _op_mean,
    "meshgrid": _op_meshgrid,
    "slice": _op_slice,
    "fft": _op_fft,
    "fft2": _op_fft2,
    "rfft": _op_rfft,
    "rfft2": _op_rfft2,
    "irfft2": _op_irfft2,
    "fftshift": _op_fftshift,
}


# -------------------------------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <manifest.json>", file=sys.stderr)
        sys.exit(1)

    manifest_path = Path(sys.argv[1])
    workdir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())

    op = manifest["op"]
    if op not in _DISPATCH:
        print(f"Unknown op: {op!r}", file=sys.stderr)
        sys.exit(1)

    inputs = [_read_input(workdir, d) for d in manifest["inputs"]]
    settings = manifest.get("settings", {})
    outputs = _DISPATCH[op](inputs, settings)
    out_names = manifest["output_payloads"]

    if len(outputs) != len(out_names):
        print(
            f"Op {op!r} produced {len(outputs)} outputs but manifest expects {len(out_names)}",
            file=sys.stderr,
        )
        sys.exit(1)

    for arr, fname in zip(outputs, out_names):
        _write_output(workdir, fname, arr)


if __name__ == "__main__":
    main()
