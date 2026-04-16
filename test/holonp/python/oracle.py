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
    arr = np.frombuffer(raw, dtype=np_dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr.copy()


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


_DISPATCH = {
    "abs":    _op_abs,
    "add":    _op_add,
    "argmax": _op_argmax,
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
