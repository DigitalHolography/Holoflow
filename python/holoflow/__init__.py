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

"""Trace Python tensor expressions into native streaming computation graphs."""

import os as _os
from pathlib import Path as _Path

# Keep directory handles alive for delayed native DLL loads as well as import.
_dll_handles = []
_dll_libraries = []
if _os.name == "nt":
    _bundled = _Path(__file__).parent / "_dlls"
    if _bundled.is_dir():
        _directories = [_bundled]
    else:
        try:
            from ._build_config import CUDA_BIN as _cuda_bin
        except ModuleNotFoundError:
            _cuda_bin = str(_Path(_os.environ["CUDA_PATH"]) / "bin") if _os.environ.get("CUDA_PATH") else ""
        _directories = [_Path(_cuda_bin), _Path(_cuda_bin) / "x64"] if _cuda_bin else []
    for _directory in _directories:
        if _directory.is_dir():
            _dll_handles.append(_os.add_dll_directory(str(_directory)))
            # NVRTC loads its builtins with LoadLibrary, without Python's search flags.
            # Preload the matching DLL by absolute path rather than modifying PATH.
            import ctypes as _ctypes
            for _builtin in _directory.glob("nvrtc-builtins64_*.dll"):
                _dll_libraries.append(_ctypes.WinDLL(str(_builtin)))

from ._native import DType, ExecutionError, __version__
from ._graph import ArrayResult, CancelledError, Event, Executable, Graph, Tensor, trace
from ._diagnostics import Artifacts
from ._benchmark import BenchmarkReport, benchmark
from ._throughput import throughput
from ._latency import latency_end, latency_start
from ._ops import batch_queue, fresnel_diffraction, mean, pca, sliding_average
from . import sources, sinks

uint8 = DType.uint8
uint16 = DType.uint16
float32 = DType.float32
complex64 = DType.complex64

from ._ops import arange, asarray, meshgrid, square, exp, abs, add, multiply, fft2, fftshift, dual_reader_batch_queue

from ._ops import filter2d, flatfield, pct_clip

from ._ops import (short_time_fresnel_diffraction, causal_sliding_average,
                   shack_hartmann_slopes, zernike_from_slopes, zernike_phase,
                   correct_phase, normalize, transpose, copy, wrap2pi)
