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

"""Typed wrappers mapping Python settings to native factory settings."""

from ._graph import Tensor, operation


def batch_queue(x: Tensor, target_capacity: int, output_size: int,
                output_stride: int, *, tail: str = "error", node_id: str | None = None) -> Tensor:
    return operation("BatchQueue", (x,), {
        "target_capacity": target_capacity, "output_size": output_size,
        "output_stride": output_stride, "tail": tail,
    }, node_id)


def pca(x: Tensor, begin: int, end: int, *, node_id: str | None = None) -> Tensor:
    return operation("Pca", (x,), {"begin": begin, "end": end}, node_id)


def fresnel_diffraction(x: Tensor, wavelength: float, dx: float, dy: float,
                        z: float, axes=(-2, -1), output_magnitude: bool = False,
                        skip_phase_shift: bool = True,
                        *, node_id: str | None = None) -> Tensor:
    return operation("FresnelDiffraction", (x,), {
        "lambda": wavelength, "dx": dx, "dy": dy, "z": z, "axes": list(axes),
        "output_magnitude": output_magnitude, "skip_phase_shift": skip_phase_shift,
    }, node_id)


def mean(x: Tensor, axis=(), keepdims: bool = False,
         *, node_id: str | None = None) -> Tensor:
    axes = [axis] if isinstance(axis, int) else list(axis)
    return operation("Mean", (x,), {"axis": axes, "keepdims": keepdims}, node_id)


def sliding_average(x: Tensor, target_capacity: int, window_size: int,
                    discard_first: int = 0, *, node_id: str | None = None) -> Tensor:
    return operation("SlidingAverage", (x,), {
        "target_capacity": target_capacity, "window_size": window_size,
        "discard_first": discard_first,
    }, node_id)


def dual_reader_batch_queue(x, target_capacity, window_size, *, node_id=None):
    return operation("DualReaderBatchQueue", (x,), dict(target_capacity=target_capacity, window_size=window_size), node_id)


def arange(start, stop, step=1, *, dtype=None, device="device", node_id=None):
    from . import _native
    if dtype not in (None, _native.DType.float32):
        raise ValueError("arange currently supports float32")
    if device not in ("host", "device"):
        raise ValueError("device must be host or device")
    return operation("Arange", (), dict(start=start, stop=stop, step=step, dtype="F32", device=device.title()), node_id)


def asarray(value, *, node_id=None):
    value = complex(value)
    return operation("AsArray", (), dict(value=value.real, imag=value.imag, dtype="CF32" if value.imag else "F32", device="Device"), node_id)


def meshgrid(x, y, *, indexing="xy", node_id=None):
    return operation("Meshgrid", (x, y), dict(indexing=indexing), node_id)


def fft2(x, *, axes=(-2, -1), node_id=None):
    return operation("FFT2", (x,), dict(axes=list(axes), inverse=False, norm="backward"), node_id)


def fftshift(x, *, axes=(-2, -1), node_id=None):
    return operation("FFTShiftNp", (x,), dict(axes=list(axes), inverse=False), node_id)


def square(x, *, node_id=None):
    return operation("Square", (x,), {}, node_id)


def exp(x, *, node_id=None):
    return operation("Exp", (x,), {}, node_id)


def abs(x, *, node_id=None):
    return operation("Abs", (x,), {}, node_id)


def add(x, y, *, node_id=None):
    return operation("Add", (x, y), {}, node_id)


def multiply(x, y, *, node_id=None):
    return operation("Multiply", (x, y), {}, node_id)


def filter2d(x: Tensor, r_inner: int, r_outer: int, s_inner: int = 0,
             s_outer: int = 1, *, node_id: str | None = None) -> Tensor:
    """Radial spatial-frequency filter; radii and transition widths are in bins."""
    return operation("Filter2D", (x,), dict(r_inner=r_inner, r_outer=r_outer,
                     s_inner=s_inner, s_outer=s_outer), node_id)


def flatfield(x: Tensor, sigma_y: float, sigma_x: float,
              *, node_id: str | None = None) -> Tensor:
    """Subtract a Gaussian background; sigmas are measured in pixels."""
    return operation("Flatfield", (x,), dict(sigma_y=sigma_y, sigma_x=sigma_x), node_id)


def short_time_fresnel_diffraction(x, wavelength, dx, dy, z, win_h, win_w,
                                   stride_y, stride_x, *, phase_ref="GLOBAL",
                                   axes=(-2, -1), output_magnitude=True,
                                   skip_phase_shift=True, node_id=None):
    return operation("ShortTimeFresnelDiffraction", (x,), {
        "lambda": wavelength, "dx": dx, "dy": dy, "z": z,
        "win_h": win_h, "win_w": win_w, "stride_y": stride_y, "stride_x": stride_x,
        "phase_ref": phase_ref, "axes": list(axes), "output_magnitude": output_magnitude,
        "skip_phase_shift": skip_phase_shift}, node_id)


def causal_sliding_average(x, window_size, *, node_id=None):
    return operation("CausalSlidingAverage", (x,), {"window_size": window_size}, node_id)


def shack_hartmann_slopes(x, wavelength, dx, dy, z, subaperture_height,
                          subaperture_width, stride_y, stride_x, *,
                          mode="full_pairwise", correlation_roi=None,
                          skip_subapertures_outside_pupil=True,
                          output_xcorr_maps=False, pair_batch_size=256, node_id=None):
    roi = dict(angle=0.0, cx=0.5, cy=0.5, rx=0.9, ry=0.9) if correlation_roi is None else correlation_roi
    return operation("ShackHartmannSlopes", (x,), {
        "lambda": wavelength, "dx": dx, "dy": dy, "z": z,
        "subaperture_height": subaperture_height, "subaperture_width": subaperture_width,
        "stride_y": stride_y, "stride_x": stride_x, "mode": mode,
        "correlation_roi": roi, "skip_subapertures_outside_pupil": skip_subapertures_outside_pupil,
        "output_xcorr_maps": output_xcorr_maps, "pair_batch_size": pair_batch_size}, node_id)


def zernike_from_slopes(x, indexes, wavelength, dx, dy, subaperture_height,
                         subaperture_width, stride_y, stride_x, *, ny=1, nx=1,
                         skip_subapertures_outside_pupil=True, node_id=None):
    return operation("ZernikeFromSlopes", (x,), {
        "indexes": list(indexes), "lambda": wavelength, "dx": dx, "dy": dy,
        "subaperture_height": subaperture_height, "subaperture_width": subaperture_width,
        "stride_y": stride_y, "stride_x": stride_x, "ny": ny, "nx": nx,
        "skip_subapertures_outside_pupil": skip_subapertures_outside_pupil}, node_id)


def zernike_phase(coefficients, indexes, ny, nx, *, device="device", node_id=None):
    if device not in ("host", "device"):
        raise ValueError("device must be host or device")
    return operation("ZernikePhase", (coefficients,), {
        "indexes": list(indexes), "ny": ny, "nx": nx, "output": device.title()}, node_id)


def correct_phase(x, phase, *, node_id=None):
    return operation("CorrectPhase", (x, phase), {}, node_id)


def normalize(x, axes=(-2, -1), lo=0.0, hi=255.0, *, node_id=None):
    return operation("Normalize", (x,), {"axes": list(axes), "lo": lo, "hi": hi}, node_id)


def transpose(x, axes, *, node_id=None):
    return operation("Transpose", (x,), {"axes": list(axes)}, node_id)


def copy(x, *, node_id=None):
    return operation("Copy", (x,), {}, node_id)


def wrap2pi(x, *, node_id=None):
    return operation("Wrap2Pi", (x,), {}, node_id)
