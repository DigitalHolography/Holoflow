"""Typed wrappers mapping Python settings to native factory settings."""

from ._graph import Tensor, operation


def batch_queue(x: Tensor, target_capacity: int, output_size: int,
                output_stride: int, *, node_id: str | None = None) -> Tensor:
    return operation("BatchQueue", (x,), {
        "target_capacity": target_capacity, "output_size": output_size,
        "output_stride": output_stride,
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
