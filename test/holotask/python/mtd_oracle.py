"""Independent NumPy references for the MTD Holoflow task interfaces."""

import json
import sys
from pathlib import Path

import numpy as np


def window(size, alpha):
    x = np.linspace(0, 1, size)
    w = np.ones(size)
    if alpha:
        left, right = x < alpha / 2, x > 1 - alpha / 2
        w[left] = 0.5 * (1 + np.cos(np.pi * (2 * x[left] / alpha - 1)))
        w[right] = 0.5 * (1 + np.cos(np.pi * (2 * x[right] / alpha - 2 / alpha + 1)))
    return np.outer(w, w)


def field_recovery(images, settings):
    half = settings["half_size"]
    side = 2 * half
    cx, cy = settings["center_x"], settings["center_y"]
    fields, positions = [], []
    for image in images:
        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(
            image.astype(np.float32) * window(image.shape[0], settings["tukey_alpha"]))))
        order = spectrum[cy-half:cy+half, cx-half:cx+half] * window(side, settings["tukey_alpha"])
        py, px = np.unravel_index(np.abs(order).argmax(), order.shape)
        field = np.fft.fftshift(np.fft.ifft2(np.roll(order, (-py, -px), (0, 1)))) * side**2
        fields.append(field)
        positions.append((px, py))
    return np.asarray(fields, np.complex64), np.asarray(positions, np.float32)


def volkov(field, settings):
    side = field.shape[0]
    mirror = np.block([[field[::-1, ::-1], field[::-1]], [field[:, ::-1], field]])
    size = 2 * side
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    spectrum = np.fft.fft2(mirror)
    gx = np.fft.ifft2(2j * np.pi * fx * spectrum)
    gy = np.fft.ifft2(2j * np.pi * fy * spectrum)
    norm = np.abs(mirror)**2
    denominator = (1 + settings["damping_local"]) * norm + settings["damping_global"] * norm.max()
    denominator = np.maximum(denominator, 1e-30)
    gx = (gx * mirror.conj()).imag / denominator
    gy = (gy * mirror.conj()).imag / denominator
    margin = settings["gradient_margin"]
    yy, xx = np.mgrid[:size, :size]
    excluded = ((xx < margin) | (yy < margin) | (xx >= size-margin) | (yy >= size-margin)
                | (np.abs(xx-side) < margin) | (np.abs(yy-side) < margin))
    gx[excluded] = 0
    gy[excluded] = 0
    denominator = 2 * np.pi * (fx**2 + fy**2)
    denominator[0, 0] = 1
    integrated = -1j * (fx * np.fft.fft2(gx) + fy * np.fft.fft2(gy)) / denominator
    integrated[0, 0] = 0
    return np.fft.ifft2(integrated).real[side:, side:]


def rytov(fields, mask, settings):
    side = fields.shape[-1]
    yy, xx = np.mgrid[:side, :side]
    x, y = 2 * xx / (side - 1) - 1, 2 * yy / (side - 1) - 1
    degree, margin, step = (settings[k] for k in ("polynomial_degree", "background_margin", "sample_step"))
    design = np.stack([x**px*y**py for px in range(degree+1) for py in range(degree+1-px)], -1)
    sample = ((mask.reshape(side, side) > settings["mask_threshold"]) & (xx % step == 0)
              & (yy % step == 0) & (xx >= margin) & (yy >= margin)
              & (xx < side-margin) & (yy < side-margin))
    result = []
    for field in fields:
        phase = volkov(field, settings) if settings["unwrap_phase"] else np.angle(field)
        amplitude = np.abs(field).astype(np.float64)
        ca = np.linalg.lstsq(design[sample], amplitude[sample], rcond=None)[0]
        cp = np.linalg.lstsq(design[sample], phase[sample], rcond=None)[0]
        background = np.maximum(design @ ca, max(settings["amplitude_floor"]*amplitude.max(), 1e-30))
        result.append(np.log(np.maximum(amplitude/background, 1e-8)) + 1j*(phase-design@cp))
    return np.asarray(result, np.complex64)


def linear(fields, positions, settings):
    side, dim = fields.shape[-1], settings["dimension"]
    half = side // 2
    wavelength = settings["wavelength"]
    radius = settings["hologram_size"]*settings["pixel_pitch_object"]*settings["medium_index"]/wavelength
    dx = settings["pixel_pitch_object"]*settings["hologram_size"]/side
    k0 = 2*np.pi/wavelength*settings["medium_index"]
    spectrum_volume = np.zeros((dim, dim, dim), np.complex64)
    coverage = np.zeros((dim, dim, dim), np.float32)
    fy, fx = np.mgrid[-half:half, -half:half]
    pupil = (fx**2+fy**2 < half**2) & (radius**2-fx**2-fy**2 >= 0)
    fx, fy = fx[pupil], fy[pupil]
    rz = np.sqrt(radius**2-fx**2-fy**2)
    for field, (px, py) in zip(fields, positions):
        px, py = int(px), int(py)
        base2 = radius**2-(px-half)**2-(py-half)**2
        if base2 <= 0:
            continue
        z = np.rint(rz-np.sqrt(base2)).astype(int)+dim//2
        x, y = dim//2+fx-(px-half), dim//2+fy-(py-half)
        valid = (x >= 0) & (x < dim) & (y >= 0) & (y < dim) & (z >= 0) & (z < dim)
        spectrum = np.fft.fft2(np.roll(field*window(side, settings["tukey_alpha"]), (half, half), (0, 1)))
        values = np.roll(spectrum, (py, px), (0, 1))[pupil]
        weights = (-2j*k0*(rz/radius)*dx**2*values).astype(np.complex64)
        np.add.at(spectrum_volume, (z[valid], y[valid], x[valid]), weights[valid])
        np.add.at(coverage, (z[valid], y[valid], x[valid]), 1)
    used = coverage > 0
    spectrum_volume[used] /= coverage[used]
    spatial = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(spectrum_volume)))
    scale = -dim**3/(dx*side)**3/(2*settings["background_index"]*(2*np.pi/wavelength)**2)
    return (spatial*scale).astype(np.complex64), coverage


def gps(volume, coverage, settings):
    def forward(array):
        return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(array)))

    def inverse(array):
        return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(array)))

    def constrain(array):
        return (np.clip(array.real, settings["delta_n_min"], settings["delta_n_max"])
                + 1j*np.clip(array.imag, settings["kappa_min"], settings["kappa_max"])).astype(np.complex64)

    measured = forward(volume)
    current = volume.copy()
    for _ in range(settings["iterations"]):
        spectrum = forward(constrain(current))
        spectrum[coverage > 0] = measured[coverage > 0]
        current = inverse(spectrum).astype(np.complex64)
    return constrain(current)


def main(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    dtypes = {"U8": np.uint8, "U16": np.uint16, "F32": np.float32, "CF32": np.complex64}
    inputs = [np.fromfile(path.parent/item["payload"], dtype=dtypes[item["dtype"]]).reshape(item["shape"])
              for item in manifest["inputs"]]
    s, op = manifest["settings"], manifest["op"]
    if op == "field":
        outputs = field_recovery(inputs[0], s)
    elif op == "rytov":
        outputs = [rytov(*inputs, s)]
    elif op == "linear":
        outputs = linear(*inputs, s)
    elif op == "gps":
        outputs = [gps(*inputs, s)]
    elif op == "slice":
        result = np.take(inputs[0], s["index"], axis=s["axis"])
        outputs = [{"real": result.real, "imaginary": result.imag, "magnitude": np.abs(result)}[s["component"]].astype(np.float32)]
    elif op == "load_npy":
        outputs = [np.load(s["path"])]
    elif op == "pipeline":
        fields, positions = field_recovery(inputs[0], s["recovery"])
        volume, coverage = linear(rytov(fields, inputs[1], s["rytov"]), positions, s["linear"])
        outputs = [gps(volume, coverage, s["gps"]), fields, volume]
    else:
        raise ValueError(op)
    for output, filename in zip(outputs, manifest["output_payloads"]):
        np.ascontiguousarray(output).tofile(path.parent/filename)


if __name__ == "__main__":
    main(sys.argv[1])
