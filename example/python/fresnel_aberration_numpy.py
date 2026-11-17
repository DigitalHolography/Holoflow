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

"""Finite aberration-corrected Fresnel Doppler pipeline and four-output dashboard."""
import argparse
import math
import holoflow as hf
from holoflow import _native

WAVELENGTH = 851e-9
PROPAGATION_DISTANCE = 0.49
PIXEL_PITCH = 20e-6
TIME_WINDOW = 32
PCA_BATCH = 4
COMPONENT_BEGIN = 0
COMPONENT_END = 16
AVERAGE_WINDOW = 64
DISCARD_FIRST = 63
XY_OFFSET = DISCARD_FIRST + AVERAGE_WINDOW - 1
ZERNIKE_INDEXES = (4, 5, 6)
SUBAPERTURES = 5


def input_range(total_frames, start=0, output_frames=None, end=None):
    if start < 0 or start >= total_frames:
        raise ValueError("start-frame must lie inside the file")
    if end is not None and output_frames is not None:
        raise ValueError("Choose output-frames or end-frame")
    requested = None
    if end is None:
        requested = 1024 if output_frames is None else output_frames
        if requested <= 0:
            raise ValueError("output-frames must be positive")
        windows = requested + XY_OFFSET
        windows = ((windows + PCA_BATCH - 1) // PCA_BATCH) * PCA_BATCH
        end = start + windows * TIME_WINDOW
    if end <= start or end > total_frames:
        raise ValueError(f"Invalid range [{start}, {end}); file has {total_frames} frames")
    if (end - start) % (TIME_WINDOW * PCA_BATCH):
        raise ValueError("Input range length must be divisible by 128 (four PCA windows)")
    count = max(0, (end - start) // TIME_WINDOW - XY_OFFSET)
    return start, end, count if requested is None else requested


def pipeline(path, start, end):
    x = hf.sources.holofile(path, batch_size=32, load_kind="live",
                           start_frame=start, end_frame=end, loop=False, tail="error")
    x = x.to("device")
    x = hf.batch_queue(x, 1024, 32, 32)
    x = x.astype(hf.float32)
    height, width = x.shape[-2:]
    win_h, win_w = height // SUBAPERTURES, width // SUBAPERTURES
    if min(win_h, win_w) < 2:
        raise ValueError("Input must have at least two pixels per subaperture")
    x = x.reshape((1, TIME_WINDOW, height, width), copy=False)
    x = hf.batch_queue(x, 8, PCA_BATCH, PCA_BATCH)
    x = hf.pca(x, COMPONENT_BEGIN, COMPONENT_END)
    x = hf.filter2d(x, 3, 1024, 0, 1)
    current, delayed, valid = hf.dual_reader_batch_queue(x, 8, AVERAGE_WINDOW)

    sensor = hf.short_time_fresnel_diffraction(
        current, WAVELENGTH, PIXEL_PITCH, PIXEL_PITCH, PROPAGATION_DISTANCE,
        win_h, win_w, win_h, win_w, phase_ref="GLOBAL", output_magnitude=True)
    sensor = hf.mean(sensor, axis=1)
    sensor = hf.causal_sliding_average(sensor, AVERAGE_WINDOW)
    sensor = hf.fftshift(sensor)
    # Physical cutoff of 10 mm, sampled on each reconstructed subaperture.
    sensor_dy = WAVELENGTH * abs(PROPAGATION_DISTANCE) / (win_h * PIXEL_PITCH)
    sensor_dx = WAVELENGTH * abs(PROPAGATION_DISTANCE) / (win_w * PIXEL_PITCH)
    sensor = hf.flatfield(sensor, 0.187 * 0.01 / sensor_dy, 0.187 * 0.01 / sensor_dx)
    slopes = hf.shack_hartmann_slopes(
        sensor, WAVELENGTH, PIXEL_PITCH, PIXEL_PITCH, PROPAGATION_DISTANCE,
        win_h, win_w, win_h, win_w, mode="full_pairwise",
        correlation_roi=dict(angle=0.0, cx=0.5, cy=0.5, rx=0.9, ry=0.9),
        skip_subapertures_outside_pupil=True, output_xcorr_maps=False, pair_batch_size=256)
    coefficients = hf.zernike_from_slopes(
        slopes, ZERNIKE_INDEXES, WAVELENGTH, PIXEL_PITCH, PIXEL_PITCH,
        win_h, win_w, win_h, win_w)[0, 0, :]
    phase = hf.zernike_phase(coefficients, ZERNIKE_INDEXES, height, width)
    corrected = hf.correct_phase(delayed, phase)
    corrected = hf.batch_queue(corrected, 2, 1, 1)

    # Full-frame Fresnel propagation, explicitly decomposed as in the target graph.
    xs = hf.arange(-width / 2 * PIXEL_PITCH, (width / 2 - 0.5) * PIXEL_PITCH, PIXEL_PITCH)
    ys = hf.arange(-height / 2 * PIXEL_PITCH, (height / 2 - 0.5) * PIXEL_PITCH, PIXEL_PITCH)
    xx, yy = hf.meshgrid(xs, ys)
    lens_phase = (hf.square(xx) + hf.square(yy)) * hf.asarray(math.pi / (WAVELENGTH * PROPAGATION_DISTANCE))
    lens = hf.exp(lens_phase * hf.asarray(1j))
    magnitude = hf.abs(hf.fft2(corrected * lens, axes=()))
    image = hf.mean(magnitude, axis=-3, keepdims=True)
    image = hf.fftshift(image)
    dy = WAVELENGTH * abs(PROPAGATION_DISTANCE) / (height * PIXEL_PITCH)
    dx = WAVELENGTH * abs(PROPAGATION_DISTANCE) / (width * PIXEL_PITCH)
    image = hf.flatfield(image, 0.187 * 0.01 / dy, 0.187 * 0.01 / dx)
    image = hf.mean(image, axis=0)
    image = hf.sliding_average(image, 8, AVERAGE_WINDOW, discard_first=DISCARD_FIRST)
    image = image.astype(hf.uint8, "scaled")
    image = hf.batch_queue(image, 32, 1, 1)

    mosaic = hf.normalize(sensor, axes=(-2, -1), lo=0, hi=255)
    mosaic = hf.transpose(mosaic, (0, 1, 3, 2, 4))
    mosaic = mosaic.reshape((1, SUBAPERTURES * win_h, SUBAPERTURES * win_w))
    mosaic = mosaic.astype(hf.uint8, "scaled")
    # Wrapped display phase is a private copy; correction retains the unwrapped phase.
    wrapped = hf.wrap2pi(hf.copy(phase)).reshape((1, height, width))
    return {
        "xy": hf.sinks.numpy(image),
        "shack_hartmann": hf.sinks.numpy(mosaic),
        "coefficients": hf.sinks.numpy(coefficients.reshape((1, len(ZERNIKE_INDEXES)))),
        "phase": hf.sinks.numpy(wrapped),
    }


def collect_frames(path, start, end, output_frames=None):
    with hf.trace(pipeline, path, start, end).compile() as run:
        run.start()
        results = run.result()
    if output_frames is not None:
        if len(results["xy"]) < output_frames:
            raise ValueError("Input range cannot produce the requested XY count")
        results["xy"] = results["xy"][:output_frames]
    return results


def create_viewer(results):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    frames = results["xy"]
    if len(frames) == 0:
        raise ValueError("No complete XY averaging windows to display")
    diagnostics = results["coefficients"]
    if any(len(results[key]) != len(diagnostics) for key in ("phase", "shack_hartmann")):
        raise ValueError("Diagnostic streams must have matching update counts")
    if len(diagnostics) < len(frames) + XY_OFFSET:
        raise ValueError("Diagnostic streams do not cover the XY averaging windows")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    xy_ax, sh_ax, phase_ax, coeff_ax = axes.flat
    xy_image = xy_ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255)
    sh_image = sh_ax.imshow(results["shack_hartmann"][XY_OFFSET], cmap="gray", vmin=0, vmax=255)
    phase_image = phase_ax.imshow(results["phase"][XY_OFFSET], cmap="twilight", vmin=0, vmax=2*np.pi)
    fig.colorbar(phase_image, ax=phase_ax, label="Wrapped phase (rad)")
    # Exclude incomplete causal-average startup from both curves and autoscaling.
    # Keep absolute update indices so the XY slider still selects the correct sample.
    updates = np.arange(DISCARD_FIRST, len(diagnostics))
    for column, noll in enumerate(ZERNIKE_INDEXES):
        coeff_ax.plot(updates, diagnostics[DISCARD_FIRST:, column], label=f"a{noll}")
    marker = coeff_ax.axvline(XY_OFFSET, color="black")
    coeff_ax.set(xlabel="Temporal-window update", ylabel="Phase coefficient (rad)")
    coeff_ax.legend()
    def update(value):
        i = int(value)
        sample = i + XY_OFFSET
        xy_image.set_data(frames[i])
        sh_image.set_data(results["shack_hartmann"][sample])
        phase_image.set_data(results["phase"][sample])
        marker.set_xdata([sample, sample])
        xy_ax.set_title(f"XY {i + 1} / {len(frames)}")
        sh_ax.set_title(f"Shack-Hartmann: update {sample}")
        phase_ax.set_title(f"Phase: update {sample}")
        fig.suptitle("Diagnostics at the end of the XY averaging window")
        fig.canvas.draw_idle()
    slider = None
    if len(frames) > 1:
        fig.subplots_adjust(bottom=0.16, hspace=0.4)
        slider = Slider(fig.add_axes([0.2, 0.04, 0.6, 0.03]), "XY frame", 0, len(frames)-1, valinit=0, valstep=1)
        slider.on_changed(update)
        def key(event):
            if event.key in ("left", "right"):
                slider.set_val(max(0, min(len(frames)-1, int(slider.val) + (1 if event.key == "right" else -1))))
        fig.canvas.mpl_connect("key_press_event", key)
    update(0)
    return fig, slider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--start-frame", type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output-frames", type=int)
    group.add_argument("--end-frame", type=int)
    args = parser.parse_args()
    try:
        start, end, count = input_range(_native.file_frame_count(args.input), args.start_frame, args.output_frames, args.end_frame)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    raw_count = max(0, (end-start)//TIME_WINDOW - XY_OFFSET)
    print(f"Input [{start}, {end}); compute {raw_count} XY frames, return {count}", flush=True)
    results = collect_frames(args.input, start, end, count)
    for name, array in results.items():
        print(f"{name}: {array.shape}, {array.dtype}", flush=True)
    if not count:
        print("No complete XY averaging windows; diagnostic arrays are still available through collect_frames().")
        return
    import matplotlib.pyplot as plt
    fig, slider = create_viewer(results)
    plt.show()


if __name__ == "__main__":
    main()
