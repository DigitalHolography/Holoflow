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

"""Holoflow processing regression fixture."""
import holoflow as hf

WAVELENGTH = 852e-9
PROPAGATION_DISTANCE = 0.492
PIXEL_PITCH = 20e-6
TIME_WINDOW = 32
COMPONENT_BEGIN = 0
COMPONENT_END = 16
AVERAGE_WINDOW = 64
INPUT_CAPACITY = 1024
PROCESSING_CAPACITY = 8
OUTPUT_CAPACITY = 32
FILTER_R_INNER = 0
FILTER_R_OUTER = 1024
FILTER_S_INNER = 3
FILTER_S_OUTER = 1
FLATFIELD_CUTOFF_UM = 10000.0


def input_range(total_frames, start=0, output_frames=None, end=None):
    if start < 0 or start >= total_frames:
        raise ValueError("start-frame must lie inside the input file")
    if end is not None and output_frames is not None:
        raise ValueError("Choose output-frames or end-frame, not both")
    if end is None:
        count = 1024 if output_frames is None else output_frames
        if count <= 0:
            raise ValueError("output-frames must be positive")
        end = start + (count + AVERAGE_WINDOW - 1) * TIME_WINDOW
    if end <= start or end > total_frames:
        raise ValueError(f"Selected range [{start}, {end}) exceeds the file or is empty (file has {total_frames} frames)")
    if (end - start) % TIME_WINDOW:
        raise ValueError(f"Input range length must be divisible by {TIME_WINDOW}")
    count = max(0, (end - start) // TIME_WINDOW - AVERAGE_WINDOW + 1)
    return start, end, count


def pipeline(path, start, end):
    x = hf.sources.holofile(path, batch_size=TIME_WINDOW, load_kind="live",
                           start_frame=start, end_frame=end, loop=False, tail="error")
    x = x.to("device")
    x = hf.batch_queue(x, INPUT_CAPACITY, TIME_WINDOW, TIME_WINDOW)
    x = x.astype(hf.float32)
    x = hf.pca(x, COMPONENT_BEGIN, COMPONENT_END)
    x = hf.filter2d(x, FILTER_R_INNER, FILTER_R_OUTER, FILTER_S_INNER, FILTER_S_OUTER)
    components = COMPONENT_END - COMPONENT_BEGIN
    field = hf.batch_queue(x, components * PROCESSING_CAPACITY, components, components)
    magnitude = hf.fresnel_diffraction(
        field,
        WAVELENGTH,
        PIXEL_PITCH,
        PIXEL_PITCH,
        PROPAGATION_DISTANCE,
        (-2, -1),
        True,
    )
    image = hf.mean(magnitude, -3, True)
    image = hf.fftshift(image)
    height, width = image.shape[-2:]
    dy = WAVELENGTH * abs(PROPAGATION_DISTANCE) / (height * PIXEL_PITCH)
    dx = WAVELENGTH * abs(PROPAGATION_DISTANCE) / (width * PIXEL_PITCH)
    cutoff_m = FLATFIELD_CUTOFF_UM * 1e-6
    image = hf.flatfield(image, sigma_y=0.187 * cutoff_m / dy,
                        sigma_x=0.187 * cutoff_m / dx)
    image = hf.sliding_average(image, PROCESSING_CAPACITY, AVERAGE_WINDOW)
    image = image.astype(hf.uint8, "scaled")
    image = hf.batch_queue(image, OUTPUT_CAPACITY, 1, 1)
    return hf.sinks.numpy(image, axis=0, node_id="output")


def collect_frames(path, start, end):
    with hf.trace(pipeline, path, start, end).compile() as run:
        run.start()
        return run.result()


def create_viewer(frames):
    """Return (figure, slider); processing and window display are separate."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    if frames.ndim != 3 or len(frames) == 0:
        raise ValueError("Viewer requires a nonempty (N, H, W) array")
    fig, ax = plt.subplots()
    image = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255,
                      interpolation="nearest", origin="upper")
    ax.set_title(f"Frame 1 / {len(frames)}")
    slider = None
    if len(frames) > 1:
        fig.subplots_adjust(bottom=0.22)
        slider = Slider(fig.add_axes([0.2, 0.08, 0.6, 0.04]), "Frame", 0,
                        len(frames) - 1, valinit=0, valstep=1)
        def update(value):
            index = int(value)
            image.set_data(frames[index])
            ax.set_title(f"Frame {index + 1} / {len(frames)}")
            fig.canvas.draw_idle()
        slider.on_changed(update)
        def key(event):
            if event.key in ("left", "right"):
                step = 1 if event.key == "right" else -1
                slider.set_val(max(0, min(len(frames) - 1, int(slider.val) + step)))
        fig.canvas.mpl_connect("key_press_event", key)
    return fig, slider
