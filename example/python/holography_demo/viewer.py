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

"""Offline CAO dashboard for the holography demo."""


def create_viewer(results, *, xy_offset, discard_first, zernike_modes):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    frames = results["xy"]
    if len(frames) == 0:
        raise ValueError("No complete XY averaging windows to display")
    diagnostics = results["coefficients"]
    if any(len(results[key]) != len(diagnostics) for key in ("phase", "shack_hartmann")):
        raise ValueError("Diagnostic streams must have matching update counts")
    if len(diagnostics) < len(frames) + xy_offset:
        raise ValueError("Diagnostic streams do not cover the XY averaging windows")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    xy_ax, sh_ax, phase_ax, coeff_ax = axes.flat
    xy_image = xy_ax.imshow(frames[0], cmap="gray", **(
        dict(vmin=0, vmax=255) if frames.dtype == np.uint8 else {}))
    sh_image = sh_ax.imshow(results["shack_hartmann"][xy_offset], cmap="gray", vmin=0, vmax=255)
    phase_image = phase_ax.imshow(results["phase"][xy_offset], cmap="twilight", vmin=0, vmax=2*np.pi)
    fig.colorbar(phase_image, ax=phase_ax, label="Wrapped phase (rad)")
    # Exclude incomplete causal-average startup from both curves and autoscaling.
    # Keep absolute update indices so the XY slider still selects the correct sample.
    updates = np.arange(discard_first, len(diagnostics))
    for column, noll in enumerate(zernike_modes):
        coeff_ax.plot(updates, diagnostics[discard_first:, column], label=f"a{noll}")
    marker = coeff_ax.axvline(xy_offset, color="black")
    coeff_ax.set(xlabel="Temporal-window update", ylabel="Phase coefficient (rad)")
    coeff_ax.legend()
    def update(value):
        i = int(value)
        sample = i + xy_offset
        xy_image.set_data(frames[i])
        if frames.dtype != np.uint8:
            xy_image.autoscale()
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
