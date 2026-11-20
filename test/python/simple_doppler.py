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


def simple_doppler(
    input_path,
    output_path,
    *,
    time_window=256,
    component_begin=4,
    component_end=64,
    accumulation=4,
):
    frames = hf.sources.holofile(
        input_path,
        batch_size=32,
        load_kind="live",
    )

    frames = frames.to("device")
    frames = hf.batch_queue(
        frames,
        target_capacity=1024,
        output_size=time_window,
        output_stride=time_window,
    )
    frames = frames.astype(hf.float32)  # [T, H, W]

    # Batch four temporal windows together, as in the C++ pipeline.
    t, h, w = frames.shape
    frames = frames.reshape((1, t, h, w))
    frames = hf.batch_queue(
        frames,
        target_capacity=8,
        output_size=4,
        output_stride=4,
    )  # [4, T, H, W]

    components = hf.pca(
        frames,
        begin=component_begin,
        end=component_end,
    )  # [4, component_end - component_begin, H, W]

    magnitude = hf.fresnel_diffraction(
        components,
        wavelength=532e-9,
        dx=12e-6,
        dy=12e-6,
        z=0.20,
        axes=(-2, -1),
        output_magnitude=True,
    )

    # Match the C++ M0 path: mean magnitude, then mean across windows.
    image = hf.mean(magnitude, axis=-3, keepdims=True)
    image = hf.mean(image, axis=0)  # [1, H, W]
    image = hf.sliding_average(
        image,
        target_capacity=8,
        window_size=accumulation,
        discard_first=0,
    )

    image = image.astype(hf.uint8, strategy="scaled").to("host")
    image = hf.batch_queue(
        image,
        target_capacity=8,
        output_size=1,
        output_stride=1,
    )
    hf.sinks.holofile(
        image,
        output_path,
        count=100,
        node_id="doppler_writer",
    )
