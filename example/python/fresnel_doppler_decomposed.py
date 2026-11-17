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

"""PCA Doppler pipeline with decomposed Fresnel operations."""

import math
import time

import holoflow as hf


INPUT_PATH = "D:\\260312_GUJ_1.holo"
WAVELENGTH = 852e-9
PROPAGATION_DISTANCE = 0.492
PIXEL_PITCH = 20e-6
SOURCE_BATCH_SIZE = 64
TIME_WINDOW = 32
COMPONENT_BEGIN = 0
COMPONENT_END = 16
AVERAGE_WINDOW = 64
INPUT_CAPACITY = 1024
PROCESSING_CAPACITY = 8
OUTPUT_CAPACITY = 32
RUN_SECONDS = 10


def fresnel_diffraction(field):
    h, w = field.shape[-2:]
    size = max(h, w)
    start_x = ((size - w) // 2 - size / 2) * PIXEL_PITCH
    start_y = ((size - h) // 2 - size / 2) * PIXEL_PITCH
    x = hf.arange(start_x, start_x + (w - 0.5) * PIXEL_PITCH, PIXEL_PITCH)
    y = hf.arange(start_y, start_y + (h - 0.5) * PIXEL_PITCH, PIXEL_PITCH)
    x, y = hf.meshgrid(x, y)
    phase = (hf.square(x) + hf.square(y)) * hf.asarray(
        math.pi / (WAVELENGTH * PROPAGATION_DISTANCE)
    )
    lens = hf.exp(phase * hf.asarray(1j))
    return hf.abs(hf.fft2(field * lens))


def pipeline():
    x = hf.sources.holofile(INPUT_PATH, SOURCE_BATCH_SIZE, "cpu_cached")
    x = x.to("device")
    x = hf.batch_queue(x, INPUT_CAPACITY, TIME_WINDOW, TIME_WINDOW)
    x = x.astype(hf.float32)
    x = hf.pca(x, COMPONENT_BEGIN, COMPONENT_END)
    components = COMPONENT_END - COMPONENT_BEGIN
    field = hf.batch_queue(x, components * PROCESSING_CAPACITY, components, components)
    magnitude = fresnel_diffraction(field)
    image = hf.mean(magnitude, -3, True)
    image = hf.fftshift(image)
    image = hf.sliding_average(image, PROCESSING_CAPACITY, AVERAGE_WINDOW)
    image = image.astype(hf.uint8, "scaled")
    image = hf.batch_queue(image, OUTPUT_CAPACITY, 1, 1)
    hf.sinks.void(image, node_id="output")


if __name__ == "__main__":
    with hf.trace(pipeline).compile() as executable:
        executable.start()
        try:
            deadline = time.monotonic() + RUN_SECONDS
            while time.monotonic() < deadline:
                executable.receive_event()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            executable.stop()
            executable.wait()
