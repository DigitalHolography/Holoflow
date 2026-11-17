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

"""Read a file once and return the processed frames in a NumPy array.

Usage: python example/python/finite_numpy.py input.holo
Requires NumPy and the CMake-built holoflow package on PYTHONPATH.
"""
import argparse
import holoflow as hf


def pipeline(path):
    frames = hf.sources.holofile(path, batch_size=32, loop=False, tail="error")
    values = frames.to("device").astype(hf.float32)
    return hf.sinks.numpy(values, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    args = parser.parse_args()
    with hf.trace(pipeline, args.input).compile() as run:
        run.start()
        images = run.result()
    print(f"Collected {images.shape}, dtype={images.dtype}")
    print(f"Range: {images.min()}..{images.max()}" if images.size else "Empty result")


if __name__ == "__main__":
    main()
