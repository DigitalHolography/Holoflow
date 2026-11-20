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

"""Benchmark a finite or looping input-to-device pipeline."""

import argparse
import json

import holoflow as hf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--mode", choices=("finite", "looping"), default="finite")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--log-dir", help="Enable graph dumps and Perfetto in a unique child folder")
    args = parser.parse_args()

    def pipeline():
        frames = hf.sources.holofile(args.input, batch_size=args.batch_size,
                                    loop=args.mode == "looping", node_id="input")
        hf.sinks.void(frames.to("device"), node_id="output")

    report = hf.benchmark(pipeline, mode=args.mode, warmup_seconds=args.warmup,
                          duration_seconds=args.duration, log_dir=args.log_dir)
    print(json.dumps(report.to_json(), indent=2))
    if report.artifacts is not None:
        print(f"Artifacts: {report.artifacts.directory}")


if __name__ == "__main__":
    main()
