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

"""Display live throughput from a finite or looping Holofile stream."""

import argparse
import time

import holoflow as hf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--loop", action="store_true", help="Loop until Ctrl+C")
    parser.add_argument("--interval-ms", type=int, default=1000)
    args = parser.parse_args()

    def pipeline():
        frames = hf.sources.holofile(args.input, load_kind="cpu_cached",loop=args.loop)
        frames = hf.throughput(frames, interval_ms=args.interval_ms, node_id="input_fps")
        hf.sinks.void(frames.to("device"))

    def display(run):
        while (event := run.receive_event()) is not None:
            if event.node_id != "input_fps":
                continue
            data = event.data
            if data["type"] == "throughput":
                print(f"\r{data['fps']:,.1f} FPS  |  {data['total_frames']:,} frames".ljust(60),
                      end="", flush=True)
            elif data["type"] == "throughput_summary":
                summary = f"{data['total_frames']:,} frames in {data['elapsed_seconds']:.2f} s"
                if "average_fps" in data:
                    summary += f"  |  {data['average_fps']:,.1f} FPS average"
                print(f"\r{summary}".ljust(60))

    with hf.trace(pipeline).compile() as run:
        run.start()
        try:
            while not run.completed():
                display(run)
                time.sleep(0.01)
        except KeyboardInterrupt:
            run.stop()
        run.wait()
        display(run)


if __name__ == "__main__":
    main()
