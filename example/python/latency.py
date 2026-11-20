# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Display source-to-output latency for 16-frame processing windows."""

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
        frames = hf.sources.holofile(args.input, batch_size=32, loop=args.loop, tail="drop")
        frames = hf.latency_start(frames, node_id="input")
        windows = hf.batch_queue(frames.to("device"), 128, 16, 16)
        result = hf.mean(windows.astype(hf.float32), axis=0)
        result = hf.latency_end(result, source_node_id="input", node_id="latency",
                                window_frames=16, stride_frames=16,
                                interval_ms=args.interval_ms)
        hf.sinks.void(result)

    def display(run):
        while (event := run.receive_event()) is not None:
            if event.node_id != "latency":
                continue
            data = event.data
            final = data["type"] == "latency_summary"
            label = "Total" if final else "Interval"
            line = (f"{label}: {data['matched_outputs']:,} matched, "
                    f"{data['unmatched_outputs']:,} unmatched")
            if data["newest_ms"] is not None:
                line += (f" | newest {data['newest_ms']['mean']:.2f} ms"
                         f" | oldest {data['oldest_ms']['mean']:.2f} ms")
            print("\r" + line.ljust(100), end="\n" if final else "", flush=True)

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
