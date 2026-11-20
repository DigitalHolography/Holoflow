# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Run a YAML-configured holography pipeline with live performance metrics."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
import sys
import time

import holoflow as hf
import yaml

from holography_demo_config import load_config
from holography_demo_graph import show_graph_report
from holography_demo_pipeline import pipeline


class TerminalMetrics:
    def __init__(self):
        self.live = {}
        self.final = {}

    def receive(self, run):
        changed = False
        while (event := run.receive_event()) is not None:
            kind = event.data["type"]
            if kind in ("throughput", "latency"):
                self.live[event.node_id] = event.data
                changed = True
            elif kind in ("throughput_summary", "latency_summary"):
                self.final[event.node_id] = event.data
        if changed and sys.stdout.isatty():
            rates = [self.live.get(node, {}).get("fps", 0.0) for node in ("input_fps", "output_fps")]
            latency = self.live.get("latency", {})
            ages = " | ".join(f"{key} {latency[key + '_ms']['mean']:.2f} ms"
                              for key in ("newest", "oldest") if latency.get(key + "_ms"))
            print(f"\rInput {rates[0]:,.1f} FPS | Output {rates[1]:,.1f} images/s | {ages}".ljust(105),
                  end="", flush=True)

    def summary(self, elapsed):
        if sys.stdout.isatty():
            print("\r" + " " * 105 + "\r", end="")
        for node, label, unit in (("input_fps", "Input", "frames/s"),
                                  ("output_fps", "Output", "images/s")):
            data = self.final[node]
            count = data["total_frames"]
            rate = count / data["elapsed_seconds"] if data["elapsed_seconds"] > 0 else 0.0
            print(f"{label}: {count:,} in {data['elapsed_seconds']:.3f} s ({rate:,.1f} {unit})")
        data = self.final["latency"]
        print(f"Latency: {data['matched_outputs']:,} matched, {data['unmatched_outputs']:,} unmatched")
        for key in ("newest_ms", "oldest_ms"):
            if (stats := data[key]) is not None:
                details = ", ".join(f"{name} {stats[name]:.3f}" for name in ("latest", "min", "mean", "max"))
                print(f"  {key.removesuffix('_ms')} frame (ms): {details}")
        print(f"Trace-to-drain wall time: {elapsed:.3f} s")


def _run_pipeline(config, stack, previous=None):
    metrics = TerminalMetrics()
    started = time.perf_counter()
    graph = hf.trace(pipeline, config)
    artifacts = hf.Artifacts.create(config.graph["log_dir"]) if config.graph["enabled"] else None
    run = stack.enter_context(graph._compile(artifacts, capture=False, previous=previous))
    run.start()
    duration = config.input["duration_seconds"]
    deadline = None if duration is None else time.monotonic() + duration
    try:
        while not run.completed():
            metrics.receive(run)
            if deadline is not None and time.monotonic() >= deadline:
                run.stop()
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        run.stop()
    run.wait()
    elapsed = time.perf_counter() - started
    metrics.receive(run)
    metrics.summary(elapsed)
    images = run.result() if config.viewer["enabled"] and run.completed() else None
    return images, run


_SOLVERS = {
    "single_reference": ("Single reference", "#808080"),
    "full_pairwise": ("Full-pairwise Laplacian", "black"),
}


def run_demo(config):
    reports = []
    with ExitStack() as stack:
        if config.viewer["cao_compare_modes"]:
            images = {}
            previous = None
            for mode, (label, _) in _SOLVERS.items():
                setup = "graph update" if previous is not None else "fresh compile"
                print(f"{label} ({setup})", flush=True)
                settings = replace(config, aberration={**config.aberration, "slope_mode": mode})
                result, previous = _run_pipeline(settings, stack, previous)
                reports.append((label, previous.artifacts))
                if result is None:
                    images = None
                    break
                images[mode] = result["coefficients"]
        else:
            images, run = _run_pipeline(config, stack)
            reports.append((None, run.artifacts))
    for label, artifacts in reports:
        if artifacts is not None:
            if label is not None:
                print(f"{label} graph", flush=True)
            show_graph_report(artifacts.compiled_dot, config.graph["open_browser"])
    return images


def show_images(images):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    image = ax.imshow(images[0], cmap="gray")

    def update(index):
        frame = images[int(index)]
        image.set_data(frame)
        image.autoscale()
        ax.set_title(f"Image {int(index) + 1} / {len(images)}")
        fig.canvas.draw_idle()

    slider = None
    if len(images) > 1:
        slider = Slider(fig.add_axes((0.2, 0.07, 0.6, 0.03)), "Image",
                        0, len(images) - 1, valstep=1, valinit=0)
        slider.on_changed(update)
    update(0)
    plt.show()


def create_coefficient_viewer(coefficients, modes, discard_first, *,
                              single_reference=None, slope_mode="full_pairwise"):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7), layout="constrained")
    histories = [(slope_mode, coefficients)]
    if single_reference is not None:
        if single_reference.shape != coefficients.shape:
            raise ValueError("CAO comparison requires matching coefficient histories")
        histories.insert(0, ("single_reference", single_reference))
    updates = range(discard_first, len(coefficients))
    for axis, mode in zip(axes, (4, 5, 6)):
        for solver, history in histories:
            label, color = _SOLVERS[solver]
            axis.plot(updates, history[discard_first:, modes.index(mode)],
                      label=label, color=color, linewidth=0.6)
        axis.set_ylabel(f"a{mode} (rad)")
        # if single_reference is not None:
        #     axis.legend()
    axes[-1].set_xlabel("CAO update")
    figure.suptitle("CAO coefficient history")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the YAML configuration")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        images = run_demo(config)
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        parser.error(str(error))
    if images is not None and config.viewer["cao_coefficients_only"]:
        import matplotlib.pyplot as plt

        discard = config.correction_window - 1
        modes = config.aberration["zernike_modes"]
        if config.viewer["cao_compare_modes"]:
            create_coefficient_viewer(images["full_pairwise"], modes, discard,
                                      single_reference=images["single_reference"])
        else:
            create_coefficient_viewer(images["coefficients"], modes, discard,
                                      slope_mode=config.aberration["slope_mode"])
        plt.show()
    elif isinstance(images, dict):
        import matplotlib.pyplot as plt
        from holography_demo_viewer import create_viewer

        discard = config.correction_window - 1
        dashboard = create_viewer(images, xy_offset=discard + config.postprocessing["average_window"] - 1,
                                  discard_first=discard, zernike_modes=config.aberration["zernike_modes"])
        plt.show()
    elif images is not None:
        show_images(images)


if __name__ == "__main__":
    main()
