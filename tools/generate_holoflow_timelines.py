"""Generate the illustrative CPU/GPU timelines used by the Holoflow introduction.

The timings are synthetic. Run this script from any directory with Matplotlib available:

    python tools/generate_holoflow_timelines.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "doc"
    / "mkdocs"
    / "docs"
    / "assets"
    / "images"
)

FRAME_COUNT = 4
TIME_LIMIT_US = 5000
TIME_TICKS_US = range(0, TIME_LIMIT_US, 500)

DURATIONS_US = {
    "load": 80,
    "h2d": 200,
    "rfft2": 280,
    "multiply": 80,
    "irfft2": 280,
    "d2h": 200,
    "save": 80,
    "queue": 20,
}

COLORS = {
    "transfer_in": "#4ade80",
    "transfer_out": "#f87171",
    "compute": "#60a5fa",
    "synchronization": "#d1d5db",
}

LEGEND = [
    ("transfer_in", "H2D"),
    ("transfer_out", "D2H"),
    ("compute", "Kernel"),
    ("synchronization", "Sync"),
]


@dataclass(frozen=True)
class Operation:
    start: float
    end: float
    category: str


@dataclass(frozen=True)
class Lane:
    label: str
    operations: tuple[Operation, ...]


def operation(start: float, duration: float, category: str) -> Operation:
    return Operation(start, start + duration, category)


def baseline_lanes() -> tuple[list[Lane], list[Lane]]:
    gpu: list[Operation] = []
    cpu: list[Operation] = []
    period = 1240

    for frame in range(FRAME_COUNT):
        start = frame * period
        gpu_start = start + 100
        rfft_start = gpu_start + DURATIONS_US["h2d"]
        multiply_start = rfft_start + DURATIONS_US["rfft2"]
        irfft_start = multiply_start + DURATIONS_US["multiply"]
        d2h_start = irfft_start + DURATIONS_US["irfft2"]
        gpu_end = d2h_start + DURATIONS_US["d2h"]

        cpu.extend(
            [
                operation(start, 20, "transfer_in"),
                operation(start + 20, 20, "compute"),
                operation(start + 40, 20, "compute"),
                operation(start + 60, 20, "compute"),
                operation(start + 80, 20, "transfer_out"),
                operation(start + 100, gpu_end - (start + 100), "synchronization"),
            ]
        )
        gpu.extend(
            [
                operation(gpu_start, DURATIONS_US["h2d"], "transfer_in"),
                operation(rfft_start, DURATIONS_US["rfft2"], "compute"),
                operation(multiply_start, DURATIONS_US["multiply"], "compute"),
                operation(irfft_start, DURATIONS_US["irfft2"], "compute"),
                operation(d2h_start, DURATIONS_US["d2h"], "transfer_out"),
            ]
        )

    return [Lane("S0", tuple(gpu))], [Lane("S0", tuple(cpu))]


def queued_lanes() -> tuple[list[Lane], list[Lane]]:
    cpu_operations = [[], [], []]
    gpu_operations = [[], [], []]

    stage_0_starts = [0, 320, 640, 960]
    compute_starts = [360, 1020, 1680, 2340]

    for frame in range(FRAME_COUNT):
        stage_0 = stage_0_starts[frame]
        h2d_start = stage_0 + 100
        h2d_end = h2d_start + DURATIONS_US["h2d"]
        cpu_operations[0].extend(
            [
                operation(stage_0, 20, "transfer_in"),
                operation(stage_0 + 20, h2d_end - (stage_0 + 20), "synchronization"),
            ]
        )
        gpu_operations[0].append(operation(h2d_start, DURATIONS_US["h2d"], "transfer_in"))

        compute_start = compute_starts[frame]
        rfft_start = compute_start
        multiply_start = rfft_start + DURATIONS_US["rfft2"]
        irfft_start = multiply_start + DURATIONS_US["multiply"]
        compute_end = irfft_start + DURATIONS_US["irfft2"]
        cpu_operations[1].extend(
            [
                operation(compute_start - 60, 20, "compute"),
                operation(compute_start - 40, 20, "compute"),
                operation(compute_start - 20, 20, "compute"),
                operation(compute_start, compute_end - compute_start, "synchronization"),
            ]
        )
        gpu_operations[1].extend(
            [
                operation(rfft_start, DURATIONS_US["rfft2"], "compute"),
                operation(multiply_start, DURATIONS_US["multiply"], "compute"),
                operation(irfft_start, DURATIONS_US["irfft2"], "compute"),
            ]
        )

        queue_read = compute_end + DURATIONS_US["queue"]
        d2h_start = queue_read + 20
        d2h_end = d2h_start + DURATIONS_US["d2h"]
        cpu_operations[2].extend(
            [
                operation(queue_read, 20, "transfer_out"),
                operation(d2h_start, DURATIONS_US["d2h"], "synchronization"),
            ]
        )
        gpu_operations[2].append(operation(d2h_start, DURATIONS_US["d2h"], "transfer_out"))

    gpu = [Lane(f"S{index}", tuple(items)) for index, items in enumerate(gpu_operations)]
    cpu = [Lane(f"S{index}", tuple(items)) for index, items in enumerate(cpu_operations)]
    return gpu, cpu


def draw_operation(axis: plt.Axes, y: float, item: Operation) -> None:
    height = 0.235
    patch = FancyBboxPatch(
        (item.start, y - height / 2),
        item.end - item.start,
        height,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=COLORS[item.category],
        edgecolor="#6b7280",
        linewidth=0.45,
        alpha=0.95,
        zorder=2,
    )
    axis.add_patch(patch)


def draw_time_axis(axis: plt.Axes, y: float) -> None:
    axis.hlines(y, 0, TIME_LIMIT_US, colors="#6b7280", linewidth=0.65)
    for tick in TIME_TICKS_US:
        axis.vlines(tick, y - 0.045, y + 0.045, colors="#6b7280", linewidth=0.65)
        axis.text(tick, y - 0.12, str(tick), ha="center", va="top", fontsize=8)


def draw_timeline(
    gpu_lanes: list[Lane],
    cpu_lanes: list[Lane],
    output_path: Path,
) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "svg.fonttype": "none",
            "svg.hashsalt": "holoflow-timeline",
        }
    )

    lane_step = 0.38
    group_gap = 0.40
    axis_y = 0.42
    cpu_y = [axis_y + 0.52 + index * lane_step for index in range(len(cpu_lanes))]
    gpu_start = cpu_y[-1] + group_gap + lane_step
    gpu_y = [gpu_start + index * lane_step for index in range(len(gpu_lanes))]
    top = gpu_y[-1] + 0.28

    figure_height = 1.9 if len(gpu_lanes) == 1 else 2.9
    figure, axis = plt.subplots(figsize=(11.0, figure_height))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    all_lanes = list(zip(gpu_lanes, gpu_y)) + list(zip(cpu_lanes, cpu_y))
    for lane, y in all_lanes:
        axis.hlines(y, 0, TIME_LIMIT_US, colors="#e5e7eb", linewidth=0.55, zorder=0)
        axis.text(-55, y, lane.label, ha="right", va="center", fontsize=8, color="#4b5563")
        for item in lane.operations:
            draw_operation(axis, y, item)

    for tick in TIME_TICKS_US:
        axis.vlines(tick, axis_y + 0.08, top, colors="#f0f1f3", linewidth=0.5, zorder=-1)

    axis.text(
        -245,
        sum(gpu_y) / len(gpu_y),
        "GPU",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
        fontweight="bold",
    )
    axis.text(
        -245,
        sum(cpu_y) / len(cpu_y),
        "CPU",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
        fontweight="bold",
    )

    draw_time_axis(axis, axis_y)
    axis.text(
        -245,
        axis_y,
        "time (µs)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8,
    )

    handles = [
        Patch(facecolor=COLORS[key], edgecolor="#6b7280", linewidth=0.45, label=label)
        for key, label in LEGEND
    ]
    axis.legend(
        handles=handles,
        ncol=4,
        loc="lower left",
        bbox_to_anchor=(0, -0.20, TIME_LIMIT_US, 0.20),
        bbox_transform=axis.transData,
        mode="expand",
        frameon=False,
        fontsize=8,
        handlelength=1.25,
        handletextpad=0.4,
        borderaxespad=0,
    )

    axis.set_xlim(-310, TIME_LIMIT_US + 40)
    axis.set_ylim(-0.28, top)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", metadata={"Date": None})
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the two SVG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_gpu, baseline_cpu = baseline_lanes()
    queued_gpu, queued_cpu = queued_lanes()
    draw_timeline(
        baseline_gpu,
        baseline_cpu,
        args.output_dir / "holoflow-bandpass-timeline.svg",
    )
    draw_timeline(
        queued_gpu,
        queued_cpu,
        args.output_dir / "holoflow-bandpass-queued-timeline.svg",
    )


if __name__ == "__main__":
    main()
