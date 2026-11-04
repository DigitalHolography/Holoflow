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

"""Regenerate section CUDA graph figures using Matplotlib and Graphviz.

    python tools/generate_section_cuda_graph_figures.py
    python tools/generate_section_cuda_graph_figures.py --preview-dir build/doc-figures

The CPU/GPU timeline is illustrative, not measured. SVG metadata is deterministic.
"""

from __future__ import annotations

import argparse
from math import lcm
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ASSETS = Path(__file__).resolve().parents[1] / "doc/mkdocs/docs/assets"
STEM = "holoflow-section-cuda-"
INK = "#172033"
MUTED = "#64748b"
BLUE = "#60a5fa"
GREEN = "#86efac"
ORANGE = "#fbbf24"
SVG = "http://www.w3.org/2000/svg"


def save(figure: plt.Figure, name: str, output: Path, preview: Path | None) -> None:
    figure.savefig(output / f"{STEM}{name}.svg", metadata={"Date": None})
    if preview:
        figure.savefig(preview / f"{STEM}{name}.png", dpi=150)
    plt.close(figure)


def block(axis: plt.Axes, x: float, y: float, width: float, label: str,
          color: str, height: float = 0.42) -> None:
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.01,rounding_size=0.055",
        facecolor=color, edgecolor=MUTED, linewidth=0.6,
    ))
    axis.text(x + width / 2, y + height / 2, label, ha="center", va="center",
              fontsize=13, color=INK)


def draw_timeline(output: Path, preview: Path | None) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.8, 5.0))
    figure.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.13, hspace=0.55)
    figure.suptitle("The same GPU work, submitted differently", fontsize=16, color=INK)
    colors = [BLUE, GREEN, ORANGE, "#c4b5fd"]
    titles = ["Ordinary execution · one API submission per kernel",
              "CUDA graph replay · one submission for the captured compute"]
    for panel, axis in enumerate(axes):
        axis.set(xlim=(0, 12), ylim=(-0.15, 1.65))
        axis.set_title(titles[panel], loc="left", fontsize=14, color=INK, pad=12)
        axis.set_yticks([1.21, 0.21], ["CPU / CUDA API", "GPU / stream"])
        axis.tick_params(axis="y", length=0, labelsize=12, colors=INK)
        axis.set_xticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        for y in [0, 1]:
            axis.axhspan(y, y + 0.42, color="#f1f5f9", zorder=0)
        if panel == 0:
            for index, color in enumerate(colors):
                cpu = 0.4 + 2.5 * index
                gpu = 1.45 + 2.5 * index
                block(axis, cpu, 1, 1.05, f"K{index + 1}", "#dbeafe")
                block(axis, gpu, 0, 1.35, f"K{index + 1}", color)
                axis.plot([cpu + 1.05, gpu], [1, 0.42], color=MUTED, lw=0.7, ls="--")
            axis.annotate("submission gap", xy=(3.35, 0.21), xytext=(4.2, 0.72),
                          fontsize=12, color=MUTED,
                          arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8})
        else:
            block(axis, 0.4, 1, 2.55, "cudaGraphLaunch", "#dbeafe")
            for index, color in enumerate(colors):
                block(axis, 3.0 + index * 1.48, 0, 1.35, f"K{index + 1}", color)
            axis.plot([2.95, 3.0], [1, 0.42], color=MUTED, lw=0.7, ls="--")
            axis.text(3.5, 0.73, "less waiting between short kernels", fontsize=12, color=MUTED)
        axis.annotate("", xy=(11.9, -0.12), xytext=(0, -0.12),
                      arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8})
    figure.text(0.97, 0.065, "Time →  shared illustrative scale", ha="right", color=MUTED, fontsize=12)
    figure.text(0.15, 0.02, "Synthetic example · identical kernel durations · not a performance measurement",
                color=MUTED, fontsize=12)
    save(figure, "timeline", output, preview)


def pointer_at(prefix: tuple[int, ...], cycle: tuple[int, ...], step: int) -> int:
    return prefix[step] if step < len(prefix) else cycle[(step - len(prefix)) % len(cycle)]


def worked_example() -> tuple[int, list[tuple[int, int]]]:
    prefix_a, cycle_a = (4, 4), (0, 1, 2, 3)
    prefix_b, cycle_b = (), (0, 1, 2, 3, 4, 5)
    horizon = max(len(prefix_a), len(prefix_b)) + lcm(len(cycle_a), len(cycle_b))
    tuples = [(pointer_at(prefix_a, cycle_a, step), pointer_at(prefix_b, cycle_b, step))
              for step in range(horizon)]
    assert horizon == 14 and len(set(tuples)) == 14
    assert len({(a, b, c) for a, b in tuples for c in range(2)}) == 28
    assert (pointer_at(prefix_a, cycle_a, 14), pointer_at(prefix_b, cycle_b, 14)) == tuples[2]
    return horizon, tuples


def draw_tuples(output: Path, preview: Path | None) -> None:
    horizon, tuples = worked_example()
    figure = plt.figure(figsize=(10.8, 7.2))
    figure.suptitle("Observe a finite horizon, keep only distinct tuples", fontsize=16, color=INK, y=0.97)
    sequence = figure.add_axes((0.115, 0.61, 0.85, 0.26))
    sequence.set(xlim=(-0.65, horizon), ylim=(-0.55, 3.5))
    sequence.axis("off")
    sequence.text(0, 3.4, "2 startup steps", fontsize=13, color=INK)
    sequence.text(3.5, 3.4, "12 steps = lcm(4, 6) · one complete joint cycle", fontsize=13, color=INK)
    for step, (a, b) in enumerate(tuples):
        color = "#fef3c7" if step < 2 else "#dbeafe"
        sequence.text(step + 0.45, 2.55, str(step), ha="center", fontsize=13, color=INK)
        for row, value in [(1.35, a), (0.25, b)]:
            sequence.add_patch(Rectangle((step, row), 0.9, 0.8, facecolor=color,
                                        edgecolor="#94a3b8", linewidth=0.6))
            sequence.text(step + 0.45, row + 0.4, str(value), ha="center", va="center",
                          fontsize=15, color=INK)
    for y, label in [(2.55, "Step"), (1.75, "A index"), (0.65, "B index")]:
        sequence.text(-0.22, y, label, ha="right", va="center", fontsize=13, color=INK)
    sequence.axvline(1.95, ymin=0.12, ymax=0.82, color=MUTED, ls="--", lw=1)
    sequence.text(7, -0.45, "H = max(prefix lengths) + lcm(cycle lengths) = 2 + 12 = 14",
                  ha="center", fontsize=13, color=INK)

    grid = figure.add_axes((0.15, 0.14, 0.49, 0.37))
    grid.set(xlim=(-0.5, 5.5), ylim=(4.5, -0.5), aspect="equal")
    rows = [4, 0, 1, 2, 3]
    reachable = set(tuples)
    for row, a in enumerate(rows):
        for b in range(6):
            color = "white"
            if (a, b) in reachable:
                color = ORANGE if a == 4 else BLUE
            grid.add_patch(Rectangle((b - 0.5, row - 0.5), 1, 1, facecolor=color,
                                    edgecolor="#94a3b8", linewidth=0.7))
            if (a, b) in reachable:
                step = tuples.index((a, b))
                grid.text(b, row, str(step), ha="center", va="center", fontsize=12, color=INK)
    grid.set_xticks(range(6), [f"b{i}" for i in range(6)])
    grid.set_yticks(range(5), ["scratch", "a0", "a1", "a2", "a3"])
    grid.tick_params(length=0, labelsize=11, colors=INK)
    grid.set_xlabel("Storage B address", fontsize=11, color=INK, labelpad=10)
    grid.set_title("Reachable address pairs · cell number = first step", fontsize=11, color=INK, pad=14)
    for spine in grid.spines.values():
        spine.set_visible(False)
    figure.text(0.70, 0.43, "14 / 30", fontsize=30, weight="bold", color="#2563eb")
    figure.text(0.70, 0.385, "pairs need an executable", fontsize=12, color=INK)
    figure.text(0.70, 0.29, "2 startup pairs\n+ 12 cyclic pairs", fontsize=12, color=INK, linespacing=1.7)
    figure.text(0.70, 0.18, "Step 14 repeats step 2.\nThe horizon covers all future states.",
                fontsize=13, color=MUTED, linespacing=1.5)
    figure.text(0.15, 0.035, "An unordered two-address domain C expands every pair: 14 × 2 = 28 variants.",
                fontsize=13, color=INK)
    save(figure, "tuples", output, preview)


def label_svg(path: Path, title: str, description: str) -> None:
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(path)
    root = tree.getroot()
    root.set("role", "img")
    root.set("aria-labelledby", "figure-title figure-description")
    # Windows browsers may not have Matplotlib's DejaVu font. Keep selectable text with
    # explicit browser fallbacks instead of silently rendering labels in a serif font.
    for element in root.iter(f"{{{SVG}}}text"):
        element.set("style", element.get("style", "") + ";font-family:Arial,Helvetica,sans-serif")
    for tag, identifier, text in [("title", "figure-title", title),
                                  ("desc", "figure-description", description)]:
        element = ET.Element(f"{{{SVG}}}{tag}", {"id": identifier})
        element.text = text
        root.insert(0, element)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ASSETS / "images")
    parser.add_argument("--preview-dir", type=Path, help="Also render PNGs for visual inspection.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": STEM,
                               "font.family": "DejaVu Sans", "figure.facecolor": "white"})
    draw_timeline(args.output_dir, args.preview_dir)
    draw_tuples(args.output_dir, args.preview_dir)
    for name in ["execution", "construction", "lifecycle", "update-order"]:
        source = ASSETS / "graphs" / f"{STEM}{name}.dot"
        subprocess.run(["dot", "-Tsvg", str(source), "-o",
                        str(args.output_dir / f"{STEM}{name}.svg")], check=True)
        if args.preview_dir:
            subprocess.run(["dot", "-Tpng", "-Gdpi=150", str(source), "-o",
                            str(args.preview_dir / f"{STEM}{name}.png")], check=True)
    for name, title, description in [
        ("timeline", "Ordinary submission versus CUDA graph replay",
         "Same illustrative time scale and kernel durations; graph replay reduces submission gaps."),
        ("tuples", "Finite sequence horizon and reachable pointer tuples",
         "Two startup steps plus a twelve-step joint cycle yield fourteen of thirty possible address pairs."),
        ("execution", "One section iteration", "Host operations surround graph lookup and compute submission."),
        ("construction", "Graph-set construction", "Compilation inspects; startup prepares executables before workers."),
        ("lifecycle", "Graph-set lifecycle", "Pause retains the graph set; restart replans; updates invalidate incompatible captures."),
        ("update-order", "Exception unwinding during a factory update",
         "Later local owners unwind before the guard; the guard invalidates executables before the old task parameter releases captured resources."),
    ]:
        label_svg(args.output_dir / f"{STEM}{name}.svg", title, description)
    print("Generated six SVGs; verified 14 reachable tuples and 28 with unordered expansion.")


if __name__ == "__main__":
    main()
