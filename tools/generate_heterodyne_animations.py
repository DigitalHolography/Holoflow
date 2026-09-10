"""Generate the MP4 figures used by the optical heterodyne detection lesson.

Run this script from any directory with Matplotlib and FFmpeg available:

    python tools/generate_heterodyne_animations.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


FPS = 30
FRAME_COUNT = 180
DURATION_SECONDS = FRAME_COUNT / FPS
FIGURE_SIZE = (12.8, 7.2)
DPI = 100

SIGNAL_COLOR = "#2563eb"
LO_COLOR = "#ea580c"
BEAT_COLOR = "#9333ea"
CAMERA_COLOR = "#047857"
GRID_COLOR = "#cbd5e1"

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "doc"
    / "mkdocs"
    / "docs"
    / "assets"
    / "videos"
    / "learn"
    / "heterodyne"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the four MP4 files.",
    )
    return parser.parse_args()


def configure_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def create_field_figure(
    title: str,
    top_ylim: tuple[float, float],
    beat_ylim: tuple[float, float],
):
    figure, (field_axis, beat_axis) = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE,
        dpi=DPI,
        sharex=True,
        gridspec_kw={"height_ratios": (1, 1.15)},
        constrained_layout=True,
    )
    figure.suptitle(title, fontsize=17, fontweight="bold")
    figure.set_facecolor("#f8fafc")

    configure_axis(field_axis, "Field amplitude")
    configure_axis(beat_axis, "Summed field")
    field_axis.set_ylim(*top_ylim)
    beat_axis.set_ylim(*beat_ylim)
    beat_axis.set_xlabel("Time (µs)")

    (signal_line,) = field_axis.plot(
        [], [], color=SIGNAL_COLOR, linewidth=1.8, label=r"$s_{sig}$"
    )
    (lo_line,) = field_axis.plot(
        [], [], color=LO_COLOR, linewidth=1.8, label=r"$s_{LO}$"
    )
    (beat_line,) = beat_axis.plot(
        [],
        [],
        color=BEAT_COLOR,
        linewidth=2.0,
        label=r"$s_{beat}=s_{sig}+s_{LO}$",
    )
    field_axis.legend(loc="upper right", ncols=2, framealpha=0.95)
    beat_axis.legend(loc="upper right", framealpha=0.95)

    parameter_text = beat_axis.text(
        0.015,
        0.96,
        "",
        transform=beat_axis.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID_COLOR,
            "alpha": 0.94,
        },
    )
    return (
        figure,
        field_axis,
        beat_axis,
        signal_line,
        lo_line,
        beat_line,
        parameter_text,
    )


def save_animation(figure: plt.Figure, update, output_path: Path) -> None:
    movie = animation.FuncAnimation(
        figure,
        update,
        frames=FRAME_COUNT,
        interval=1000 / FPS,
        blit=False,
        repeat=True,
    )
    writer = animation.FFMpegWriter(
        fps=FPS,
        codec="libx264",
        metadata={"artist": "Holoflow", "title": output_path.stem},
        extra_args=["-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    print(f"Rendering {output_path.name} ({DURATION_SECONDS:.0f} s at {FPS} fps)...")
    movie.save(output_path, writer=writer, dpi=DPI)
    plt.close(figure)


def generate_fixed_detuning(output_dir: Path) -> None:
    time = np.linspace(0.0, 250e-6, 1800)
    time_us = time * 1e6
    signal_frequency = 200e3
    lo_frequency = 220e3
    detuning = lo_frequency - signal_frequency

    (
        figure,
        field_axis,
        _,
        signal_line,
        lo_line,
        beat_line,
        parameter_text,
    ) = create_field_figure(
        "Equal amplitudes, fixed detuning",
        (-1.25, 1.25),
        (-2.25, 2.25),
    )
    field_axis.set_xlim(time_us[0], time_us[-1])

    def update(frame: int):
        time_offset = frame / FRAME_COUNT / detuning
        absolute_time = time + time_offset
        signal = np.cos(2 * np.pi * signal_frequency * absolute_time)
        lo = np.cos(2 * np.pi * lo_frequency * absolute_time)
        signal_line.set_data(time_us, signal)
        lo_line.set_data(time_us, lo)
        beat_line.set_data(time_us, signal + lo)
        parameter_text.set_text(
            r"$A_{sig}=A_{LO}=1$"
            "\n"
            r"$f_{sig,display}=200\ kHz,\ f_{LO,display}=220\ kHz$"
            "\n"
            r"$\Delta f=20\ kHz$"
        )
        return signal_line, lo_line, beat_line, parameter_text

    save_animation(figure, update, output_dir / "heterodyne-fixed-detuning.mp4")


def generate_variable_detuning(output_dir: Path) -> None:
    time = np.linspace(0.0, 250e-6, 1800)
    time_us = time * 1e6
    signal_frequency = 200e3

    (
        figure,
        field_axis,
        _,
        signal_line,
        lo_line,
        beat_line,
        parameter_text,
    ) = create_field_figure(
        "Equal amplitudes, variable detuning",
        (-1.25, 1.25),
        (-2.25, 2.25),
    )
    field_axis.set_xlim(time_us[0], time_us[-1])

    def update(frame: int):
        phase = 2 * np.pi * frame / FRAME_COUNT
        detuning = 20e3 - 15e3 * np.cos(phase)
        lo_frequency = signal_frequency + detuning
        signal = np.cos(2 * np.pi * signal_frequency * time + phase)
        lo = np.cos(2 * np.pi * lo_frequency * time + phase)
        signal_line.set_data(time_us, signal)
        lo_line.set_data(time_us, lo)
        beat_line.set_data(time_us, signal + lo)
        parameter_text.set_text(
            r"$A_{sig}=A_{LO}=1$"
            "\n"
            rf"$\Delta f={detuning / 1e3:4.1f}\ kHz$"
            "\n"
            r"$5\ kHz\leq\Delta f\leq35\ kHz$"
        )
        return signal_line, lo_line, beat_line, parameter_text

    save_animation(figure, update, output_dir / "heterodyne-variable-detuning.mp4")


def generate_lo_amplitude(output_dir: Path) -> None:
    time = np.linspace(0.0, 250e-6, 1800)
    time_us = time * 1e6
    signal_frequency = 200e3
    lo_frequency = 220e3
    detuning = lo_frequency - signal_frequency
    signal_amplitude = 1.0

    (
        figure,
        field_axis,
        _,
        signal_line,
        lo_line,
        beat_line,
        parameter_text,
    ) = create_field_figure(
        "Local-oscillator amplitude and detected cross-term",
        (-3.3, 3.3),
        (-4.25, 4.25),
    )
    field_axis.set_xlim(time_us[0], time_us[-1])

    def update(frame: int):
        phase = 2 * np.pi * frame / FRAME_COUNT
        lo_amplitude = 1.75 - 1.25 * np.cos(phase)
        time_offset = frame / FRAME_COUNT / detuning
        absolute_time = time + time_offset
        signal = signal_amplitude * np.cos(2 * np.pi * signal_frequency * absolute_time)
        lo = lo_amplitude * np.cos(2 * np.pi * lo_frequency * absolute_time)
        signal_line.set_data(time_us, signal)
        lo_line.set_data(time_us, lo)
        beat_line.set_data(time_us, signal + lo)
        parameter_text.set_text(
            rf"$A_{{sig}}={signal_amplitude:.1f},\ A_{{LO}}={lo_amplitude:.2f}$"
            "\n"
            r"$\Delta f=20\ kHz$"
            "\n"
            rf"Detected cross-term amplitude: $A_{{sig}}A_{{LO}}={signal_amplitude * lo_amplitude:.2f}$"
        )
        return signal_line, lo_line, beat_line, parameter_text

    save_animation(figure, update, output_dir / "heterodyne-lo-amplitude.mp4")


def generate_camera_averaging(output_dir: Path) -> None:
    time = np.linspace(0.0, 150e-6, 6000)
    time_us = time * 1e6
    signal_frequency = 2e6
    detuning = 20e3
    lo_frequency = signal_frequency + detuning
    camera_frequency = 50e3
    exposure = 1 / camera_frequency
    attenuation = np.sinc(detuning * exposure)

    figure, (field_axis, detector_axis) = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE,
        dpi=DPI,
        sharex=True,
        gridspec_kw={"height_ratios": (1, 1.2)},
        constrained_layout=True,
    )
    figure.suptitle(
        "A finite-exposure camera rejects the optical carrier",
        fontsize=17,
        fontweight="bold",
    )
    figure.set_facecolor("#f8fafc")
    configure_axis(field_axis, "Field amplitude")
    configure_axis(detector_axis, "Relative irradiance")
    field_axis.set_ylim(-1.25, 1.25)
    detector_axis.set_ylim(-0.15, 4.15)
    detector_axis.set_xlim(time_us[0], time_us[-1])
    detector_axis.set_xlabel("Time (µs)")

    (signal_line,) = field_axis.plot(
        [], [], color=SIGNAL_COLOR, linewidth=1.0, alpha=0.85, label=r"$s_{sig}$"
    )
    (lo_line,) = field_axis.plot(
        [], [], color=LO_COLOR, linewidth=1.0, alpha=0.85, label=r"$s_{LO}$"
    )
    (raw_line,) = detector_axis.plot(
        [],
        [],
        color=BEAT_COLOR,
        linewidth=0.75,
        alpha=0.25,
        label=r"Raw $|s_{beat}|^2$",
    )
    (averaged_line,) = detector_axis.plot(
        [],
        [],
        color=CAMERA_COLOR,
        linewidth=2.5,
        label="Exposure-averaged beat",
    )
    sample_markers = detector_axis.scatter(
        [],
        [],
        s=48,
        color=CAMERA_COLOR,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label="50 kfps samples",
    )
    field_axis.legend(loc="upper right", ncols=2, framealpha=0.95)
    detector_axis.legend(loc="upper right", ncols=3, framealpha=0.95)

    parameter_text = detector_axis.text(
        0.015,
        0.96,
        "",
        transform=detector_axis.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID_COLOR,
            "alpha": 0.94,
        },
    )
    sample_times = np.arange(
        0.0,
        time[-1] + 0.5 / camera_frequency,
        1 / camera_frequency,
    )

    def update(frame: int):
        time_offset = frame / FRAME_COUNT / detuning
        absolute_time = time + time_offset
        signal = np.cos(2 * np.pi * signal_frequency * absolute_time)
        lo = np.cos(2 * np.pi * lo_frequency * absolute_time)
        beat = signal + lo
        raw_intensity = beat**2
        averaged_intensity = 1.0 + attenuation * np.cos(
            2 * np.pi * detuning * absolute_time
        )

        absolute_sample_times = sample_times + time_offset
        samples = 1.0 + attenuation * np.cos(
            2 * np.pi * detuning * absolute_sample_times
        )

        signal_line.set_data(time_us, signal)
        lo_line.set_data(time_us, lo)
        raw_line.set_data(time_us, raw_intensity)
        averaged_line.set_data(time_us, averaged_intensity)
        sample_markers.set_offsets(np.column_stack((sample_times * 1e6, samples)))
        parameter_text.set_text(
            r"$f_{display}=2\ MHz\ \leftrightarrow\ f_0=351.87\ THz$"
            "\n"
            r"$\Delta f=20\ kHz,\ f_s=50\ kfps,\ f_{Nyquist}=25\ kHz$"
            "\n"
            rf"$T_{{exp}}=20\ \mu s,\ sinc(\Delta f T_{{exp}})={attenuation:.3f}$"
        )
        return (
            signal_line,
            lo_line,
            raw_line,
            averaged_line,
            sample_markers,
            parameter_text,
        )

    save_animation(figure, update, output_dir / "heterodyne-camera-averaging.mp4")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_fixed_detuning(output_dir)
    generate_variable_detuning(output_dir)
    generate_lo_amplitude(output_dir)
    generate_camera_averaging(output_dir)

    print(f"Generated four animations in {output_dir}")


if __name__ == "__main__":
    main()
