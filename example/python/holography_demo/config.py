# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the holography demo settings."""

from dataclasses import dataclass
from math import isfinite, lcm
from pathlib import Path
from warnings import warn

import yaml

from holoflow import _native


@dataclass(frozen=True)
class DemoConfig:
    input: dict
    time: dict
    space: dict
    spatial_filter: dict
    aberration: dict
    metrics: dict
    postprocessing: dict
    viewer: dict
    graph: dict
    queues: dict
    temporal_frames: int
    correction_window: int
    output_count: int
    temporal_count: int


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=1, maximum=(1 << 31) - 1):
    return type(value) is int and minimum <= value <= maximum


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def load_config(filename):
    filename = Path(filename).resolve()
    # Keep defaults independent of the editable example.
    defaults = {
        "input": dict(path=None, mode="finite", start_frame=0, end_frame=None, batch_size=32,
                      load_kind="live", tail="drop", max_fps=None, duration_seconds=None),
        "time": dict(method="pca", precision="fp32", window_frames=32,
                     component_begin=0, component_end=16),
        "space": dict(method="fresnel", wavelength_m=8.51e-7, pixel_pitch_m=2.0e-5,
                      distance_m=0.49),
        "spatial_filter": dict(enabled=False, r_inner=3, r_outer=1024, s_inner=0, s_outer=1),
        "aberration": dict(enabled=False, subapertures=5, zernike_modes=[4, 5, 6],
                           skip_outside_pupil=True,
                           slope_mode="full_pairwise"),
        "metrics": dict(interval_ms=1000, history_batches=8192),
        "postprocessing": dict(average_window=1, flatfield_cutoff_m=None, percentile_clip=None),
        "viewer": dict(enabled=False, max_frames=256, cao_diagnostics=False,
                       cao_coefficients_only=False, cao_compare_modes=False),
        "queues": dict(input_frames=1024, propagation_windows=8,
                       corrected_windows=2, output_images=32),
        "graph": dict(enabled=False, log_dir="logs/python", open_browser=True),
    }
    raw = yaml.safe_load(filename.read_text(encoding="utf-8"))
    _check(isinstance(raw, dict), "config must be a YAML mapping")
    legacy_keys = {"time": {"batch_windows"}, "queues": {"preprocessing_windows"}}
    for name, fields in raw.items():
        _check(name in defaults, f"Unknown YAML section: {name}")
        _check(isinstance(fields, dict), f"{name} must be a mapping")
        for key, value in fields.items():
            if key in legacy_keys.get(name, ()):
                continue
            _check(key in defaults[name], f"Unknown YAML key: {name}.{key}")
            defaults[name][key] = value
    _check("input" in raw and "path" in raw["input"], "input.path is required")
    source = defaults["input"]
    _check(isinstance(source["path"], str) and source["path"].strip(), "input.path must be a path")
    source["path"] = (filename.parent / source["path"]).resolve()
    _check(source["mode"] in ("finite", "looping"), "input.mode must be finite or looping")
    _check(source["load_kind"] in ("live", "cpu_cached", "gpu_cached"), "Invalid input.load_kind")
    _check(source["tail"] in ("drop", "error"), "input.tail must be drop or error")
    for key in ("batch_size", "start_frame", "end_frame", "max_fps"):
        value = source[key]
        if key in ("end_frame", "max_fps") and value is None:
            continue
        _check(_integer(value, 0 if key == "start_frame" else 1), f"Invalid input.{key}")
    duration = source["duration_seconds"]
    _check(duration is None or (source["mode"] == "looping" and _number(duration) and duration > 0),
           "input.duration_seconds must be positive and is only used in looping mode")

    time = defaults["time"]
    _check(time["method"] in ("none", "pca"), "time.method must be none or pca")
    temporal_frames = 1
    if time["method"] == "pca":
        temporal_frames = time["window_frames"]
        _check(_integer(temporal_frames, 2), "time.window_frames must be at least 2")
        begin, end = time["component_begin"], time["component_end"]
        _check(_integer(begin, 0) and _integer(end) and begin < end <= temporal_frames,
               "time.component_begin and time.component_end must bound a nonempty range within time.window_frames")
    _check(time["precision"] in ("fp32", "fp16"), "time.precision must be fp32 or fp16")
    space = defaults["space"]
    _check(space["method"] in ("none", "fresnel"), "space.method must be none or fresnel")
    spatial_filter = defaults["spatial_filter"]
    _check(type(spatial_filter["enabled"]) is bool, "spatial_filter.enabled must be a boolean")
    for key in ("r_inner", "r_outer", "s_inner", "s_outer"):
        _check(_integer(spatial_filter[key], 0), f"spatial_filter.{key} must be a nonnegative integer")
    _check(spatial_filter["r_inner"] < spatial_filter["r_outer"],
           "spatial_filter.r_inner must be less than spatial_filter.r_outer")
    aberration = defaults["aberration"]
    viewer = defaults["viewer"]
    _check(type(viewer["cao_compare_modes"]) is bool, "viewer.cao_compare_modes must be a boolean")
    _check(not viewer["cao_compare_modes"] or viewer["cao_coefficients_only"] is True,
           "viewer.cao_compare_modes requires viewer.cao_coefficients_only")
    _check(aberration["slope_mode"] in ("single_reference", "full_pairwise"),
           "aberration.slope_mode must be single_reference or full_pairwise")
    if time["precision"] == "fp16":
        _check(source["batch_size"] == 32 and time["method"] == "pca"
               and temporal_frames == 32 and end - begin <= 16,
               "time.precision: fp16 requires 32-frame input and PCA windows with at most 16 components")
        _check(space["method"] == "fresnel" and not aberration["enabled"]
               and not spatial_filter["enabled"],
               "time.precision: fp16 requires Fresnel with aberration correction and spatial filtering disabled")
    graph = defaults["graph"]
    post = defaults["postprocessing"]
    average_window = post["average_window"]
    _check(_integer(average_window), "postprocessing.average_window must be a positive integer")
    for key, value in defaults["queues"].items():
        _check(_integer(value), f"queues.{key} must be positive")
    for key in ("cao_diagnostics", "cao_coefficients_only"):
        _check(type(viewer[key]) is bool, f"viewer.{key} must be a boolean")
        _check(not viewer[key] or (viewer["enabled"] and aberration["enabled"]),
               f"viewer.{key} requires viewer.enabled and aberration.enabled")
    _check(not (viewer["cao_diagnostics"] and viewer["cao_coefficients_only"]),
           "Choose either viewer.cao_diagnostics or viewer.cao_coefficients_only")
    cutoff = post["flatfield_cutoff_m"]
    _check(cutoff is None or (_number(cutoff) and cutoff > 0),
           "postprocessing.flatfield_cutoff_m must be positive or null")
    clip = post["percentile_clip"]
    if clip is not None:
        keys = {"min_pct", "max_pct", "roi_radius"}
        _check(isinstance(clip, dict) and set(clip) == keys and all(map(_number, clip.values())),
               "postprocessing.percentile_clip requires min_pct, max_pct, and roi_radius")
        _check(0 <= clip["min_pct"] < clip["max_pct"] <= 100 and 0 < clip["roi_radius"] <= 1,
               "Invalid postprocessing.percentile_clip bounds or ROI radius")
    for name, section in (("aberration", aberration), ("viewer", viewer), ("graph", graph)):
        _check(type(section["enabled"]) is bool, f"{name}.enabled must be a boolean")
    _check(type(graph["open_browser"]) is bool, "graph.open_browser must be a boolean")
    _check(isinstance(graph["log_dir"], str) and graph["log_dir"].strip(), "graph.log_dir must be a path")
    graph["log_dir"] = (filename.parent / graph["log_dir"]).resolve()
    correction_window = average_window if aberration["enabled"] else 1
    if aberration["enabled"]:
        _check(type(aberration["skip_outside_pupil"]) is bool,
               "aberration.skip_outside_pupil must be a boolean")
        _check(_integer(aberration["subapertures"], 2), "aberration.subapertures must be at least 2")
        modes = aberration["zernike_modes"]
        _check(isinstance(modes, list) and modes and all(_integer(mode, 2) for mode in modes)
               and len(set(modes)) == len(modes), "aberration.zernike_modes must contain unique Noll indices >= 2")
        _check(not viewer["cao_coefficients_only"] or {4, 5, 6}.issubset(modes),
               "viewer.cao_coefficients_only requires aberration.zernike_modes to include 4, 5, and 6")
    for key in ("wavelength_m", "pixel_pitch_m"):
        _check(_number(space[key]) and space[key] > 0, f"space.{key} must be positive")
    distance = space["distance_m"]
    if aberration["enabled"]:
        _check(_number(distance) and distance > 0, "space.distance_m must be positive for CAO")
    elif space["method"] == "fresnel":
        _check(_number(distance) and distance != 0, "space.distance_m must be finite and nonzero")
    for key, value in defaults["metrics"].items():
        _check(_integer(value), f"metrics.{key} must be a positive integer")
    _check(_integer(viewer["max_frames"]), "viewer.max_frames must be a positive integer")
    _check(not viewer["enabled"] or source["mode"] == "finite", "viewer.enabled requires input.mode: finite")

    try:
        total = _native.file_frame_count(str(source["path"]))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"input.path: {exc}") from exc
    start = source["start_frame"]
    end = total if source["end_frame"] is None else source["end_frame"]
    _check(start < end <= total, f"input.start_frame and input.end_frame must select a range within {total} frames")
    alignment = lcm(source["batch_size"], temporal_frames)
    remainder = (end - start) % alignment
    _check(not remainder or source["tail"] == "drop",
           f"input.tail: selected range must be divisible by {alignment} for complete source batches and temporal windows")
    source["end_frame"] = end - remainder
    windows = (source["end_frame"] - start) // temporal_frames
    _check(windows > 0, "input range has no complete source batch and temporal window")
    if time["precision"] == "fp16" and source["mode"] == "finite":
        group_size = 4
        _check(windows >= group_size, "time.precision: finite fp16 requires at least four PCA windows")
        dropped_windows = windows % group_size
        if dropped_windows:
            dropped_frames = dropped_windows * temporal_frames
            source["end_frame"] -= dropped_frames
            windows -= dropped_windows
            warn(f"Finite FP16 input dropped {dropped_windows} trailing PCA window(s) "
                 f"({dropped_frames} frames) to keep complete four-window groups.",
                 stacklevel=2)
    output_count = max(0, windows - correction_window - average_window + 2)
    if viewer["cao_coefficients_only"]:
        _check(windows >= correction_window, "viewer.cao_coefficients_only needs a full sensor averaging window")
    elif viewer["enabled"]:
        _check(0 < output_count <= viewer["max_frames"],
               f"viewer.max_frames: selected range would produce {output_count} images")
    if viewer["cao_diagnostics"] or viewer["cao_coefficients_only"]:
        _check(windows <= viewer["max_frames"],
               f"viewer.max_frames: selected range would produce {windows} CAO updates")
    return DemoConfig(**defaults, temporal_frames=temporal_frames,
                      correction_window=correction_window, output_count=output_count,
                      temporal_count=windows)
