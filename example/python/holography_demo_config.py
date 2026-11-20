# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the holography demo settings."""

from dataclasses import dataclass
from math import isfinite, lcm
from pathlib import Path

import yaml

from holoflow import _native


@dataclass(frozen=True)
class DemoConfig:
    input: dict
    time: dict
    optics: dict
    space: dict
    aberration: dict
    metrics: dict
    output: dict
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
        "time": dict(method="pca", window_frames=32, component_begin=0, component_end=16,
                     batch_windows=4),
        "optics": dict(wavelength_m=8.51e-7, pixel_pitch_m=2.0e-5),
        "space": dict(method="fresnel", distance_m=0.49),
        "aberration": dict(enabled=False, distance_m=0.49, subapertures=5,
                           average_window=64, zernike_modes=[4, 5, 6], prefilter=None,
                           skip_outside_pupil=True, sensor_flatfield_cutoff_m=0.01),
        "metrics": dict(interval_ms=1000, history_batches=8192),
        "output": dict(average_window=1, dtype="float32", flatfield_cutoff_m=None,
                       percentile_clip=None),
        "viewer": dict(enabled=False, max_frames=256, cao_diagnostics=False),
        "queues": dict(input_frames=1024, preprocessing_windows=8, propagation_windows=8,
                       corrected_windows=2, output_images=32),
        "graph": dict(enabled=False, log_dir="logs/python", open_browser=True),
    }
    raw = yaml.safe_load(filename.read_text(encoding="utf-8"))
    _check(isinstance(raw, dict), "config must be a YAML mapping")
    for name, fields in raw.items():
        _check(name in defaults, f"Unknown YAML section: {name}")
        _check(isinstance(fields, dict), f"{name} must be a mapping")
        for key in fields:
            _check(key in defaults[name], f"Unknown YAML key: {name}.{key}")
        defaults[name].update(fields)
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
    pca_batch = 1
    if time["method"] == "pca":
        pca_batch = time["batch_windows"]
        _check(_integer(pca_batch), "time.batch_windows must be positive")
        temporal_frames = time["window_frames"]
        _check(_integer(temporal_frames, 2), "time.window_frames must be at least 2")
        begin, end = time["component_begin"], time["component_end"]
        _check(_integer(begin, 0) and _integer(end) and begin < end <= temporal_frames,
               "time.component_begin and time.component_end must bound a nonempty range within time.window_frames")
    space = defaults["space"]
    _check(space["method"] in ("none", "fresnel"), "space.method must be none or fresnel")
    aberration = defaults["aberration"]
    viewer = defaults["viewer"]
    graph = defaults["graph"]
    output = defaults["output"]
    _check(output["dtype"] in ("float32", "uint8"), "output.dtype must be float32 or uint8")
    for key, value in defaults["queues"].items():
        _check(_integer(value), f"queues.{key} must be positive")
    _check(type(viewer["cao_diagnostics"]) is bool, "viewer.cao_diagnostics must be a boolean")
    _check(not viewer["cao_diagnostics"] or (viewer["enabled"] and aberration["enabled"]),
           "viewer.cao_diagnostics requires viewer.enabled and aberration.enabled")
    cutoff = output["flatfield_cutoff_m"]
    _check(cutoff is None or (_number(cutoff) and cutoff > 0 and space["method"] == "fresnel"),
           "output.flatfield_cutoff_m must be positive and requires Fresnel")
    clip = output["percentile_clip"]
    if clip is not None:
        keys = {"min_pct", "max_pct", "roi_radius"}
        _check(isinstance(clip, dict) and set(clip) == keys and all(map(_number, clip.values())),
               "output.percentile_clip requires min_pct, max_pct, and roi_radius")
        _check(0 <= clip["min_pct"] < clip["max_pct"] <= 100 and 0 < clip["roi_radius"] <= 1,
               "Invalid output.percentile_clip bounds or ROI radius")
    for name, section in (("aberration", aberration), ("viewer", viewer), ("graph", graph)):
        _check(type(section["enabled"]) is bool, f"{name}.enabled must be a boolean")
    _check(type(graph["open_browser"]) is bool, "graph.open_browser must be a boolean")
    _check(isinstance(graph["log_dir"], str) and graph["log_dir"].strip(), "graph.log_dir must be a path")
    graph["log_dir"] = (filename.parent / graph["log_dir"]).resolve()
    correction_window = 1
    if aberration["enabled"]:
        prefilter = aberration["prefilter"]
        if prefilter is not None:
            keys = {"r_inner", "r_outer", "s_inner", "s_outer"}
            _check(isinstance(prefilter, dict) and set(prefilter) == keys and
                   all(_integer(value, 0) for value in prefilter.values()) and
                   prefilter["r_inner"] < prefilter["r_outer"], "Invalid aberration.prefilter")
        _check(type(aberration["skip_outside_pupil"]) is bool,
               "aberration.skip_outside_pupil must be a boolean")
        cutoff = aberration["sensor_flatfield_cutoff_m"]
        _check(cutoff is None or (_number(cutoff) and cutoff > 0),
               "aberration.sensor_flatfield_cutoff_m must be positive or null")
        correction_window = aberration["average_window"]
        _check(_integer(correction_window), "Invalid aberration.average_window")
        _check(_integer(aberration["subapertures"], 2), "aberration.subapertures must be at least 2")
        modes = aberration["zernike_modes"]
        _check(isinstance(modes, list) and modes and all(_integer(mode, 2) for mode in modes)
               and len(set(modes)) == len(modes), "aberration.zernike_modes must contain unique Noll indices >= 2")
    if aberration["enabled"] or space["method"] == "fresnel":
        for key, value in defaults["optics"].items():
            _check(_number(value) and value > 0, f"optics.{key} must be positive")
    for name, enabled in (("space", space["method"] == "fresnel"),
                          ("aberration", aberration["enabled"])):
        distance = defaults[name]["distance_m"]
        if enabled:
            _check(_number(distance) and distance != 0, f"{name}.distance_m must be finite and nonzero")
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
    alignment = lcm(source["batch_size"], temporal_frames * pca_batch)
    remainder = (end - start) % alignment
    _check(not remainder or source["tail"] == "drop",
           f"input.tail: selected range must be divisible by {alignment} for complete source and PCA batches")
    source["end_frame"] = end - remainder
    windows = (source["end_frame"] - start) // temporal_frames
    _check(windows > 0, "input range has no complete source batch and temporal window")
    average_window = defaults["output"]["average_window"]
    _check(_integer(average_window), "output.average_window must be a positive integer")
    output_count = max(0, windows - correction_window - average_window + 2)
    if viewer["enabled"]:
        _check(0 < output_count <= viewer["max_frames"],
               f"viewer.max_frames: selected range would produce {output_count} images")
        _check(not viewer["cao_diagnostics"] or windows <= viewer["max_frames"],
               f"viewer.max_frames: selected range would produce {windows} CAO updates")
    return DemoConfig(**defaults, temporal_frames=temporal_frames,
                      correction_window=correction_window, output_count=output_count,
                      temporal_count=windows)
