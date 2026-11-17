"""Independently specified native GraphSpec for comparison with Python tracing."""

import json
import holoflow as hf
from holoflow import _native


def reference_graph(input_path, output_path):
    stages = [
        ("Holofile", {"path": str(input_path), "batch_size": 32, "load_kind": "Live",
                      "start_frame": 0, "end_frame": 1024, "keep_cursor": False}),
        ("Memcpy", {"target": "Device"}),
        ("BatchQueue", {"target_capacity": 1024, "output_size": 256, "output_stride": 256}),
        ("Conversion", {"target": "F32", "strategy": "Real"}),
        ("Reshape", {"shape": [1, 256, 8, 8], "copy": None}),
        ("BatchQueue", {"target_capacity": 8, "output_size": 4, "output_stride": 4}),
        ("Pca", {"begin": 4, "end": 64}),
        ("FresnelDiffraction", {"lambda": 532e-9, "dx": 12e-6, "dy": 12e-6,
                                "z": .20, "axes": [-2, -1], "output_magnitude": True,
                                "skip_phase_shift": True}),
        ("Mean", {"axis": [-3], "keepdims": True}),
        ("Mean", {"axis": [0], "keepdims": False}),
        ("SlidingAverage", {"target_capacity": 8, "window_size": 4, "discard_first": 0}),
        ("Conversion", {"target": "U8", "strategy": "Scaled"}),
        ("Memcpy", {"target": "Host"}),
        ("BatchQueue", {"target_capacity": 8, "output_size": 1, "output_stride": 1}),
        ("HolofileWriter", {"path": str(output_path), "count": 100,
                            "pipeline_settings": {}, "use_buffer": True}),
    ]
    ids = [f"stage_{i}" for i in range(len(stages) - 1)] + ["doppler_writer"]
    spec = {
        "nodes": {node_id: {"type": kind, "params": params}
                  for node_id, (kind, params) in zip(ids, stages)},
        "edges": [{"from": src, "to": dst, "out": 0, "in": 0}
                  for src, dst in zip(ids, ids[1:])],
    }
    # Bypass tracing and its wrappers, entering the native compiler at GraphSpec.
    return hf.Graph(_native.Graph(json.dumps(spec)))
