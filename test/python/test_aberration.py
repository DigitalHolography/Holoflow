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

import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "example/python"))
import holoflow as hf
from fixture import write_holo
from fresnel_aberration_numpy import input_range, pipeline, collect_frames, create_viewer, XY_OFFSET


class AberrationTest(unittest.TestCase):
    def test_ranges_and_rounding(self):
        self.assertEqual(input_range(163840), (0, 36864, 1024))
        self.assertEqual(input_range(163840, 128, 2), (128, 4224, 2))
        self.assertEqual(input_range(163840, end=4096), (0, 4096, 2))
        self.assertEqual(input_range(163840, end=128), (0, 128, 0))
        with self.assertRaises(ValueError):
            input_range(163840, end=4128)

    def test_causal_average_and_strided_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.holo"
            write_holo(path, frames=8, width=8, height=8)
            expected = np.frombuffer(path.read_bytes(), dtype=np.uint16, offset=64).reshape(8, 8, 8)
            def graph():
                x = hf.sources.holofile(path, batch_size=1, loop=False)
                avg = hf.causal_sliding_average(x.to("device").astype(hf.float32), 3)
                return {"avg": hf.sinks.numpy(avg), "slice": hf.sinks.numpy(x[:, 1:, 1:3])}
            with hf.trace(graph).compile() as run:
                run.start()
                result = run.result()
            reference = np.stack([expected[max(0, i-2):i+1].mean(axis=0) for i in range(8)])
            np.testing.assert_allclose(result["avg"], reference, rtol=1e-5)
            np.testing.assert_array_equal(result["slice"], expected[:, 1:, 1:3])

    def test_complete_aberration_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.holo"
            write_holo(path, frames=4096, width=20, height=20)
            graph = hf.trace(pipeline, str(path), 0, 4096).to_json()
            kinds = [node["type"] for node in graph["nodes"].values()]
            self.assertEqual(kinds.count("ArrayCollector"), 4)
            for kind in ("ShortTimeFresnelDiffraction", "CausalSlidingAverage", "ShackHartmannSlopes",
                         "ZernikeFromSlopes", "ZernikePhase", "CorrectPhase", "Slice", "Transpose", "FFT2"):
                self.assertIn(kind, kinds)
            result = collect_frames(str(path), 0, 4096, 1)
            self.assertEqual(result["xy"].shape, (1, 20, 20))
            self.assertEqual(result["shack_hartmann"].shape, (128, 20, 20))
            self.assertEqual(result["coefficients"].shape, (128, 3))
            self.assertEqual(result["phase"].shape, (128, 20, 20))
            self.assertTrue(np.isfinite(result["coefficients"]).all())
            self.assertTrue(np.isfinite(result["phase"]).all())

    def test_dashboard_mapping(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backend_bases import KeyEvent
        except ImportError:
            self.skipTest("Matplotlib is required for dashboard tests")
        count = XY_OFFSET + 3
        result = {"xy": np.zeros((3, 4, 4), np.uint8),
                  "shack_hartmann": np.arange(count, dtype=np.uint8)[:, None, None] * np.ones((count, 4, 4), np.uint8),
                  "coefficients": np.zeros((count, 3), np.float32),
                  "phase": np.zeros((count, 4, 4), np.float32)}
        result["coefficients"][:63] = 1e9
        fig, slider = create_viewer(result)
        self.addCleanup(plt.close, fig)
        coefficient_axes = fig.axes[3]
        for line in coefficient_axes.lines[:3]:
            self.assertEqual(line.get_xdata()[0], 63)
            np.testing.assert_array_equal(line.get_ydata(), 0)
        self.assertLess(max(abs(v) for v in coefficient_axes.get_ylim()), 1)
        np.testing.assert_array_equal(result["coefficients"][:63], 1e9)
        slider.set_val(1)
        np.testing.assert_array_equal(fig.axes[1].images[0].get_array(), result["shack_hartmann"][XY_OFFSET+1])
        for _ in range(4):
            fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key="right"))
        self.assertEqual(slider.val, 2)


class HolographyDemoTest(unittest.TestCase):
    def setUp(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("Install the examples dependency group to test the YAML demo")
        self.yaml = yaml
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_holo(self.root / "input.holo", frames=35, width=20, height=20)

    def config(self, **sections):
        from holography_demo_config import load_config

        settings = {"input": {"path": "input.holo", "batch_size": 8},
                    "time": {"method": "pca", "window_frames": 4, "component_end": 2},
                    "metrics": {"interval_ms": 10}}
        for name, section in sections.items():
            settings.setdefault(name, {}).update(section)
        path = self.root / "demo.yaml"
        path.write_text(self.yaml.safe_dump(settings), encoding="utf-8")
        return load_config(path)

    def test_processing_paths_and_exact_latency_counts(self):
        from holography_demo_pipeline import pipeline

        cases = [
            ({"time": {"method": "none"}, "space": {"method": "none"}}, 32),
            ({}, 8),
            ({"aberration": {"enabled": True, "average_window": 3}}, 6),
            ({"aberration": {"enabled": True, "average_window": 3},
              "output": {"average_window": 2}}, 5),
        ]
        for sections, expected in cases:
            with self.subTest(sections=sections):
                config = self.config(**sections, viewer={"enabled": True, "max_frames": 32})
                graph = hf.trace(pipeline, config)
                spec = graph.to_json()
                self.assertEqual(spec["nodes"]["host_output"]["type"], "BatchQueue")
                self.assertEqual(spec["nodes"]["xy_output"]["type"], "ArrayCollector")
                self.assertNotIn("Void", {node["type"] for node in spec["nodes"].values()})
                with graph.compile() as run:
                    run.start()
                    result = run.result()
                    stats = run.latency("latency")
                self.assertEqual(config.input["end_frame"], 32)
                self.assertEqual(result.shape, (expected, 20, 20))
                self.assertTrue(np.isfinite(result).all())
                self.assertEqual(stats["matched_outputs"], expected)
                self.assertEqual(stats["unmatched_outputs"], 0)
                self.assertGreaterEqual(stats["oldest_ms"]["mean"], stats["newest_ms"]["mean"])
                if sections.get("time", {}).get("method") == "none":
                    raw = np.frombuffer((self.root / "input.holo").read_bytes(), np.uint16, offset=64)
                    np.testing.assert_array_equal(result, raw.reshape(35, 20, 20)[:32])

    def test_cao_reference_processing_and_dashboard_alignment(self):
        from holography_demo_pipeline import pipeline

        reference = Path(__file__).resolve().parents[2] / "example/python/holography_cao.yaml"
        settings = self.yaml.safe_load(reference.read_text(encoding="utf-8"))
        settings["input"].update(path="input.holo", batch_size=8, end_frame=32)
        settings["time"].update(window_frames=4, component_end=2)
        settings["aberration"]["average_window"] = 3
        settings["output"]["average_window"] = 2
        config = self.config(**settings)
        graph = hf.trace(pipeline, config)
        nodes = graph.to_json()["nodes"]
        self.assertEqual(nodes["pca_windows"]["params"]["output_size"], 4)
        self.assertEqual(nodes["correction_input"]["type"], "DualReaderBatchQueue")
        self.assertEqual(nodes["corrected_windows"]["type"], "BatchQueue")
        self.assertIn("PctClip", {node["type"] for node in nodes.values()})
        for name in ("shack_hartmann", "phase", "coefficients"):
            self.assertEqual(nodes[name + "_queue_device"]["type"], "BatchQueue")
        with graph.compile(log_dir=self.root / "compiled", capture=False) as run:
            run.start()
            images = run.result()
            self.assertEqual(run.latency("latency")["matched_outputs"], 5)
        self.assertIn("cluster_section_", run.artifacts.compiled_dot.read_text(encoding="utf-8"))
        sections = json.loads((run.artifacts.directory / "section_cuda_graphs.json").read_text(encoding="utf-8"))
        correction, = [section for section in sections
                       if any(task["kind"] == "ShortTimeFresnelDiffraction" for task in section["tasks"])]
        self.assertEqual(correction["launches"], config.temporal_count)
        self.assertNotIn("Memcpy", {task["kind"] for task in correction["tasks"]})
        self.assertEqual(images["xy"].dtype, np.uint8)
        self.assertEqual(images["xy"].shape, (5, 20, 20))
        self.assertEqual(images["coefficients"].shape, (8, 3))
        self.assertEqual(images["phase"].shape, (8, 20, 20))
        self.assertEqual(images["shack_hartmann"].shape, (8, 20, 20))
        import matplotlib.pyplot as plt
        fig, slider = create_viewer(images, xy_offset=3, discard_first=2, zernike_modes=[4, 5, 6])
        self.addCleanup(plt.close, fig)
        slider.set_val(4)
        np.testing.assert_array_equal(fig.axes[2].images[0].get_array(), images["phase"][7])

    def test_output_sliding_average_matches_numpy(self):
        from holography_demo_pipeline import pipeline

        config = self.config(time={"method": "none"}, space={"method": "none"},
                             output={"average_window": 3}, viewer={"enabled": True})
        graph = hf.trace(pipeline, config)
        self.assertEqual(graph.to_json()["nodes"]["latency"]["params"]["window_frames"], 3)
        with graph.compile() as run:
            run.start()
            result = run.result()
            self.assertEqual(run.latency("latency")["matched_outputs"], 30)
        raw = np.frombuffer((self.root / "input.holo").read_bytes(), np.uint16, offset=64)
        frames = raw.reshape(35, 20, 20)[:32].astype(np.float32)
        expected = (frames[:-2] + frames[1:-1] + frames[2:]) / 3
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-4)

    def test_config_validation(self):
        for sections, key in [
            ({"viewer": {"enabled": True, "max_frames": 7}}, "viewer.max_frames"),
            ({"input": {"tail": "error"}}, "input.tail"),
            ({"time": {"method": "fft"}}, "time.method"),
            ({"space": {"method": "angular"}}, "space.method"),
            ({"output": {"average_window": 0}}, "output.average_window"),
        ]:
            with self.subTest(sections=sections), self.assertRaisesRegex(ValueError, key):
                self.config(**sections)

    def test_looping_cli_reports_summary(self):
        from holography_demo import run_demo

        config = self.config(input={"mode": "looping", "load_kind": "gpu_cached",
                                    "max_fps": 320, "duration_seconds": 0.2},
                             time={"method": "none"}, space={"method": "none"})
        with redirect_stdout(StringIO()) as capture:
            self.assertIsNone(run_demo(config))
        self.assertRegex(capture.getvalue(), r"Latency: [1-9][0-9,]* matched, 0 unmatched")
        self.assertIn("Trace-to-drain wall time", capture.getvalue())
        self.assertFalse((self.root / "logs").exists())

    def test_compiled_graph_report_with_viewer_and_without_graphviz(self):
        from holography_demo import run_demo

        config = self.config(time={"method": "none"}, space={"method": "none"},
                             viewer={"enabled": True},
                             graph={"enabled": True, "log_dir": "reports", "open_browser": False})
        with patch("holography_demo_graph.shutil.which", return_value=None), \
                patch("holography_demo_graph.webbrowser.open") as browser, \
                redirect_stdout(StringIO()) as output:
            images = run_demo(config)
        self.assertEqual(images.shape, (32, 20, 20))
        browser.assert_not_called()
        dot, = (self.root / "reports").glob("*/compilation_success.dot")
        self.assertIn("cluster_section_", dot.read_text(encoding="utf-8"))
        self.assertFalse(dot.with_name("trace_events.perfetto-trace").exists())
        report = dot.with_name("compiled_graph.html").read_text(encoding="utf-8")
        self.assertIn("Copy DOT", report)
        self.assertIn("Graphviz rendering is unavailable", report)
        self.assertIn("cluster_section_", report)
        self.assertIn(str(dot.resolve()), output.getvalue())


if __name__ == "__main__":
    unittest.main()
