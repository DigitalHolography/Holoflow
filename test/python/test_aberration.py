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
from fresnel_aberration_numpy import input_range, pipeline, collect_frames, XY_OFFSET
from holography_demo.viewer import create_viewer


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

    def test_finite_independent_readers_join_by_position(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.holo"
            write_holo(path, frames=8, width=8, height=8)
            expected = np.frombuffer(path.read_bytes(), dtype=np.uint16, offset=64)
            expected = expected.reshape(8, 8, 8)[:6].astype(np.float32) * 2
            expected = np.concatenate((expected[:3], expected[3:]), axis=1)

            def graph(separate_sections):
                source = hf.sources.holofile(path, batch_size=1, loop=False)
                x = source.to("device").astype(hf.float32)
                branch, direct = hf.multi_reader_batch_queue(x, 3, 3, 3, 2, tail="drop")
                paired = hf.batch_queue(hf.copy(branch), 3, 3, 3) if separate_sections else branch
                return hf.sinks.numpy(direct + paired, axis=1)

            for separate_sections in (False, True):
                with self.subTest(separate_sections=separate_sections):
                    with hf.trace(graph, separate_sections).compile(capture=False) as run:
                        run.start()
                        np.testing.assert_array_equal(run.result(), expected)

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
        fig, slider = create_viewer(result, xy_offset=XY_OFFSET, discard_first=63, zernike_modes=[4, 5, 6])
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
    @staticmethod
    def scaled_uint8(images):
        minimum = images.min(axis=(-2, -1), keepdims=True)
        maximum = images.max(axis=(-2, -1), keepdims=True)
        return ((images - minimum) * (255.0 / (maximum - minimum)) + 0.5).astype(np.uint8)

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
        from holography_demo.config import load_config

        settings = {"input": {"path": "input.holo", "batch_size": 8},
                    "time": {"method": "pca", "window_frames": 4, "component_end": 2},
                    "metrics": {"interval_ms": 10}}
        for name, section in sections.items():
            settings.setdefault(name, {}).update(section)
        path = self.root / "demo.yaml"
        path.write_text(self.yaml.safe_dump(settings), encoding="utf-8")
        return load_config(path)

    def test_processing_paths_and_exact_latency_counts(self):
        from holography_demo.pipeline import pipeline

        cases = [
            ({"time": {"method": "none"}, "space": {"method": "none"}}, 32),
            ({}, 8),
            ({"aberration": {"enabled": True}, "postprocessing": {"average_window": 3}}, 4),
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
                self.assertEqual(result.dtype, np.uint8)
                self.assertTrue(np.isfinite(result).all())
                self.assertEqual(stats["matched_outputs"], expected)
                self.assertEqual(stats["unmatched_outputs"], 0)
                self.assertGreaterEqual(stats["oldest_ms"]["mean"], stats["newest_ms"]["mean"])
                if sections.get("time", {}).get("method") == "none":
                    raw = np.frombuffer((self.root / "input.holo").read_bytes(), np.uint16, offset=64)
                    frames = raw.reshape(35, 20, 20)[:32].astype(np.float32)
                    np.testing.assert_allclose(result, self.scaled_uint8(frames), atol=1, rtol=0)

    def test_cao_reference_processing_and_dashboard_alignment(self):
        from holography_demo.aberration import correct_aberration_native
        from holography_demo.pipeline import pipeline

        def trace_reference(config):
            with patch("holography_demo.pipeline.correct_aberration", correct_aberration_native):
                return hf.trace(pipeline, config)

        reference = Path(__file__).resolve().parents[2] / "example/python/holography_cao.yaml"
        write_holo(self.root / "input.holo", frames=35, width=30, height=20)
        settings = self.yaml.safe_load(reference.read_text(encoding="utf-8"))
        settings["input"].update(path="input.holo", batch_size=8, end_frame=32)
        settings["time"].update(precision="fp32", window_frames=4, component_end=2)
        settings["postprocessing"]["average_window"] = 3
        settings["viewer"].update(cao_diagnostics=True, cao_coefficients_only=False,
                                  cao_compare_modes=False)
        config = self.config(**settings)
        graph = trace_reference(config)
        spec = graph.to_json()
        nodes = spec["nodes"]
        self.assertTrue(any(e["from"] == "spatial_filter" and e["to"] == "correction_input"
                            for e in spec["edges"]))
        self.assertEqual(nodes["latency"]["params"]["window_frames"], 20)
        flatfields = [n["params"] for n in nodes.values() if n["type"] == "Flatfield"]
        scale = 0.187 * 0.01 * config.space["pixel_pitch_m"] / (
            config.space["wavelength_m"] * config.space["distance_m"])
        np.testing.assert_allclose(sorted((n["sigma_y"], n["sigma_x"]) for n in flatfields),
                                   [(4 * scale, 6 * scale), (20 * scale, 30 * scale)], rtol=1e-6)
        self.assertNotIn("pca_windows", nodes)
        self.assertTrue(any(nodes[e["from"]]["type"] == "Reshape" and nodes[e["to"]]["type"] == "Pca"
                            for e in spec["edges"]))
        self.assertEqual(nodes["correction_input"]["type"], "DualReaderBatchQueue")
        self.assertEqual(nodes["corrected_windows"]["type"], "BatchQueue")
        self.assertIn("PctClip", {node["type"] for node in nodes.values()})
        for name in ("shack_hartmann", "phase", "coefficients"):
            self.assertEqual(nodes[name + "_queue_device"]["type"], "BatchQueue")
        with graph.compile(log_dir=self.root / "compiled", capture=False) as run:
            run.start()
            images = run.result()
            self.assertEqual(run.latency("latency")["matched_outputs"], 4)
        self.assertIn("cluster_section_", run.artifacts.compiled_dot.read_text(encoding="utf-8"))
        sections = json.loads((run.artifacts.directory / "section_cuda_graphs.json").read_text(encoding="utf-8"))
        correction, = [section for section in sections
                       if any(task["kind"] == "ShortTimeFresnelDiffraction" for task in section["tasks"])]
        self.assertEqual(correction["launches"], config.temporal_count)
        self.assertNotIn("Memcpy", {task["kind"] for task in correction["tasks"]})
        self.assertEqual(images["xy"].dtype, np.uint8)
        self.assertEqual(images["xy"].shape, (4, 20, 30))
        self.assertEqual(images["coefficients"].shape, (8, 3))
        self.assertEqual(images["phase"].shape, (8, 20, 30))
        self.assertEqual(images["shack_hartmann"].shape, (8, 20, 30))
        import matplotlib.pyplot as plt
        fig, slider = create_viewer(images, xy_offset=4, discard_first=2, zernike_modes=[4, 5, 6])
        self.addCleanup(plt.close, fig)
        slider.set_val(3)
        np.testing.assert_array_equal(fig.axes[2].images[0].get_array(), images["phase"][7])

        from holography_demo.cli import create_coefficient_viewer

        settings["viewer"].update(cao_diagnostics=False, cao_coefficients_only=True)
        config = self.config(**settings)
        graph = trace_reference(config)
        nodes = graph.to_json()["nodes"]
        self.assertEqual(nodes["xy_output"]["type"], "Void")
        self.assertEqual(sum(node["type"] == "ArrayCollector" for node in nodes.values()), 1)
        self.assertNotIn("phase_queue_device", nodes)
        self.assertNotIn("shack_hartmann_queue_device", nodes)
        with graph.compile(log_dir=self.root / "coefficients", capture=False) as run:
            run.start()
            result = run.result()
            self.assertEqual(run.latency("latency")["matched_outputs"], 4)
            self.assertEqual(run.progress()["successful_emissions"]["output_fps"], 4)
        sections = json.loads((run.artifacts.directory / "section_cuda_graphs.json").read_text(encoding="utf-8"))
        correction, = [section for section in sections
                       if any(task["kind"] == "ShortTimeFresnelDiffraction" for task in section["tasks"])]
        self.assertEqual(correction["launches"], config.temporal_count)
        self.assertEqual(set(result), {"coefficients"})
        np.testing.assert_allclose(result["coefficients"], images["coefficients"], rtol=1e-5, atol=1e-6)
        reordered = result["coefficients"][:, [2, 0, 1]]
        figure = create_coefficient_viewer(reordered, [6, 4, 5], discard_first=2)
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(figure.axes), 3)
        for index, axis in enumerate(figure.axes):
            self.assertEqual(axis.get_ylabel(), f"a{index + 4} (rad)")
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), np.arange(2, 8))
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), result["coefficients"][2:, index])

        from holography_demo.cli import run_demo

        settings["viewer"]["cao_compare_modes"] = True
        settings["graph"].update(enabled=True, open_browser=False, log_dir="comparison")
        config = self.config(**settings)
        with redirect_stdout(StringIO()) as terminal:
            histories = run_demo(config)
        self.assertIn("Single reference (fresh compile)", terminal.getvalue())
        self.assertIn("Full-pairwise Laplacian (graph update)", terminal.getvalue())
        self.assertEqual(terminal.getvalue().count("Latency: 4 matched, 0 unmatched"), 2)
        np.testing.assert_allclose(histories["full_pairwise"], result["coefficients"],
                                   rtol=1e-5, atol=1e-6)
        settings["aberration"]["slope_mode"] = "single_reference"
        settings["viewer"]["cao_compare_modes"] = False
        single = self.config(**settings)
        with hf.trace(pipeline, single).compile() as run:
            run.start()
            expected = run.result()["coefficients"]
        np.testing.assert_allclose(histories["single_reference"], expected, rtol=1e-5, atol=1e-6)
        reports = list((self.root / "comparison").glob("*/section_cuda_graphs.json"))
        self.assertEqual(len(reports), 2)
        sections = [json.loads(report.read_text(encoding="utf-8")) for report in reports]
        self.assertEqual({section[0]["compilation_generation"] for section in sections}, {1, 2})
        figure = create_coefficient_viewer(histories["full_pairwise"], [4, 5, 6], 2,
                                           single_reference=histories["single_reference"])
        self.addCleanup(plt.close, figure)
        for col, axis in enumerate(figure.axes):
            self.assertEqual([line.get_color() for line in axis.lines], ["#808080", "black"])
            self.assertEqual([line.get_linewidth() for line in axis.lines], [0.6, 0.6])
            for line, history in zip(axis.lines, histories.values()):
                np.testing.assert_array_equal(line.get_xdata(), np.arange(2, 8))
                np.testing.assert_array_equal(line.get_ydata(), history[2:, col])

    def test_output_sliding_average_matches_numpy(self):
        from holography_demo.pipeline import pipeline

        config = self.config(time={"method": "none"}, space={"method": "none"},
                             postprocessing={"average_window": 3}, viewer={"enabled": True})
        graph = hf.trace(pipeline, config)
        self.assertEqual(graph.to_json()["nodes"]["latency"]["params"]["window_frames"], 3)
        with graph.compile() as run:
            run.start()
            result = run.result()
            self.assertEqual(run.latency("latency")["matched_outputs"], 30)
        raw = np.frombuffer((self.root / "input.holo").read_bytes(), np.uint16, offset=64)
        frames = raw.reshape(35, 20, 20)[:32].astype(np.float32)
        expected = (frames[:-2] + frames[1:-1] + frames[2:]) / 3
        np.testing.assert_allclose(result, self.scaled_uint8(expected), rtol=0, atol=1)

    def test_config_validation(self):
        config = self.config(
            input={"end_frame": 24, "tail": "error"},
            time={"batch_windows": 4},
            queues={"preprocessing_windows": 8},
        )
        self.assertEqual(config.input["end_frame"], 24)
        self.assertEqual(config.temporal_count, 6)

        for sections, key in [
            ({"viewer": {"enabled": True, "max_frames": 7}}, "viewer.max_frames"),
            ({"input": {"tail": "error"}}, "input.tail"),
            ({"time": {"method": "fft"}}, "time.method"),
            ({"space": {"method": "angular"}}, "space.method"),
            ({"postprocessing": {"average_window": 0}}, "postprocessing.average_window"),
        ]:
            with self.subTest(sections=sections), self.assertRaisesRegex(ValueError, key):
                self.config(**sections)

    def test_spatial_filter_and_flatfield_without_cao(self):
        from holography_demo.pipeline import pipeline

        spatial_filter = {"enabled": True, "r_inner": 0, "r_outer": 16, "s_inner": 0, "s_outer": 1}
        for method in ("none", "fresnel"):
            with self.subTest(method=method):
                config = self.config(space={"method": method}, spatial_filter=spatial_filter,
                                     postprocessing={"flatfield_cutoff_m": 0.001},
                                     viewer={"enabled": True})
                graph = hf.trace(pipeline, config)
                spec = graph.to_json()
                self.assertEqual(spec["nodes"]["spatial_filter"]["type"], "Filter2D")
                self.assertTrue(any(e["from"] == "spatial_filter" and e["to"] == "propagation_windows"
                                    for e in spec["edges"]))
                flatfield, = [n["params"] for n in spec["nodes"].values() if n["type"] == "Flatfield"]
                pitch = config.space["pixel_pitch_m"]
                if method == "fresnel":
                    pitch = config.space["wavelength_m"] * config.space["distance_m"] / (20 * pitch)
                np.testing.assert_allclose([flatfield["sigma_y"], flatfield["sigma_x"]],
                                           0.187 * 0.001 / pitch, rtol=1e-6)
                with graph.compile() as run:
                    run.start()
                    self.assertEqual(run.result().shape, (8, 20, 20))

    def test_decomposed_aberration_matches_native_stages(self):
        from holography_demo import aberration_ops

        path = self.root / "aberration.holo"
        write_holo(path, frames=18, width=21, height=16)
        modes = [10, 4, 2, 7, 6, 9, 3, 8, 5]

        def compare():
            real = hf.sources.holofile(path, batch_size=9, loop=False).to("device").astype(hf.float32)
            coefficients = hf.mean(real, axis=(-2, -1)) * 1e-4
            phase = aberration_ops.zernike_phase(coefficients, modes, 16, 21)
            native_phase = hf.zernike_phase(coefficients, modes, 16, 21)
            outputs = [
                native_phase.reshape((1, 16, 21)),
                phase.reshape((1, 16, 21)),
            ]
            for field in (real, real * (1 + 0.5j)):
                args = (field, 851e-9, 20e-6, 20e-6, 0.49, 5, 6, 5, 6)
                outputs.extend((
                    hf.short_time_fresnel_diffraction(*args),
                    aberration_ops.subaperture_fresnel(*args),
                    hf.correct_phase(field, phase),
                    aberration_ops.apply_phase(field, phase),
                ))
            return tuple(hf.sinks.numpy(output) for output in outputs)

        with hf.trace(compare).compile() as run:
            run.start()
            outputs = run.result()
        self.assertEqual(outputs[3].shape, (18, 3, 3, 5, 6))
        for native, decomposed in zip(outputs[::2], outputs[1::2]):
            np.testing.assert_allclose(decomposed, native, rtol=2e-4,
                                       atol=2e-6 * np.max(np.abs(native)))

    def test_decomposed_normalization_matches_native(self):
        from holography_demo.aberration_ops import normalize

        def compare():
            image = hf.sources.holofile(self.root / "input.holo", batch_size=5, loop=False)
            image = image.to("device").astype(hf.float32).reshape((5, 2, 4, 5, 10))
            return tuple(
                hf.sinks.numpy(result)
                for scale in (0, 1, 1e-10)
                for result in (
                    hf.normalize(image * scale, axes=(-2, -1), lo=-2, hi=3),
                    normalize(image * scale, axes=(-2, -1), lo=-2, hi=3),
                )
            )

        with hf.trace(compare).compile() as run:
            run.start()
            results = run.result()
        np.testing.assert_array_equal(results[1], np.full((35, 2, 4, 5, 10), -2))
        for reference, result in zip(results[::2], results[1::2], strict=True):
            np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)

    def test_decomposed_aberration_in_both_slope_modes(self):
        from holography_demo.pipeline import pipeline

        for mode in ("single_reference", "full_pairwise"):
            with self.subTest(mode=mode):
                config = self.config(
                    aberration={"enabled": True, "slope_mode": mode, "zernike_modes": [6, 4, 5]},
                    postprocessing={"average_window": 3},
                    viewer={"enabled": True, "cao_diagnostics": True},
                )
                with patch("holography_demo.pipeline.fresnel_diffraction", hf.fresnel_diffraction):
                    graph = hf.trace(pipeline, config)
                nodes = graph.to_json()["nodes"].values()
                kinds = {node["type"] for node in nodes}
                self.assertTrue({"FFT2", "Exp", "ShackHartmannSlopes", "ZernikeFromSlopes"} <= kinds)
                self.assertTrue({"ShortTimeFresnelDiffraction", "ZernikePhase", "CorrectPhase"}.isdisjoint(kinds))
                self.assertIn("Remainder", kinds)
                self.assertNotIn("Wrap2Pi", kinds)
                self.assertTrue({"Min", "Max", "Subtract", "Divide"} <= kinds)
                self.assertNotIn("Normalize", kinds)
                fit, = (node for node in nodes if node["type"] == "ZernikeFromSlopes")
                self.assertEqual(fit["params"]["indexes"], [6, 4, 5])
                with graph.compile(log_dir=self.root / mode, capture=False) as run:
                    run.start()
                    images = run.result()
                    self.assertEqual(run.latency("latency")["matched_outputs"], config.output_count)
                    dot = run.artifacts.compiled_dot
                self.assertEqual(images["xy"].shape, (config.output_count, 20, 20))
                self.assertEqual(images["coefficients"].shape, (config.temporal_count, 3))
                for name in ("phase", "shack_hartmann"):
                    self.assertEqual(images[name].shape, (config.temporal_count, 20, 20))
                report = dot.read_text(encoding="utf-8")
                self.assertGreaterEqual(report.count("\\n(Exp)\\nConstant (runs once per scheduler)"), 1)
                self.assertGreaterEqual(report.count("\\n(Square)\\nConstant (runs once per scheduler)"), 4)

    def test_decomposed_fresnel_matches_fused_magnitude(self):
        from holography_demo.propagation import fresnel_diffraction_decomposed

        def compare(path, complex_input, distance, shared):
            field = hf.sources.holofile(path, batch_size=2, loop=False).to("device")
            field = field.astype(hf.float32)
            if complex_input:
                field = hf.multiply(field, hf.asarray(1 + 0.5j))
            field = field.reshape((1, *field.shape), copy=False)
            args = (field, 851e-9, 20e-6, 20e-6, distance)
            if shared:
                from holography_demo.propagation import quadratic_lens
                spectrum = hf.fft.fft2(field * quadratic_lens(*field.shape[-2:], *args[1:]))
                return hf.sinks.numpy(hf.abs(spectrum)), hf.sinks.numpy(spectrum)
            return (
                hf.sinks.numpy(hf.fresnel_diffraction(*args, output_magnitude=True)),
                hf.sinks.numpy(fresnel_diffraction_decomposed(*args, output_magnitude=True)),
            )

        for width, height, complex_input, distance in (
            (20, 20, False, 0.49),
            (21, 16, True, -0.49),
            (16, 21, False, -0.49),
        ):
            with self.subTest(width=width, height=height, complex_input=complex_input):
                path = self.root / "fresnel.holo"
                write_holo(path, frames=4, width=width, height=height)
                graph = hf.trace(compare, path, complex_input, distance, False)
                self.assertIn("FFT2", {node["type"] for node in graph.to_json()["nodes"].values()})
                for enabled in (True, False):
                    with graph.compile(self.root / "fft32", fuse_tasks=enabled) as run:
                        run.start()
                        fused, decomposed = run.result()
                        dot = run.artifacts.compiled_dot
                    self.assertEqual("(FusedFFT2)" in dot.read_text(encoding="utf-8"), enabled)
                    self.assertEqual(decomposed.shape, (2, 2, height, width))
                    self.assertEqual(decomposed.dtype, np.float32)
                    np.testing.assert_allclose(decomposed, fused, rtol=2e-4,
                                               atol=2e-6 * fused.max())

        with hf.trace(compare, path, False, .49, True).compile(self.root / "shared_fft") as run:
            run.start()
            actual, raw_fft = run.result()
            dot = run.artifacts.compiled_dot
        self.assertNotIn("(FusedFFT2)", dot.read_text(encoding="utf-8"))
        np.testing.assert_allclose(actual, np.abs(raw_fft), rtol=2e-4)

    def test_demo_with_decomposed_fresnel(self):
        from holography_demo.pipeline import pipeline
        from holography_demo.propagation import fresnel_diffraction_decomposed

        config = self.config(viewer={"enabled": True})
        with patch("holography_demo.pipeline.fresnel_diffraction", fresnel_diffraction_decomposed):
            graph = hf.trace(pipeline, config)
        spec = graph.to_json()
        nodes = spec["nodes"]
        kinds = {node["type"] for node in nodes.values()}
        self.assertTrue({"Arange", "Meshgrid", "Exp", "FFT2", "Abs"} <= kinds)
        self.assertNotIn("FresnelDiffraction", kinds)
        with graph.compile(log_dir=self.root / "decomposed_graph", fuse_tasks=False) as run:
            run.start()
            self.assertEqual(run.result().shape, (config.output_count, 20, 20))
            dot = run.artifacts.compiled_dot

        labels = dot.read_text(encoding="utf-8").splitlines()
        lens, = (name for name, node in nodes.items() if node["type"] == "Exp")
        field_multiply, = (edge["to"] for edge in spec["edges"] if edge["from"] == lens)
        marker = "Constant (runs once per scheduler)"
        for name, node in nodes.items():
            kind = node["type"]
            if kind in {"Arange", "Meshgrid", "Exp", "FFT2", "Abs"} or name == field_multiply:
                label, = (line for line in labels if f'label="{name}\\n' in line)
                self.assertIn(f"\\n({kind})\\n", label)
                self.assertEqual(marker in label, kind in {"Arange", "Meshgrid", "Exp"}, name)

    def write_fp16_fixture(self, frames):
        import struct

        width, height = 512, 320
        size = frames * width * height
        header = struct.pack("<IHHIIIQB35s", 0x4F4C4F48, 66, 8, width, height, frames,
                             size, 0, bytes(35))
        with (self.root / "fp16.holo").open("wb") as fixture:
            fixture.write(header)
            fixture.truncate(len(header) + size)

    def test_fp16_demo_traces_native_pipeline(self):
        from dataclasses import replace
        from holography_demo.pipeline import pipeline

        self.write_fp16_fixture(128)
        config = self.config(
            input={"path": "fp16.holo", "mode": "looping", "load_kind": "cpu_cached",
                   "batch_size": 32},
            time={"precision": "fp16", "window_frames": 32, "component_end": 16},
            postprocessing={"average_window": 64},
        )
        nodes = hf.trace(pipeline, config).to_json()["nodes"].values()
        by_type = {node["type"]: node["params"] for node in nodes}
        self.assertNotIn("Pca", by_type)
        queue = by_type["MultiReaderBatchQueue"]
        self.assertEqual((queue["reader_count"], queue["output_stride"]), (2, 128))
        self.assertIn("PcaCovariance", by_type)
        self.assertIn("SymmetricEigendecomposition", by_type)
        self.assertEqual(by_type["PcaProjection"]["output_dtype"], "F16")
        self.assertTrue(by_type["FFT2"]["prefer_fp16"])
        self.assertEqual(by_type["DualReaderBatchQueue"]["window_size"], 64)
        self.assertEqual(by_type["Memcpy"]["target"], "Device")

        filtered = replace(config, spatial_filter={**config.spatial_filter, "enabled": True})
        corrected = replace(
            filtered, aberration={**filtered.aberration, "enabled": True}, correction_window=64
        )
        for settings, sm120 in ((filtered, True), (corrected, True), (corrected, False)):
            with self.subTest(aberration=settings.aberration["enabled"], sm120=sm120):
                with patch("holoflow._native.supports_sm120", return_value=sm120):
                    nodes = hf.trace(pipeline, settings).to_json()["nodes"].values()
                by_type = {node["type"]: node["params"] for node in nodes}
                self.assertTrue(by_type["Filter2D"]["normalize"])
                self.assertEqual(by_type["Filter2D"].get("use_fused_f16", False), sm120)
                self.assertEqual(by_type["FFT2"].get("prefer_fp16", False),
                                 sm120 or not settings.aberration["enabled"])
                if settings.aberration["enabled"]:
                    self.assertIn("ShortTimeFresnelDiffraction", by_type)
                    self.assertEqual(by_type["CorrectPhase"].get("output_cf16", False), sm120)

    def test_fp16_finite_playback_drains_and_trims_complete_windows(self):
        from holography_demo.pipeline import pipeline

        self.write_fp16_fixture(224)
        for remainder, load_kind in enumerate(("live", "cpu_cached", "gpu_cached"), 1):
            with self.subTest(load_kind=load_kind):
                with self.assertWarnsRegex(UserWarning, f"dropped {remainder} .*{remainder * 32} frames") as warning:
                    config = self.config(
                        input={"path": "fp16.holo", "load_kind": load_kind, "batch_size": 32,
                               "end_frame": 128 + 32 * remainder, "tail": "error"},
                        time={"precision": "fp16", "window_frames": 32, "component_end": 4},
                        viewer={"enabled": True},
                    )
                self.assertEqual(len(warning.warnings), 1)
                self.assertEqual(config.input["end_frame"], 128)
                self.assertEqual((config.temporal_count, config.output_count), (4, 4))
                with hf.trace(pipeline, config).compile() as run:
                    run.start()
                    self.assertEqual(run.result().shape, (4, 320, 512))
                    self.assertTrue(run.completed())

    def test_fp16_fresnel_fuses_and_matches_coarse(self):
        if not hf._native.supports_sm120():
            self.skipTest("Requires SM120")
        from holography_demo.propagation import quadratic_lens

        self.write_fp16_fixture(128)
        args = (8.51e-7, 2e-5, 2e-5, .49)

        def graph(complex_input):
            frames = hf.sources.holofile(self.root / "fp16.holo", batch_size=4, loop=False,
                                         end_frame=8).to("device")
            value = (frames.astype(hf.float32) + 32768).astype(hf.float16)
            if complex_input:
                value = (value.astype(hf.float32) * hf.asarray(1 + .25j)).astype(hf.DType.complex32)
            lens = quadratic_lens(320, 512, *args)
            spectrum = hf.fft.fft2(value.astype(hf.complex64 if complex_input else hf.float32) * lens,
                                   prefer_fp16=True)
            magnitude = hf.mean(hf.abs(spectrum), axis=-3, keepdims=not complex_input)
            coarse = hf.fresnel_diffraction(
                value, *args, output_magnitude=True, use_fused_f16=True, reduce_magnitude=True
            )
            if complex_input:
                coarse = coarse.reshape(coarse.shape[1:])
            return hf.sinks.numpy(magnitude), hf.sinks.numpy(coarse)

        for complex_input in (False, True):
            traced = hf.trace(graph, complex_input)
            for enabled in (True, False):
                with self.subTest(complex=complex_input, fusion=enabled):
                    with traced.compile(self.root / "fft_fusion", fuse_tasks=enabled) as run:
                        run.start()
                        actual, coarse = run.result()
                        compiled = run.artifacts.compiled_dot
                    self.assertEqual("(FusedFFT2)" in compiled.read_text(encoding="utf-8"), enabled)
                    self.assertTrue(np.isfinite(actual).all())
                    relative_error = np.linalg.norm(actual - coarse) / np.linalg.norm(coarse)
                    self.assertLess(relative_error, .01)

    def test_fp16_finite_playback_requires_one_group(self):
        self.write_fp16_fixture(96)
        with self.assertRaisesRegex(ValueError, "requires at least four PCA windows"):
            self.config(input={"path": "fp16.holo", "batch_size": 32},
                        time={"precision": "fp16", "window_frames": 32, "component_end": 4})

    def test_looping_cli_reports_summary(self):
        from holography_demo.cli import run_demo

        config = self.config(input={"mode": "looping", "load_kind": "gpu_cached",
                                    "max_fps": 320, "duration_seconds": 0.2},
                             time={"method": "none"}, space={"method": "none"})
        with redirect_stdout(StringIO()) as capture:
            self.assertIsNone(run_demo(config))
        self.assertRegex(capture.getvalue(), r"Latency: [1-9][0-9,]* matched, 0 unmatched")
        self.assertIn("Trace-to-drain wall time", capture.getvalue())
        self.assertFalse((self.root / "logs").exists())

    def test_cli_exports_perfetto_when_enabled(self):
        from holoflow import _native
        from holography_demo.cli import run_demo

        if not _native.perfetto_available():
            self.skipTest("Perfetto support is disabled")
        config = self.config(
            time={"method": "none"}, space={"method": "none"},
            graph={"enabled": True, "perfetto": True, "log_dir": "traces",
                   "open_browser": False},
        )
        with patch("holography_demo.graph.shutil.which", return_value=None), \
                redirect_stdout(StringIO()) as output:
            run_demo(config)
        trace, = (self.root / "traces").glob("*/trace_events.perfetto-trace")
        self.assertGreater(trace.stat().st_size, 0)
        self.assertIn(str(trace.resolve()), output.getvalue())

    def test_compiled_graph_report_with_viewer_and_without_graphviz(self):
        from holography_demo.cli import run_demo

        config = self.config(time={"method": "none"}, space={"method": "none"},
                             viewer={"enabled": True},
                             graph={"enabled": True, "log_dir": "reports", "open_browser": False})
        with patch("holography_demo.graph.shutil.which", return_value=None), \
                patch("holography_demo.graph.webbrowser.open") as browser, \
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
