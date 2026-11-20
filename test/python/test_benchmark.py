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

"""Benchmark accounting and diagnostic lifecycle integration tests."""

import json
from pathlib import Path
import tempfile
import time
import unittest

import holoflow as hf
from holoflow import _native
from fixture import write_holo


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.holo"
        write_holo(self.input, frames=128)

    def pipeline(self, *, loop=False, tail="error", end_frame=128, collect=False):
        x = hf.sources.holofile(self.input, batch_size=32, loop=loop,
                               end_frame=end_frame, tail=tail, max_fps=640,
                               node_id="input")
        x = x.to("device").astype(hf.float32)
        if collect:
            return hf.sinks.numpy(x)
        hf.sinks.void(x, node_id="output")

    def test_finite_counts_results_and_timings(self):
        def slow_trace():
            time.sleep(0.02)
            return self.pipeline(collect=True)
        report = hf.benchmark(slow_trace, warmup_seconds=0)
        self.assertEqual(report.input_frames, 128)
        self.assertEqual(report.results.shape, (128, 8, 8))
        self.assertGreater(report.makespan_seconds, report.timings["execution_seconds"])
        self.assertGreaterEqual(report.timings["trace_seconds"], 0.02)
        self.assertFalse(report.diagnostics_enabled)
        self.assertIsNone(report.artifacts)
        self.assertNotIn("results", report.to_json())
        if report.input_frames_per_second is not None:
            self.assertAlmostEqual(report.input_frames_per_second,
                                   report.steady_input_frames / report.steady_seconds)

    def test_dropped_tail_and_short_warmup(self):
        report = hf.benchmark(self.pipeline, kwargs={"tail": "drop", "end_frame": 127},
                              warmup_seconds=5)
        self.assertEqual(report.input_frames, 96)
        self.assertEqual(report.steady_input_frames, 0)
        self.assertIsNone(report.input_frames_per_second)

    def test_looping_warmup_and_artifacts(self):
        report = hf.benchmark(self.pipeline, kwargs={"loop": True}, mode="looping",
                              warmup_seconds=0.1, duration_seconds=0.2,
                              log_dir=self.root / "logs")
        self.assertGreater(report.input_frames, report.steady_input_frames)
        self.assertGreater(report.input_frames_per_second, 0)
        self.assertGreaterEqual(report.steady_seconds, 0.19)
        self.assertTrue(report.artifacts.descriptor_json.is_file())
        self.assertTrue(report.artifacts.descriptor_dot.is_file())
        self.assertTrue(report.artifacts.compiled_dot.is_file())
        self.assertTrue(report.artifacts.compiler_log.is_file())
        self.assertEqual(report.artifacts.perfetto.is_file(), report.perfetto_available)
        saved = json.loads(report.artifacts.report.read_text())
        self.assertEqual(saved["input_frames"], report.input_frames)

    def test_progress_and_manual_artifact_lifetime(self):
        graph = hf.trace(self.pipeline)
        with graph.compile(log_dir=self.root) as run:
            artifacts = run.artifacts
            run.start()
            before = run.progress()
            time.sleep(0.03)
            after = run.progress()
            run.wait()
            final = run.progress()
        self.assertGreaterEqual(after["timestamp_seconds"], before["timestamp_seconds"])
        self.assertGreaterEqual(after["successful_emissions"]["input"],
                                before["successful_emissions"]["input"])
        self.assertEqual(final["successful_emissions"]["input"], 4)
        self.assertEqual(final["successful_emissions"]["output"], 4)
        self.assertTrue(artifacts.compiled_dot.is_file())
        second = graph.compile(log_dir=self.root)
        self.addCleanup(second.close)
        self.assertNotEqual(artifacts.directory, second.artifacts.directory)

    def test_multiple_sources_and_mode_validation(self):
        def two():
            self.pipeline()
            hf.sinks.void(hf.sources.holofile(self.input, loop=False, node_id="second"))
        with self.assertRaisesRegex(ValueError, "source_node_id"):
            hf.benchmark(two)
        self.assertEqual(hf.benchmark(two, source_node_id="input").input_frames, 128)
        with self.assertRaisesRegex(ValueError, "match"):
            hf.benchmark(self.pipeline, mode="looping")
        with self.assertRaises(ValueError):
            hf.benchmark(self.pipeline, warmup_seconds=-1)

    def test_compile_failure_preserves_diagnostics(self):
        spec = hf.trace(self.pipeline).to_json()
        spec["nodes"]["input"]["type"] = "MissingFactory"
        graph = hf.Graph(_native.Graph(json.dumps(spec)))
        with self.assertRaises(Exception) as caught:
            graph.compile(log_dir=self.root / "failed")
        directory = next((self.root / "failed").iterdir())
        self.assertTrue((directory / "descriptor.json").is_file())
        self.assertTrue((directory / "compiler.log").is_file())
        self.assertTrue(any("Holoflow diagnostics" in note
                            for note in caught.exception.__notes__))

    @unittest.skipUnless(_native.perfetto_available(), "Perfetto not compiled in")
    def test_conflicting_capture(self):
        first = _native.ProfileCapture(str(self.root / "first"))
        try:
            with self.assertRaisesRegex(RuntimeError, "capture is active"):
                hf.trace(self.pipeline).compile(log_dir=self.root)
        finally:
            first.close()

    def test_execution_failure_finalizes_artifacts(self):
        def pipeline():
            x = hf.sources.holofile(self.input, loop=False, node_id="input")
            return hf.sinks.numpy(x, length=256)
        with self.assertRaises(hf.ExecutionError):
            hf.benchmark(pipeline, log_dir=self.root / "failed", warmup_seconds=0)
        directory = next((self.root / "failed").iterdir())
        self.assertEqual(json.loads((directory / "benchmark.json").read_text())["status"], "failed")
        self.assertTrue((directory / "descriptor.json").is_file())
        self.assertTrue((directory / "compilation_success.dot").is_file())
        self.assertEqual((directory / "trace_events.perfetto-trace").is_file(),
                         _native.perfetto_available())

    def test_progress_across_queue_sections(self):
        def pipeline():
            x = hf.sources.holofile(self.input, batch_size=32, loop=False, node_id="input")
            x = hf.batch_queue(x.to("device"), 128, 32, 32)
            x = hf.throughput(x, interval_ms=60_000, node_id="device_fps")
            x = x.astype(hf.float32, node_id="conversion")
            x = x.astype(hf.complex64, node_id="second_conversion")
            x = hf.batch_queue(x, 128, 32, 32)
            return hf.sinks.numpy(x)
        with hf.trace(pipeline).compile(log_dir=self.root) as run:
            run.start()
            run.result()
            counts = run.progress()["successful_emissions"]
            self.assertEqual(counts["input"], 4)
            self.assertEqual(counts["conversion"], 4)
            self.assertEqual(counts["second_conversion"], 4)
            event = run.receive_event()
            self.assertEqual(event.node_id, "device_fps")
            self.assertEqual(event.data["type"], "throughput_summary")
            self.assertEqual(event.data["total_frames"], 128)
            self.assertTrue(event.data["completed"])
        diagnostics = json.loads((run.artifacts.directory / "section_cuda_graphs.json").read_text())
        measured_section = next(section for section in diagnostics
                                if any(task["name"] == "device_fps" for task in section["tasks"]))
        self.assertGreater(measured_section["launches"], 0)

    def test_throughput_finite_preserves_data_and_counts_dropped_tail(self):
        def pipeline(measure):
            frames = hf.sources.holofile(self.input, loop=False, end_frame=127, tail="drop")
            if measure:
                frames = hf.throughput(frames, axis=-3, interval_ms=60_000, node_id="fps")
            return hf.sinks.numpy(frames)

        with hf.trace(pipeline, False).compile() as reference:
            reference.start()
            expected = reference.result()
        with hf.trace(pipeline, True).compile() as run:
            run.start()
            self.assertEqual(run.result().tobytes(), expected.tobytes())
            event = run.receive_event()
            self.assertEqual(event.node_id, "fps")
            self.assertEqual(event.data["total_frames"], 96)
            self.assertAlmostEqual(event.data["average_fps"], 96 / event.data["elapsed_seconds"])
            self.assertIsNone(run.receive_event())

    def test_throughput_independent_periods_and_idle_windows(self):
        def pipeline():
            frames = hf.sources.holofile(self.input, loop=True, max_fps=128)
            frames = hf.throughput(frames, interval_ms=15, node_id="fast")
            hf.sinks.void(hf.throughput(frames, axis=None, interval_ms=90, node_id="slow"))

        with hf.trace(pipeline).compile() as run:
            run.start()
            samples = {"fast": [], "slow": []}
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                event = run.receive_event()
                if event is not None:
                    samples[event.node_id].append(event.data)
                if (len(samples["slow"]) >= 2
                        and any(s["frames"] == 0 for s in samples["fast"])
                        and any(s["total_frames"] > 0 for s in samples["slow"])):
                    break
                time.sleep(0.002)
            run.stop()
            run.wait()
            summaries = {event.node_id: event.data
                         for event in iter(run.receive_event, None)}
        self.assertGreaterEqual(len(samples["slow"]), 2)
        self.assertGreater(len(samples["fast"]), len(samples["slow"]))
        self.assertTrue(any(s["fps"] == 0 for s in samples["fast"]))
        self.assertEqual(summaries["fast"]["total_frames"], 32 * summaries["slow"]["total_frames"])
        self.assertFalse(summaries["fast"]["completed"])

    def test_throughput_coexists_with_recording_events(self):
        from test_execution import wait_recording

        output = self.root / "recorded.holo"

        def pipeline():
            frames = hf.sources.holofile(self.input, loop=True, max_fps=640)
            frames = hf.throughput(frames, interval_ms=5, node_id="fps")
            hf.sinks.holofile(frames, output, count=64, node_id="writer")

        with hf.trace(pipeline).compile() as run:
            run.start()
            wait_recording(run, "writer", output)
            run.stop()
            run.wait()
            events = list(iter(run.receive_event, None))
            self.assertTrue(any(event.node_id == "fps" and
                                event.data["type"] == "throughput_summary" for event in events))

    def test_stopped_finite_execution_is_cancelled(self):
        with hf.trace(self.pipeline, collect=True).compile(log_dir=self.root) as run:
            run.start()
            run.stop()
            with self.assertRaises(hf.CancelledError):
                run.result()
        self.assertTrue(run.artifacts.compiled_dot.is_file())


if __name__ == "__main__":
    unittest.main()
