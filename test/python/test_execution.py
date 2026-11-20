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

from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import holoflow as hf
from fixture import write_holo
from test_tracing import load_demo
from native_reference import reference_graph


def wait_recording(executable, node_id, path):
    if not executable.send_event(node_id, {"type": "start_recording", "record_path": str(path)}):
        raise RuntimeError("Event queue full")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        event = executable.receive_event()
        if event is None:
            time.sleep(.005)
        elif event.node_id == node_id:
            if event.data["type"] == "recording_failed":
                raise RuntimeError(event.data["message"])
            if event.data["type"] == "recording_finished":
                return
    raise TimeoutError("Recording did not complete")


class ExecutionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.input = Path(self.directory.name) / "input.holo"
        self.output = Path(self.directory.name) / "output.holo"
        write_holo(self.input)

    def test_pca_demo_records_and_cleans_up(self):
        graph = hf.trace(load_demo().simple_doppler, self.input, self.output)
        executable = graph.compile()
        del graph
        with executable:
            with self.assertRaises(ValueError):
                executable.send_event("missing", {"type": "start_recording"})
            with self.assertRaises((TypeError, ValueError)):
                executable.send_event("doppler_writer", {"invalid": object()})
            executable.start()
            wait_recording(executable, "doppler_writer", self.output)
            executable.stop()
            executable.wait()
            executable.wait()
            self.assertIsInstance(executable.metrics(), dict)
        executable.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            executable.start()
        header = struct.unpack("<IHHIIIQB35s", self.output.read_bytes()[:64])
        self.assertEqual(header[0], 0x4F4C4F48)
        self.assertEqual(header[2:6], (8, 8, 8, 100))
        self.assertGreaterEqual(self.output.stat().st_size, 64 + 100 * 8 * 8)

        reference = self.output.with_name("reference.holo")
        with reference_graph(self.input, reference).compile() as native:
            native.start()
            wait_recording(native, "doppler_writer", reference)
            native.stop()
            native.wait()
        actual = self.output.read_bytes()[64:64 + 100 * 8 * 8]
        expected = reference.read_bytes()[64:64 + 100 * 8 * 8]
        self.assertEqual(len(actual), len(expected))
        self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 2)

    def test_fresnel_variants_with_void_sink(self):
        import importlib.util
        for filename in ("fresnel_doppler_decomposed.py", "fresnel_doppler.py"):
            path = Path(__file__).with_name(filename)
            spec = importlib.util.spec_from_file_location(path.stem, path)
            demo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(demo)
            demo.INPUT_PATH = self.input
            pipeline = demo.pipeline
            with self.subTest(pipeline=pipeline.__name__):
                graph = hf.trace(pipeline)
                nodes = graph.to_json()["nodes"]
                self.assertEqual(nodes["output"]["type"], "Void")
                kinds = [n["type"] for n in nodes.values()]
                self.assertNotIn("DualReaderBatchQueue", kinds)
                self.assertNotIn("Reshape", kinds)
                self.assertIn("FFT2" if filename == "fresnel_doppler_decomposed.py" else "FresnelDiffraction", kinds)
                with graph.compile() as executable:
                    executable.start()
                    time.sleep(1)
                    executable.receive_event()
                    executable.stop()
                    executable.wait()

    def test_worker_io_error_reaches_python(self):
        # Keep valid metadata but remove the declared frame payload.
        self.input.write_bytes(self.input.read_bytes()[:64])
        graph = hf.trace(lambda: hf.sinks.void(hf.sources.holofile(self.input, batch_size=32)))
        executable = graph.compile()
        try:
            executable.start()
            deadline = time.monotonic() + 10
            with self.assertRaises(hf.ExecutionError):
                while time.monotonic() < deadline:
                    executable.receive_event()
                    time.sleep(.005)
                self.fail("Worker failure was not reported")
            with self.assertRaises(hf.ExecutionError):
                executable.wait()
            with self.assertRaises(hf.ExecutionError):
                executable.start()
        finally:
            try:
                executable.close()
            except hf.ExecutionError:
                pass

    @unittest.skipUnless((Path(hf.__file__).parent / "_assets/nvrtc").is_dir(),
                         "Requires the installed package assets")
    def test_packaged_assets_with_small_pca_depth(self):
        assets = Path(hf.__file__).parent / "_assets/nvrtc"
        self.assertTrue((assets / "mathdx/lib/libcusolverdx.fatbin").is_file())
        # Depths below 256 try cuSolverDx; native cuSOLVER fallback remains available.
        graph = hf.trace(load_demo().simple_doppler, self.input, self.output,
                         time_window=64, component_end=32)
        with graph.compile() as executable:
            executable.start()
            wait_recording(executable, "doppler_writer", self.output)
            executable.stop()
            executable.wait()
        self.assertTrue(self.output.is_file())

    def test_small_depth_requires_cusolverdx(self):
        # Check a child process's native diagnostics: numerical success alone can
        # hide a failed Dx compilation followed by conventional cuSOLVER fallback.
        for depth in (2, 64):
            with self.subTest(depth=depth):
                script = (
                    f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
                    "import holoflow as hf; from test_tracing import load_demo; "
                    "from test_execution import wait_recording; "
                    f"graph = hf.trace(load_demo().simple_doppler, {str(self.input)!r}, "
                    f"{str(self.output)!r}, time_window={depth}, component_begin=0, "
                    f"component_end={min(depth, 32)}); "
                    "executable = graph.compile(); executable.start(); "
                    f"wait_recording(executable, 'doppler_writer', {str(self.output)!r}); "
                    "executable.stop(); executable.wait(); executable.close()"
                )
                result = subprocess.run([sys.executable, "-c", script],
                                        capture_output=True, text=True, timeout=120)
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, output)
                self.assertIn("Selected backend: cuSolverDx", output)
                self.assertNotIn("Falling back to the conventional", output)

    def test_recording_failure_keeps_node_identity(self):
        invalid = self.output / "missing" / "out.holo"
        def pipeline():
            x = hf.sources.holofile(self.input, batch_size=32)
            hf.sinks.holofile(x, invalid, count=32, node_id="writer")
        with hf.trace(pipeline).compile() as executable:
            executable.start()
            with self.assertRaises(RuntimeError):
                wait_recording(executable, "writer", invalid)

    def test_wait_releases_gil_for_stop_from_another_thread(self):
        graph = hf.trace(lambda: hf.sinks.void(hf.sources.holofile(self.input, batch_size=32)))
        with graph.compile() as executable:
            executable.start()
            ready = threading.Event()
            errors = []
            def wait():
                ready.set()
                try:
                    executable.wait()
                except Exception as error:
                    errors.append(error)
            worker = threading.Thread(target=wait, daemon=True)
            worker.start()
            self.assertTrue(ready.wait(5))
            executable.stop()
            worker.join(10)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])

    def test_full_event_queue_reports_false(self):
        def pipeline():
            hf.sources.holofile(self.input, node_id="source")
        with hf.trace(pipeline).compile() as executable:
            results = [executable.send_event("source", {"command": "queued"})
                       for _ in range(1025)]
            self.assertTrue(results[0])
            self.assertFalse(results[-1])


if __name__ == "__main__":
    unittest.main()
