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

"""Finite native stream, EOF draining, and NumPy ownership integration tests."""
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
import unittest

import numpy as np
import holoflow as hf
from fixture import write_holo


class FiniteTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "input.holo"
        write_holo(self.path, frames=96, width=4, height=3)
        self.expected = np.frombuffer(self.path.read_bytes(), dtype=np.uint16, offset=64).reshape(96, 3, 4)

    def collect(self, pipeline):
        with hf.trace(pipeline).compile() as run:
            run.start()
            result = run.result()
            if isinstance(result, np.ndarray):
                self.assertIs(run.result(), result)
        return result

    def source(self, **kwargs):
        return hf.sources.holofile(self.path, batch_size=8, loop=False, **kwargs)

    def test_filter2d_real_and_complex_against_fft(self):
        for complex_input in (False, True):
            def pipeline():
                x = self.source().to("device").astype(hf.float32)
                if complex_input:
                    x = x.astype(hf.complex64)
                return hf.sinks.numpy(hf.filter2d(x, 0, 100, 0, 0))
            # Native FFT forward/inverse pairs are intentionally unnormalized.
            expected = np.fft.ifft2(np.fft.fft2(self.expected, axes=(-2, -1)), axes=(-2, -1)) * 12
            if complex_input:
                # Native complex mask stores the pass coefficient in both components.
                expected *= 1 + 1j
            else:
                expected = expected.real
            np.testing.assert_allclose(self.collect(pipeline), expected, rtol=1e-5, atol=0.01)

    def test_flatfield_gaussian_reference(self):
        sy, sx = 0.75, 1.25
        def pipeline():
            return hf.sinks.numpy(hf.flatfield(self.source().to("device").astype(hf.float32), sy, sx))
        background = self.expected.astype(np.float64)
        for axis, sigma in ((-1, sx), (-2, sy)):
            radius = int(np.ceil(3 * sigma))
            kernel = np.exp(-np.arange(-radius, radius + 1)**2 / (2 * sigma**2))
            kernel /= kernel.sum()
            background = np.apply_along_axis(
                lambda row: np.convolve(np.pad(row, (radius, radius), mode="edge"), kernel, mode="valid"),
                axis, background)
        np.testing.assert_allclose(self.collect(pipeline), self.expected - background, atol=0.002)

    def test_spatial_filter_validation(self):
        for op in (lambda x: hf.filter2d(x, -1, 10), lambda x: hf.flatfield(x, -1, 1)):
            with self.assertRaises(ValueError):
                hf.trace(lambda: hf.sinks.numpy(op(self.source().to("device").astype(hf.float32))))
        for op in (lambda x: hf.filter2d(x, 0, 10), lambda x: hf.flatfield(x, 1, 1)):
            with self.assertRaises(ValueError):
                hf.trace(lambda: hf.sinks.numpy(op(self.source())))

    def test_source_copy_and_lifetime(self):
        result = self.collect(lambda: hf.sinks.numpy(self.source()))
        np.testing.assert_array_equal(result, self.expected)
        self.assertEqual(result.dtype, np.uint16)
        self.assertTrue(result.flags.c_contiguous)
        self.assertIsNotNone(result.base)
        # A later executable must not reuse or free the first result's backing storage.
        self.collect(lambda: hf.sinks.numpy(self.source()))
        np.testing.assert_array_equal(result, self.expected)

    def test_cached_range_and_gpu_transfer(self):
        for kind in ("live", "cpu_cached", "gpu_cached"):
            with self.subTest(kind=kind):
                result = self.collect(lambda: hf.sinks.numpy(
                    self.source(load_kind=kind, start_frame=16, end_frame=80)))
                np.testing.assert_array_equal(result, self.expected[16:80])

    def test_device_conversion_and_fanout(self):
        def pipeline():
            x = self.source()
            return {"raw": hf.sinks.numpy(x),
                    "float": hf.sinks.numpy(x.to("device").astype(hf.float32))}
        results = self.collect(pipeline)
        np.testing.assert_array_equal(results["raw"], self.expected)
        np.testing.assert_array_equal(results["float"], self.expected.astype(np.float32))

    def test_complex_values_survive_device_collection(self):
        def pipeline():
            x = self.source().to("device").astype(hf.float32).astype(hf.complex64)
            return hf.sinks.numpy(x + hf.asarray(1j))
        actual = self.collect(pipeline)
        self.assertEqual(actual.dtype, np.complex64)
        np.testing.assert_array_equal(actual, self.expected.astype(np.complex64) + 1j)

    def test_device_concatenation_nonzero_axis(self):
        actual = self.collect(lambda: hf.sinks.numpy(self.source().to("device"), axis=1))
        expected = np.concatenate(np.split(self.expected, 12), axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_concatenate_nonzero_and_negative_axes(self):
        for axis in (1, -1):
            with self.subTest(axis=axis):
                actual = self.collect(lambda: hf.sinks.numpy(self.source(), axis=axis))
                expected = np.concatenate(np.split(self.expected, 12), axis=axis)
                np.testing.assert_array_equal(actual, expected)

    def test_queues_drain_under_backpressure(self):
        def pipeline():
            x = hf.batch_queue(self.source(), 16, 16, 16)
            x = hf.batch_queue(x.to("host"), 32, 32, 32)
            return hf.sinks.numpy(x)
        np.testing.assert_array_equal(self.collect(pipeline), self.expected)

    def test_gpu_math_constants_and_queued_eof(self):
        def pipeline():
            x = self.source().to("device").astype(hf.float32)
            x = hf.batch_queue(x, 16, 8, 8)
            y = hf.abs(hf.fft.fft2((x + hf.asarray(1)).astype(hf.complex64)))
            y = hf.batch_queue(y, 16, 8, 8)
            return hf.sinks.numpy(y)
        actual = self.collect(pipeline)
        expected = np.abs(np.fft.fft2(self.expected.astype(np.float32) + 1, axes=(-2, -1)))
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=0.02)

    def test_numpy_style_fft_and_scalar_operators(self):
        def pipeline():
            x = self.source().to("device")
            real = x + 2
            complex_field = 0.5j + real
            scaled = 2 * complex_field * (1 + 0.5j)
            return tuple(hf.sinks.numpy(y) for y in (
                hf.fft.fftshift(scaled),
                hf.fft.fftshift(scaled, axes=(-2, -1)),
                hf.fft.fftshift(scaled, -1),
                hf.fft.fftshift(scaled, ()),
                hf.fftshift(scaled),
                hf.fft.fft2(real),
                hf.fft.fft2(real, norm="forward"),
                hf.fft.fft2(real, norm="ortho"),
                hf.fft2(real.astype(hf.complex64)),
            ))

        nodes = hf.trace(pipeline).to_json()["nodes"]
        self.assertTrue({"AsArray", "Add", "Multiply"} <= {n["type"] for n in nodes.values()})
        actual = self.collect(pipeline)
        real = self.expected.astype(np.float32) + 2
        scaled = 2 * (0.5j + real) * (1 + 0.5j)
        # Default shift runs on each source emission, including its batch axis.
        batches = scaled.reshape(-1, 8, *scaled.shape[1:])
        shifted = np.fft.fftshift(batches, axes=(1, 2, 3)).reshape(scaled.shape)
        np.testing.assert_array_equal(actual[0], shifted)
        for result, axes in zip(actual[1:5], ((-2, -1), -1, (), (-2, -1))):
            np.testing.assert_array_equal(result, np.fft.fftshift(scaled, axes=axes))
        for result, norm in zip(actual[5:], (None, "forward", "ortho", None)):
            np.testing.assert_allclose(result, np.fft.fft2(real, norm=norm), rtol=1e-5, atol=0.02)
        with self.assertRaisesRegex(ValueError, "last two axes"):
            hf.trace(lambda: hf.fft.fft2(self.source(), axes=(0, 1)))

    def test_captured_conversion_does_not_launch_after_eof(self):
        # Native diagnostics prove that this exercises actual CUDA graph replay,
        # rather than merely accepting a graph-ineligible pipeline.
        script = """
import holoflow as hf
import numpy as np
import sys

def pipeline():
    x = hf.sources.holofile(sys.argv[1], batch_size=8, loop=False).to("device")
    x = hf.batch_queue(x, 16, 8, 8)
    x = x.astype(hf.float32)
    x = hf.batch_queue(x, 16, 8, 8)
    return hf.sinks.numpy(x)

with hf.trace(pipeline).compile() as run:
    run.start()
    result = run.result()
expected = np.frombuffer(open(sys.argv[1], "rb").read(), dtype=np.uint16, offset=64).reshape(96, 3, 4)
np.testing.assert_array_equal(result, expected)
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.path)],
                                capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertRegex(output, r"shutdown: 12 launches, 1 ordinary iterations")

    def test_empty_selected_range(self):
        for kind in ("live", "cpu_cached", "gpu_cached"):
            with self.subTest(kind=kind):
                actual = self.collect(lambda: hf.sinks.numpy(
                    self.source(start_frame=16, end_frame=16, load_kind=kind)))
                self.assertEqual(actual.shape, (0, 3, 4))
                self.assertEqual(actual.dtype, np.uint16)

    def test_dual_reader_ports_close_together(self):
        def pipeline():
            current, delayed, valid = hf.dual_reader_batch_queue(self.source(), 16, 5)
            return (hf.sinks.numpy(current), hf.sinks.numpy(delayed), hf.sinks.numpy(valid))
        current, delayed, valid = self.collect(pipeline)
        np.testing.assert_array_equal(current, self.expected)
        np.testing.assert_array_equal(delayed[2:], self.expected[:-2])
        np.testing.assert_array_equal(valid, np.r_[np.zeros(4, dtype=np.uint8), np.ones(92, dtype=np.uint8)])

    def test_finite_void_branch_without_returned_collectors(self):
        with hf.trace(lambda: hf.sinks.void(self.source())).compile() as run:
            run.start()
            self.assertIsNone(run.result())

    def test_truncated_input_is_failure_not_eof(self):
        self.path.write_bytes(self.path.read_bytes()[:64])
        with self.assertRaises(hf.ExecutionError):
            self.collect(lambda: hf.sinks.numpy(self.source()))

    def test_source_tail_policies(self):
        with self.assertRaisesRegex(ValueError, "Incomplete final source batch"):
            hf.trace(lambda: hf.sinks.numpy(self.source(end_frame=95)))
        result = self.collect(lambda: hf.sinks.numpy(self.source(end_frame=95, tail="drop")))
        np.testing.assert_array_equal(result, self.expected[:88])

    def test_batch_queue_tail_policies(self):
        def pipeline(tail):
            x = hf.batch_queue(self.source(end_frame=88), 32, 32, 32, tail=tail)
            return hf.sinks.numpy(x)
        with self.assertRaisesRegex(hf.ExecutionError, "Incomplete final BatchQueue"):
            self.collect(lambda: pipeline("error"))
        np.testing.assert_array_equal(self.collect(lambda: pipeline("drop")), self.expected[:64])

    def test_stride_skipping_is_not_a_tail_error(self):
        def pipeline():
            return hf.sinks.numpy(hf.batch_queue(self.source(), 32, 8, 16))
        expected = np.concatenate([self.expected[i:i+8] for i in range(0, 96, 16)])
        np.testing.assert_array_equal(self.collect(pipeline), expected)

    def test_sliding_average_full_windows_and_empty_result(self):
        for n in (3, 10):
            with self.subTest(n=n):
                def pipeline():
                    x = hf.sources.holofile(self.path, batch_size=1, end_frame=n, loop=False)
                    x = x.to("device").astype(hf.float32)
                    return hf.sinks.numpy(hf.sliding_average(x, 4, 4))
                actual = self.collect(pipeline)
                expected = np.stack([self.expected[i:i+4].mean(axis=0) for i in range(n-3)]) if n >= 4 else np.empty((0, 3, 4))
                np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_declared_length_checked_at_eof_and_overflow(self):
        for length, message in ((88, "overflow"), (104, "underflow")):
            with self.subTest(length=length):
                with self.assertRaisesRegex(hf.ExecutionError, message):
                    self.collect(lambda: hf.sinks.numpy(self.source(), length=length))

    def test_unequal_streams_fail(self):
        def pipeline():
            a = self.source()
            b = self.source(end_frame=88)
            return hf.sinks.numpy(a.to("device").astype(hf.float32) + b.to("device").astype(hf.float32))
        with self.assertRaisesRegex(hf.ExecutionError, "unequal lengths"):
            self.collect(pipeline)

    def test_whole_graph_failure(self):
        def pipeline():
            x = self.source()
            hf.sinks.numpy(x, length=104)  # Unreturned failing side branch.
            return hf.sinks.numpy(x)
        with self.assertRaisesRegex(hf.ExecutionError, "underflow"):
            self.collect(pipeline)

    def test_cancellation_and_wait_releases_gil(self):
        def pipeline():
            return hf.sinks.numpy(self.source(max_fps=1))
        with hf.trace(pipeline).compile() as run:
            with self.assertRaisesRegex(RuntimeError, "not been started"):
                run.result()
            run.start()
            errors = []
            ready = threading.Event()
            def wait():
                ready.set()
                try:
                    run.result()
                except Exception as exc:
                    errors.append(exc)
            worker = threading.Thread(target=wait)
            worker.start()
            self.assertTrue(ready.wait(5))
            run.stop()
            worker.join(10)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], hf.CancelledError)

    def test_nested_returns_and_duplicate_handles(self):
        def pipeline():
            result = hf.sinks.numpy(self.source())
            return {"items": (result, [result]), "none": None}
        result = self.collect(pipeline)
        self.assertIs(result["items"][0], result["items"][1][0])
        self.assertIsNone(result["none"])

    def test_reject_looping_and_unsupported_finite_operator(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            hf.trace(lambda: hf.sinks.numpy(hf.sources.holofile(self.path))).compile()
        def unsupported():
            hf.sinks.holofile(self.source(), self.path.with_name("out.holo"), count=96)
        with self.assertRaisesRegex(ValueError, "finite-stream contract"):
            hf.trace(unsupported).compile()

    def test_invalid_trace_returns_and_cross_trace_handles(self):
        with self.assertRaises(TypeError):
            hf.trace(lambda: self.source())
        saved = []
        def capture():
            saved.append(hf.sinks.numpy(self.source()))
        hf.trace(capture)
        def invalid():
            self.source()
            return saved[0]
        with self.assertRaisesRegex(ValueError, "active trace"):
            hf.trace(invalid)


if __name__ == "__main__":
    unittest.main()
