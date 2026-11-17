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

import sys
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


if __name__ == "__main__":
    unittest.main()
