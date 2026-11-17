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
try:
    import matplotlib
except ImportError:
    raise unittest.SkipTest("Matplotlib is required for viewer tests")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "example/python"))
from fresnel_numpy import input_range, create_viewer, collect_frames
from fixture import write_holo


class FresnelViewerTest(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(input_range(163840), (0, 34784, 1024))
        self.assertEqual(input_range(163840, 320, 1024), (320, 35104, 1024))
        self.assertEqual(input_range(163840, 320, end=35104), (320, 35104, 1024))
        self.assertEqual(input_range(163840, end=32), (0, 32, 0))
        for args in ((100,), (100000, -1), (100000, 0, 0), (100000, 0, None, 33)):
            with self.assertRaises(ValueError):
                input_range(*args)

    def test_viewer_navigation(self):
        frames = np.arange(24, dtype=np.uint8).reshape(3, 2, 4)
        fig, slider = create_viewer(frames)
        self.addCleanup(plt.close, fig)
        slider.set_val(1)
        np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), frames[1])
        for key in ("right", "right"):
            fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key=key))
        self.assertEqual(slider.val, 2)
        for _ in range(4):
            fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key="left"))
        self.assertEqual(slider.val, 0)
        fig, slider = create_viewer(frames[:1])
        self.addCleanup(plt.close, fig)
        self.assertIsNone(slider)
        with self.assertRaises(ValueError):
            create_viewer(frames[:0])

    def test_native_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.holo"
            write_holo(path, frames=2080)
            result = collect_frames(str(path), 0, 2080)
            self.assertEqual(result.shape, (2, 8, 8))
            self.assertEqual(result.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
