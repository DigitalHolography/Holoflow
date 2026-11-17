import importlib.util
from pathlib import Path
import tempfile
import unittest

import holoflow as hf
from fixture import write_holo


def load_demo():
    path = Path(__file__).resolve().parents[2] / "example/python/simple_doppler.py"
    spec = importlib.util.spec_from_file_location("simple_doppler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TracingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.input = Path(self.directory.name) / "input.holo"
        self.output = Path(self.directory.name) / "output.holo"
        write_holo(self.input)

    def test_demo_topology(self):
        graph = hf.trace(load_demo().simple_doppler, self.input, self.output)
        spec = graph.to_json()
        self.assertEqual(len(spec["nodes"]), 15)
        self.assertEqual(len(spec["edges"]), 14)
        self.assertEqual(spec["nodes"]["doppler_writer"]["type"], "HolofileWriter")
        self.assertEqual(sum(n["type"] == "BatchQueue" for n in spec["nodes"].values()), 3)
        self.assertIn("digraph", graph.to_dot())
        self.assertFalse(self.output.exists())

    def test_static_shapes_and_source_metadata(self):
        def pipeline():
            x = hf.sources.holofile(self.input, batch_size=32)
            self.assertEqual(x.shape, (32, 8, 8))
            self.assertEqual(x.dtype, hf.uint16)
            self.assertEqual(x.device, "host")
            self.assertEqual(x.strides, (128, 16, 2))
            self.assertEqual(x.offset, 0)
        hf.trace(pipeline)

    def test_context_cleanup_and_nested_trace(self):
        def failing():
            raise ValueError("test failure")
        with self.assertRaisesRegex(ValueError, "test failure"):
            hf.trace(failing)
        with self.assertRaisesRegex(RuntimeError, "require hf.trace"):
            hf.sources.holofile(self.input)
        with self.assertRaisesRegex(RuntimeError, "Nested"):
            hf.trace(lambda: hf.trace(failing))
        hf.trace(lambda: hf.sources.holofile(self.input))

    def test_mixed_trace_and_value_conversion(self):
        saved = []
        hf.trace(lambda: saved.append(hf.sources.holofile(self.input)))
        with self.assertRaisesRegex(ValueError, "active trace"):
            hf.trace(lambda: saved[0].to("device"))
        with self.assertRaises(TypeError):
            bool(saved[0])
        with self.assertRaises(TypeError):
            saved[0].item()

    def test_explicit_ids_reserve_generated_names(self):
        def pipeline():
            x = hf.sources.holofile(self.input)
            hf.sinks.holofile(x, self.output, count=32, node_id="node_0")
        nodes = hf.trace(pipeline).to_json()["nodes"]
        self.assertEqual(nodes["node_0"]["type"], "HolofileWriter")
        self.assertIn("node_1", nodes)

    def test_math_tuple_routing_and_cross_trace(self):
        saved = []
        def pipeline():
            x = hf.arange(-2, 2)
            y = hf.arange(-1, 1)
            xx, yy = hf.meshgrid(x, y)
            self.assertEqual(xx.shape, (2, 4))
            self.assertEqual(yy.shape, (2, 4))
            saved.append(xx)
            hf.sinks.void(hf.square(xx) + hf.square(yy))
        graph = hf.trace(pipeline).to_json()
        self.assertTrue(any(e["out"] == 1 for e in graph["edges"]))
        with self.assertRaisesRegex(ValueError, "active trace"):
            hf.trace(lambda: saved[0] * hf.asarray(1))

    def test_duplicate_ids_and_invalid_settings(self):
        def pipeline():
            x = hf.sources.holofile(self.input, node_id="same")
            hf.sinks.holofile(x, self.output, count=32, node_id="same")
        with self.assertRaisesRegex(ValueError, "unique"):
            hf.trace(pipeline)
        with self.assertRaisesRegex(ValueError, "[Bb]atch"):
            hf.trace(lambda: hf.sources.holofile(self.input, batch_size=0))
        with self.assertRaises(Exception):
            hf.trace(lambda: hf.sources.holofile(self.input.with_name("missing.holo")))


if __name__ == "__main__":
    unittest.main()
