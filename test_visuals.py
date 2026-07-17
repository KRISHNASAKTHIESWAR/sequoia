import os
import tempfile
import unittest

from src.visualizations import generate_visuals


class VisualizationTests(unittest.TestCase):
    def test_generate_visuals_creates_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = generate_visuals(tmpdir)

            self.assertTrue(os.path.exists(files["comparison"]))
            self.assertTrue(os.path.exists(files["metrics"]))
            self.assertGreater(os.path.getsize(files["comparison"]), 0)
            self.assertGreater(os.path.getsize(files["metrics"]), 0)


if __name__ == "__main__":
    unittest.main()
