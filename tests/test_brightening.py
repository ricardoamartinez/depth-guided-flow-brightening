# tests/test_brightening.py

import unittest
import numpy as np
from src.brightening import Brightener

class TestBrightener(unittest.TestCase):
    def setUp(self):
        self.brightener = Brightener(alpha=1.0, beta=0.0)
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.flow = np.ones((100, 100, 2), dtype=np.float32)

    def test_compute_flow_magnitude(self):
        magnitude = self.brightener.compute_flow_magnitude(self.flow)
        expected = np.sqrt(2) * np.ones((100, 100))
        np.testing.assert_allclose(magnitude, expected, rtol=1e-5)

    def test_brighten_regions(self):
        flow_magnitude = self.brightener.compute_flow_magnitude(self.flow)
        brightened = self.brightener.brighten_regions(self.frame, flow_magnitude)
        expected = np.ones_like(self.frame) * 255
        np.testing.assert_array_equal(brightened, expected)

    def test_process(self):
        brightened = self.brightener.process(self.frame, self.flow)
        expected = np.ones_like(self.frame) * 255
        np.testing.assert_array_equal(brightened, expected)

if __name__ == '__main__':
    unittest.main()
