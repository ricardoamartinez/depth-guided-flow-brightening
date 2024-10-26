# tests/test_brightening.py

import unittest
import numpy as np
import cv2
from src.brightening import Brightener

class TestBrightener(unittest.TestCase):
    def setUp(self):
        self.brightener = Brightener(alpha=1.0, beta=0.0)
        # Create a dummy black frame
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create a dummy flow magnitude
        self.flow_magnitude = np.ones((100, 100), dtype=np.float32)

    def test_brighten_regions(self):
        brightened = self.brightener.brighten_regions(self.frame, self.flow_magnitude)
        # Since the frame is black and bright_mask is 1, the brightened frame should be 255
        expected = np.ones_like(self.frame) * 255
        self.assertTrue(np.array_equal(brightened, expected), "Brightened frame does not match expected output.")

if __name__ == '__main__':
    unittest.main()
