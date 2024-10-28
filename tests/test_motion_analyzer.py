# tests/test_motion_analyzer.py

import unittest
import numpy as np
from src.motion_analysis.motion_analyzer import MotionAnalyzer

class TestMotionAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MotionAnalyzer()

    def test_compute_optical_flow(self):
        prev_gray = np.zeros((100, 100), dtype=np.uint8)
        curr_gray = np.zeros((100, 100), dtype=np.uint8)
        flow = self.analyzer.compute_optical_flow(prev_gray, curr_gray)
        self.assertEqual(flow.shape, (100, 100, 2))
    
    def test_generate_flow_mask(self):
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = self.analyzer.generate_flow_mask(flow, frame)
        self.assertEqual(mask.shape, (100, 100))
        self.assertTrue(np.all(mask >= 0) and np.all(mask <= 1))

if __name__ == '__main__':
    unittest.main()
