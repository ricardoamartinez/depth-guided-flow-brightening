import unittest
import numpy as np
import cv2
import sys
import os
import torch
from unittest.mock import patch, MagicMock

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.optical_flow import OpticalFlowEstimator, compute_optical_flow

class TestOpticalFlowEstimator(unittest.TestCase):
    def setUp(self):
        self.frame1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.frame2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_estimate_flow_farneback(self):
        estimator = OpticalFlowEstimator(method='Farneback')
        flow = estimator.estimate_flow(self.frame1, self.frame2)
        
        self.assertEqual(flow.shape, (100, 100, 2))
        self.assertTrue(np.any(flow != 0))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_estimate_flow_raft(self):
        model_path = os.path.join(project_root, 'RAFT', 'models', 'raft-things.pth')
        if not os.path.exists(model_path):
            self.skipTest("RAFT model not found")
        
        estimator = OpticalFlowEstimator(method='RAFT', model_path=model_path)
        flow = estimator.estimate_flow(self.frame1, self.frame2)
        
        self.assertEqual(flow.shape, (100, 100, 2))
        self.assertTrue(np.any(flow != 0))

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            OpticalFlowEstimator(method='InvalidMethod')

    def test_raft_invalid_path(self):
        with self.assertRaises(ValueError):
            OpticalFlowEstimator(method='RAFT', model_path='invalid_path')

    def test_compute_optical_flow_with_raft(self):
        frame1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        model_path = os.path.join(project_root, 'RAFT', 'models', 'raft-things.pth')
        if not os.path.exists(model_path):
            self.skipTest("RAFT model not found")
        
        flow = compute_optical_flow(frame1, frame2, method='RAFT', model_path=model_path)
        
        self.assertEqual(flow.shape, (100, 100, 2))
        self.assertTrue(np.any(flow != 0))

if __name__ == '__main__':
    unittest.main()
