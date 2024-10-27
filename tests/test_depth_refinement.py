import unittest
import numpy as np
import os
import sys
import tempfile
import cv2  # Add this import

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from depth_refinement import DepthRefiner

class TestDepthRefiner(unittest.TestCase):
    def setUp(self):
        self.refiner = DepthRefiner(near=0.0, far=20.0)
        self.bright_mask = np.ones((100, 100), dtype=np.float32)
        
    def test_load_depth_map(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_filename = temp_file.name
            depth = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
            cv2.imwrite(temp_filename, depth)
        
        loaded_depth = self.refiner.load_depth_map(temp_filename)
        self.assertEqual(loaded_depth.shape, (100, 100))
        self.assertTrue(np.all(loaded_depth >= 0) and np.all(loaded_depth <= 20))
        
        os.unlink(temp_filename)

    def test_refine_brightening(self):
        depth_map = np.linspace(0, 20, 10000).reshape(100, 100)
        refined_mask = self.refiner.refine_brightening(self.bright_mask, depth_map)
        
        self.assertEqual(refined_mask.shape, (100, 100))
        self.assertTrue(np.all(refined_mask >= 0) and np.all(refined_mask <= 1))
        self.assertTrue(np.allclose(refined_mask[0, 0], 1.0))
        self.assertTrue(np.allclose(refined_mask[-1, -1], 0.0))

    def test_process(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_filename = temp_file.name
            depth = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
            cv2.imwrite(temp_filename, depth)
        
        refined_mask = self.refiner.process(self.bright_mask, temp_filename)
        
        self.assertEqual(refined_mask.shape, (100, 100))
        self.assertTrue(np.all(refined_mask >= 0) and np.all(refined_mask <= 1))
        
        os.unlink(temp_filename)

if __name__ == '__main__':
    unittest.main()
