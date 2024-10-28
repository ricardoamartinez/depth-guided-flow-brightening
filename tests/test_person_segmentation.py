# tests/test_person_segmentation.py

import unittest
import numpy as np
import cv2
from src.motion_analysis.person_segmentation import PersonSegmenter

class TestPersonSegmenter(unittest.TestCase):
    def setUp(self):
        self.segmenter = PersonSegmenter()

    def test_get_person_mask(self):
        # Create a more realistic test image that looks like a person silhouette
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Draw an oval for head
        cv2.ellipse(test_image, (50, 30), (15, 20), 0, 0, 360, (255, 255, 255), -1)
        
        # Draw a trapezoid for body
        body_pts = np.array([[35, 50], [65, 50], [70, 90], [30, 90]], dtype=np.int32)
        cv2.fillPoly(test_image, [body_pts], (255, 255, 255))
        
        mask = self.segmenter.get_person_mask(test_image)
        
        # Test that some pixels in the mask are non-zero
        self.assertTrue(np.any(mask > 0))
        
        # Test mask dimensions match input
        self.assertEqual(mask.shape, (100, 100))

if __name__ == '__main__':
    unittest.main()
