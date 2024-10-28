# tests/test_person_segmentation.py

import unittest
import numpy as np
import cv2
from src.motion_analysis.person_segmentation import PersonSegmenter

class TestPersonSegmenter(unittest.TestCase):
    def setUp(self):
        self.segmenter = PersonSegmenter()

    def test_get_person_mask(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a white rectangle to simulate a person
        cv2.rectangle(frame, (30, 30), (70, 70), (255, 255, 255), -1)
        mask = self.segmenter.get_person_mask(frame)
        self.assertEqual(mask.shape, (100, 100))
        # Check that the center area is detected
        self.assertTrue(mask[50, 50] == 1.0)
        # Check that the corners are not detected
        self.assertTrue(mask[10, 10] == 0.0)

if __name__ == '__main__':
    unittest.main()
