# src/motion_analysis/person_segmentation.py

import cv2
import numpy as np

class PersonSegmenter:
    """
    PersonSegmenter class for segmenting persons in video frames.
    Currently uses a simple background subtractor as a placeholder.
    """

    def __init__(self):
        """
        Initialize the person segmentation model.
        Replace this with actual model initialization if using a pre-trained model.
        """
        # Example: Using a simple background subtractor as a placeholder
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)

    def get_person_mask(self, frame):
        """
        Returns a binary mask where persons are detected.
        
        Args:
            frame (np.ndarray): Current frame
        
        Returns:
            np.ndarray: Binary person mask
        """
        fg_mask = self.back_sub.apply(frame)
        
        # Threshold the mask to remove shadows (if any)
        _, fg_mask = cv2.threshold(fg_mask, 250, 1, cv2.THRESH_BINARY)
        
        # Perform morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return fg_mask.astype(np.float32)
