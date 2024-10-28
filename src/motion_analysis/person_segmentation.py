import cv2
import numpy as np
import mediapipe as mp

class PersonSegmenter:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
    def get_person_mask(self, frame):
        """
        Get person segmentation mask using MediaPipe with stricter inward masking
        """
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        # Get segmentation mask
        results = self.segmenter.process(frame_rgb)
        if results.segmentation_mask is None:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            
        mask = results.segmentation_mask
        
        # Higher threshold for initial mask
        mask_binary = (mask > 0.8).astype(np.uint8)  # Reduced threshold back to 0.8
        
        # Find contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create strict mask from largest contour
        strict_mask = np.zeros_like(mask_binary)
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create eroded mask with gradient
            cv2.drawContours(strict_mask, [largest_contour], -1, 1, -1)
            
            # Simplified masking approach
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            eroded = cv2.erode(strict_mask, kernel, iterations=2)
            strict_mask = cv2.dilate(eroded, kernel, iterations=1)
            
            # Ensure binary mask (0 or 1)
            strict_mask = (strict_mask > 0.5).astype(np.float32)
        
        return strict_mask
        
    def __del__(self):
        self.segmenter.close()
