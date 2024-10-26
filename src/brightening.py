# src/brightening.py

import numpy as np
import cv2

class Brightener:
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize the Brightener.

        Args:
            alpha (float): Scaling factor for brightening.
            beta (float): Bias added to the brightening.
        """
        self.alpha = alpha
        self.beta = beta

    def compute_flow_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute the magnitude of the optical flow.

        Args:
            flow (np.ndarray): Optical flow array of shape (H, W, 2).

        Returns:
            np.ndarray: Flow magnitude of shape (H, W).
        """
        return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    def brighten_regions(self, frame: np.ndarray, flow_magnitude: np.ndarray) -> np.ndarray:
        """
        Brighten regions in the frame based on flow magnitude.

        Args:
            frame (np.ndarray): Original RGB frame.
            flow_magnitude (np.ndarray): Optical flow magnitude.

        Returns:
            np.ndarray: Brightened RGB frame.
        """
        # Normalize flow magnitude to range [0, 1]
        norm_mag = cv2.normalize(flow_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Create a brightening mask
        bright_mask = np.clip(norm_mag * self.alpha + self.beta, 0, 1)

        # Convert mask to 3 channels
        bright_mask_3ch = np.repeat(bright_mask[:, :, np.newaxis], 3, axis=2)

        # Apply the brightening effect
        brightened_frame = cv2.convertScaleAbs(frame * (1 + bright_mask_3ch))

        return brightened_frame

    def process(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        flow_magnitude = self.compute_flow_magnitude(flow)
        return self.brighten_regions(frame, flow_magnitude)
