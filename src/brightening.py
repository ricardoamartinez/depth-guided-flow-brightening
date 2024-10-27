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
        # No need to normalize, as flow_magnitude is already in a good range
        print("Flow magnitude:", flow_magnitude)
        bright_mask = np.clip(flow_magnitude * self.alpha + self.beta, 0, 1)
        print("Bright mask:", bright_mask)
        bright_mask_3ch = np.repeat(bright_mask[:, :, np.newaxis], 3, axis=2)
        print("3-channel bright mask:", bright_mask_3ch)
        brightened_frame = np.clip(frame + (255 * bright_mask_3ch), 0, 255).astype(np.uint8)
        print("Brightened frame:", brightened_frame)
        return brightened_frame

    def process(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        flow_magnitude = self.compute_flow_magnitude(flow)
        return self.brighten_regions(frame, flow_magnitude)
