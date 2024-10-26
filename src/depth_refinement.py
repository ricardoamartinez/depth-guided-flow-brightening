# src/depth_refinement.py

import numpy as np
import cv2

class DepthRefiner:
    def __init__(self, near: float = 0.0, far: float = 20.0):
        """
        Initialize the DepthRefiner.

        Args:
            near (float): Minimum depth value in meters.
            far (float): Maximum depth value in meters.
        """
        self.near = near
        self.far = far

    def load_depth_map(self, depth_path: str) -> np.ndarray:
        """
        Load and preprocess the depth map.

        Args:
            depth_path (str): Path to the depth map PNG file.

        Returns:
            np.ndarray: Depth map in meters with shape (H, W).
        """
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth is None:
            raise FileNotFoundError(f"Depth map not found at {depth_path}")
        
        # Assuming depth maps are stored as linear uint16 PNGs with range 0-65535
        depth = depth / 65535.0  # Normalize to [0, 1]
        depth = depth * self.far  # Scale to [0, far] meters
        return depth

    def refine_brightening(self, bright_mask: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Refine the brightening mask based on depth.

        Args:
            bright_mask (np.ndarray): Initial brightening mask [0, 1].
            depth_map (np.ndarray): Depth map in meters.

        Returns:
            np.ndarray: Refined brightening mask [0, 1].
        """
        # Invert depth: closer objects have higher influence
        depth_factor = 1 - ((depth_map - self.near) / (self.far - self.near))
        depth_factor = np.clip(depth_factor, 0, 1)

        # Multiply bright mask with depth factor
        refined_mask = bright_mask * depth_factor

        # Normalize again to [0, 1]
        refined_mask = cv2.normalize(refined_mask, None, 0, 1, cv2.NORM_MINMAX)

        return refined_mask
