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

    def normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        return (depth_map - self.near) / (self.far - self.near)

    def refine_brightening(self, bright_mask: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Refine the brightening mask based on depth.

        Args:
            bright_mask (np.ndarray): Initial brightening mask [0, 1].
            depth_map (np.ndarray): Depth map in meters.

        Returns:
            np.ndarray: Refined brightening mask [0, 1].
        """
        # Resize depth_map to match bright_mask dimensions
        if depth_map.shape != bright_mask.shape:
            depth_map = cv2.resize(depth_map, (bright_mask.shape[1], bright_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        normalized_depth = self.normalize_depth(depth_map)
        depth_factor = 1 - normalized_depth
        refined_mask = bright_mask * depth_factor
        return refined_mask

    def process(self, bright_mask: np.ndarray, depth_path: str) -> np.ndarray:
        depth_map = self.load_depth_map(depth_path)
        return self.refine_brightening(bright_mask, depth_map)
