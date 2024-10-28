# src/motion_analysis/motion_analyzer.py

import cv2
import numpy as np
from collections import deque
from .person_segmentation import PersonSegmenter

class MotionAnalyzer:
    """
    MotionAnalyzer class for analyzing motion in video frames, extracting foreground based on depth,
    compensating for camera motion, and generating dynamic highlights.
    """

    # Quality presets for easy configuration
    PRESETS = {
        'speed': {
            'flow_params': {
                'pyr_scale': 0.9,    # Fastest pyramid scaling
                'levels': 1,         # Minimum levels
                'winsize': 3,        # Smallest window
                'iterations': 1,     # Minimum iterations
                'poly_n': 3,         # Smallest neighborhood
                'poly_sigma': 1.1    # Minimum smoothing
            },
            'feature_params': {
                'maxCorners': 30,
                'qualityLevel': 0.05,
                'minDistance': 50,
                'blockSize': 3
            },
            'target_width': 240,     # Low resolution
            'batch_size': 4,         # Small batch
            'temporal_buffer_size': 3 # Minimum temporal smoothing
        },
        
        'balanced': {
            'flow_params': {
                'pyr_scale': 0.5,
                'levels': 2,
                'winsize': 7,
                'iterations': 3,
                'poly_n': 5,
                'poly_sigma': 1.2
            },
            'feature_params': {
                'maxCorners': 100,
                'qualityLevel': 0.01,
                'minDistance': 40,
                'blockSize': 3
            },
            'target_width': 480,
            'batch_size': 32,
            'temporal_buffer_size': 5
        },
        
        'quality': {
            'flow_params': {
                'pyr_scale': 0.3,
                'levels': 3,
                'winsize': 15,
                'iterations': 5,
                'poly_n': 5,
                'poly_sigma': 1.5
            },
            'feature_params': {
                'maxCorners': 200,
                'qualityLevel': 0.01,
                'minDistance': 30,
                'blockSize': 3
            },
            'target_width': 720,
            'batch_size': 32,
            'temporal_buffer_size': 8
        },
        
        'max_quality': {
            'flow_params': {
                'pyr_scale': 0.1,
                'levels': 5,
                'winsize': 21,
                'iterations': 8,
                'poly_n': 7,
                'poly_sigma': 2.0
            },
            'feature_params': {
                'maxCorners': 400,
                'qualityLevel': 0.005,
                'minDistance': 20,
                'blockSize': 3
            },
            'target_width': 960, 
            'batch_size': 32,
            'temporal_buffer_size': 10
        }
    }

    def __init__(self, threshold_flow=0.5, threshold_dark=0.0, contrast_sensitivity=0.0, flow_brightness=1.0, preset='balanced'):
        """
        Initialize the MotionAnalyzer with a preset configuration.
        
        Args:
            threshold_flow (float): Normalized flow threshold [0.0, 1.0]
            threshold_dark (float): Normalized depth threshold [0.0, 1.0]
            contrast_sensitivity (float): Normalized contrast sensitivity [0.0, 1.0]
            flow_brightness (float): Controls brightness of optical flow visualization (0.5 to 3.0)
            preset (str): One of 'speed', 'balanced', 'quality', or 'max_quality'
        """
        # Load preset configuration
        config = self.PRESETS[preset]
        
        # Initialize parameters from preset
        self.flow_params = config['flow_params']
        self.feature_params = config['feature_params']
        self.temporal_buffer = deque(maxlen=config['temporal_buffer_size'])
        
        # Initialize thresholds and sensitivities
        self.threshold_flow = 1.0 + threshold_flow * (4.0 - 1.0)  # [1.0, 4.0]
        self.threshold_dark = threshold_dark  # [0,1]
        self.contrast_sensitivity = 0.8 + contrast_sensitivity * (3.0 - 0.8)  # [0.8, 3.0]
        self.flow_brightness = max(0.5, min(3.0, flow_brightness))  # Clamp between 0.5 and 3.0
        
        # Initialize person segmenter
        self.person_segmenter = PersonSegmenter()

    def compute_optical_flow(self, prev_gray, curr_gray):
        """
        High-precision motion detection using Farneback's dense optical flow.
        
        Args:
            prev_gray (np.ndarray): Previous grayscale frame
            curr_gray (np.ndarray): Current grayscale frame
        
        Returns:
            np.ndarray: Optical flow
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            **self.flow_params,
            flags=0
        )
        return flow

    def estimate_global_motion(self, prev_gray, curr_gray):
        """
        Advanced camera motion compensation system.
        
        Uses a two-stage approach:
        1. Feature Detection & Tracking
           - Identifies stable points in the scene
           - Tracks their movement between frames
        
        2. Homography Estimation
           - Calculates geometric transform between frames
           - Filters out outliers using RANSAC
           
        This helps distinguish between:
        - Intentional subject motion (preserved)
        - Unwanted camera shake (removed)
        
        Returns stabilized optical flow that focuses on true motion.
        
        Args:
            prev_gray (np.ndarray): Previous grayscale frame
            curr_gray (np.ndarray): Current grayscale frame
        
        Returns:
            np.ndarray: Stabilized optical flow
        """
        # Estimate global motion using Lucas-Kanade method on feature points
        feature_params = dict(
            maxCorners=50,     # Reduced from 200 to 50 for performance
            qualityLevel=0.01, # Minimum quality of corners
            minDistance=40,    # Increased from 30 to 40
            blockSize=3
        )
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if prev_pts is None:
            return self.compute_optical_flow(prev_gray, curr_gray)

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 4:
            return self.compute_optical_flow(prev_gray, curr_gray)

        # Find homography
        H, mask = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        if H is None:
            return self.compute_optical_flow(prev_gray, curr_gray)

        # Warp previous frame to current frame
        height, width = curr_gray.shape
        warped_prev_gray = cv2.warpPerspective(prev_gray, H, (width, height))

        # Compute optical flow on stabilized frames
        flow = cv2.calcOpticalFlowFarneback(
            warped_prev_gray,
            curr_gray,
            None,
            **self.flow_params,
            flags=0
        )

        return flow

    def generate_flow_mask(self, flow, frame):
        """
        Creates intelligent motion mask using optical flow magnitude and person segmentation.
        
        Args:
            flow (np.ndarray): Optical flow
            frame (np.ndarray): Current frame
        
        Returns:
            np.ndarray: Combined motion mask
        """
        # Calculate flow magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Get basic flow mask
        flow_mask = (mag > (self.threshold_flow * 0.3)).astype(np.float32)
        
        # Get person mask
        person_mask = self.person_segmenter.get_person_mask(frame)
        
        # Combine masks - only keep flow where there's a person
        combined_mask = flow_mask * person_mask
        
        # Clean up combined mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.GaussianBlur(combined_mask, (15, 15), 0)
        
        return combined_mask

    def extract_foreground_mask(self, depth_map):
        """
        Depth-based foreground extraction system.
        Modified to handle very close objects correctly.
        
        Args:
            depth_map (np.ndarray): Depth map of the current frame
        
        Returns:
            np.ndarray: Foreground mask based on depth
        """
        if len(depth_map.shape) == 3:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

        # Convert depth map to float32 for accurate normalization
        depth_map = depth_map.astype(np.float32)

        # Normalize depth map to [0, 1] and invert so closer objects are brighter
        depth_norm = cv2.normalize(depth_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        depth_norm = 1.0 - depth_norm  # Invert so closer is brighter

        # Create initial mask - everything ABOVE threshold is foreground
        foreground_mask = (depth_norm >= self.threshold_dark).astype(np.float32)

        # Clean up mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

        return foreground_mask

    def analyze_color_gradients(self, frame, mask):
        """
        Analyzes color gradients to enhance motion areas.
        Modified to ensure full coverage of motion areas, not just edges.
        
        Args:
            frame (np.ndarray): Current frame
            mask (np.ndarray): Combined motion mask
        
        Returns:
            np.ndarray: Contrast map
        """
        # First ensure we're working with the full motion area
        motion_area = mask.copy()
        
        # Convert to float32 for precise calculations
        masked_frame_float = frame.astype(np.float32) / 255.0
        
        # Instead of edge detection, we'll use the full motion mask as base
        contrast_map = motion_area.copy()
        
        # Optional: Add subtle edge enhancement while maintaining full coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        enhanced_edges = cv2.dilate(contrast_map, kernel) - cv2.erode(contrast_map, kernel)
        
        # Combine full coverage with subtle edge enhancement
        contrast_map = cv2.addWeighted(contrast_map, 0.7, enhanced_edges, 0.3, 0)
        
        # Ensure mask coverage
        contrast_map = contrast_map * mask
        
        return contrast_map

    def create_highlight_mask(self, contrast_map):
        """
        Creates a highlight mask based on the contrast map.
        Modified to maintain full coverage of motion areas.
        
        Args:
            contrast_map (np.ndarray): Contrast map
        
        Returns:
            np.ndarray: Highlight mask
        """
        highlight_mask = contrast_map.copy()

        # Temporal smoothing
        self.temporal_buffer.append(highlight_mask)
        if len(self.temporal_buffer) > 1:
            weights = np.exp(-0.5 * np.linspace(0, 2, len(self.temporal_buffer))**2)
            weights /= weights.sum()

            smooth_mask = np.zeros_like(highlight_mask)
            for mask, weight in zip(self.temporal_buffer, weights):
                smooth_mask += mask * weight

            # Fill internal areas using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, kernel)
            
            if hasattr(self, 'prev_smooth_mask'):
                smooth_mask = 0.7 * smooth_mask + 0.3 * self.prev_smooth_mask
            
            self.prev_smooth_mask = smooth_mask.copy()
            highlight_mask = smooth_mask

        # Ensure full coverage of motion areas
        highlight_mask = cv2.dilate(highlight_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        # Smooth edges while maintaining internal coverage
        highlight_mask = cv2.GaussianBlur(highlight_mask, (9, 9), 0)

        return np.clip(highlight_mask, 0, 1)
