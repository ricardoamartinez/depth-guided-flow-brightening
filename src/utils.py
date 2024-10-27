# src/utils.py

import os
import cv2
import numpy as np
from typing import List

def extract_frames(video_path: str) -> List[np.ndarray]:
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the input video.

    Returns:
        List[np.ndarray]: List of frames as BGR images.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames: List[np.ndarray], output_path: str, fps: int = 60):
    """
    Save a list of frames as a video file.

    Args:
        frames (List[np.ndarray]): List of BGR frames.
        output_path (str): Path to save the output video.
        fps (int): Frames per second.
    """
    if not frames:
        raise ValueError("No frames to save.")

    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

    # Verify the video was saved correctly
    cap = cv2.VideoCapture(output_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open the saved video at {output_path}")
    cap.release()

def load_depth_maps(depth_dir: str) -> List[np.ndarray]:
    """
    Load all depth maps from a directory.

    Args:
        depth_dir (str): Path to the directory containing depth map PNGs.

    Returns:
        List[np.ndarray]: List of depth maps.
    """
    depth_map_paths = sorted([os.path.join(depth_dir, fname) for fname in os.listdir(depth_dir) if fname.endswith('.png')])
    depth_maps = []
    for path in depth_map_paths:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_maps.append(depth)
    return depth_maps
