# src/video_processing.py

import os
import cv2
import numpy as np
from optical_flow import OpticalFlowEstimator
from brightening import Brightener
from depth_refinement import DepthRefiner
from utils import extract_frames, load_depth_maps, save_video
import time
import argparse
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(description="Depth Guided Flow Brightening Visual Effect")
    parser.add_argument('--input', type=str, required=True, help='Path to input video file.')
    parser.add_argument('--depth_dir', type=str, required=True, help='Path to depth maps directory.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output video.')
    parser.add_argument('--optical_flow_method', type=str, default='Farneback', choices=['Farneback'], help='Optical flow estimation method.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Brightening scaling factor.')
    parser.add_argument('--beta', type=float, default=0.0, help='Brightening bias.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    input_video = args.input
    depth_dir = args.depth_dir
    output_video = args.output
    optical_flow_method = args.optical_flow_method
    alpha = args.alpha
    beta = args.beta

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    # Initialize components
    optical_flow_estimator = OpticalFlowEstimator(method=optical_flow_method)
    brightener = Brightener(alpha=alpha, beta=beta)
    depth_refiner = DepthRefiner(near=0.0, far=20.0)

    # Extract frames from the video
    print("Extracting frames from video...")
    frames = extract_frames(input_video)
    num_frames = len(frames)
    print(f"Total frames extracted: {num_frames}")

    # Load depth maps
    print("Loading depth maps...")
    depth_map_paths = sorted([os.path.join(depth_dir, fname) for fname in os.listdir(depth_dir) if fname.endswith('.png')])
    if len(depth_map_paths) != num_frames:
        raise ValueError("Number of depth maps does not match number of video frames.")
    depth_maps = [depth_refiner.load_depth_map(path) for path in depth_map_paths]
    print("Depth maps loaded successfully.")

    # Process each frame
    print("Processing frames...")
    processed_frames: List[np.ndarray] = []
    optical_flow_times = []
    brightening_times = []
    refinement_times = []
    total_start = time.time()

    for i in range(num_frames - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        depth_map = depth_maps[i + 1]  # Assuming depth map corresponds to frame2

        # Convert frames to grayscale for optical flow
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Optical Flow Estimation
        start_time = time.time()
        flow = optical_flow_estimator.estimate_flow(gray1, gray2)
        optical_flow_time = time.time() - start_time
        optical_flow_times.append(optical_flow_time)

        # Compute Flow Magnitude and Brighten
        start_time = time.time()
        flow_magnitude = brightener.compute_flow_magnitude(flow)
        bright_frame = brightener.brighten_regions(frame2, flow_magnitude)
        brightening_time = time.time() - start_time
        brightening_times.append(brightening_time)

        # Depth-Guided Refinement
        start_time = time.time()
        refined_mask = depth_refiner.refine_brightening(brightener.compute_flow_magnitude(flow), depth_map)
        # Apply refined mask
        bright_mask_3ch = np.repeat(refined_mask[:, :, np.newaxis], 3, axis=2)
        refined_frame = cv2.convertScaleAbs(frame2 * (1 + bright_mask_3ch))
        refinement_time = time.time() - start_time
        refinement_times.append(refinement_time)

        processed_frames.append(refined_frame)

        if (i + 1) % 50 == 0 or (i + 1) == num_frames -1:
            print(f"Processed frame {i + 1}/{num_frames - 1}")

    total_end = time.time()
    total_time = total_end - total_start

    # Save the processed frames as video
    print("Saving output video...")
    save_video(processed_frames, output_video, fps=60)
    print(f"Output video saved at {output_video}")

    # Print Performance Metrics
    avg_optical_flow_time = np.mean(optical_flow_times)
    avg_brightening_time = np.mean(brightening_times)
    avg_refinement_time = np.mean(refinement_times)
    print("\nPerformance Metrics (per frame):")
    print(f"Optical Flow Estimation Time: {avg_optical_flow_time:.4f} seconds")
    print(f"Brightening Time: {avg_brightening_time:.4f} seconds")
    print(f"Depth Refinement Time: {avg_refinement_time:.4f} seconds")
    print(f"Total Processing Time: {total_time:.2f} seconds for {num_frames -1} frames")

if __name__ == "__main__":
    main()
