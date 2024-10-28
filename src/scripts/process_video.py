# src/scripts/process_video.py

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob
import sys
import argparse
from motion_analysis import MotionAnalyzer

def process_video(input_video_path, depth_folder_path, output_path, flow_threshold_norm=0.25, threshold_dark_norm=0.9, contrast_sensitivity_norm=1.0, flow_brightness=1.0, target_width=480, batch_size=32, preset='balanced'):
    """
    Process video to highlight moving foreground based on optical flow and depth.
    
    Args:
        input_video_path (str): Path to the input video file.
        depth_folder_path (str): Path to the folder containing depth maps.
        output_path (str): Path to save the output visualization video.
        flow_threshold_norm (float): Normalized flow threshold [0.0, 1.0].
        threshold_dark_norm (float): Normalized depth threshold [0.0, 1.0].
        contrast_sensitivity_norm (float): Normalized contrast sensitivity [0.0, 1.0].
        flow_brightness (float): Controls brightness of optical flow visualization (0.5 to 3.0).
        target_width (int): Target width for resizing frames.
        batch_size (int): Batch size for processing (if applicable).
        preset (str): One of 'speed', 'balanced', 'quality', or 'max_quality'.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_height = int(target_width * original_height / original_width)
    target_size = (target_width, target_height)

    # Get depth maps sorted
    depth_files = sorted(glob.glob(str(Path(depth_folder_path) / "*.png")))
    if not depth_files:
        raise ValueError(f"No depth maps found in {depth_folder_path}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Four panels: Original, Depth Mask, Raw Optical Flow, Highlighted Frame
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (target_width * 4, target_height)  # 4 panels side by side
    )

    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_path}")

    # Map normalized settings to actual values for printing
    actual_flow_threshold = 1.0 + flow_threshold_norm * (4.0 - 1.0)  # [1.0,4.0]
    actual_contrast_sensitivity = 0.8 + contrast_sensitivity_norm * (3.0 - 0.8)  # [0.8,3.0]

    # Print the settings with normalized and actual values
    print("Video Processing Settings:")
    print(f"  Motion Detection Sensitivity (flow_threshold): {flow_threshold_norm:.2f} (Normalized) -> {actual_flow_threshold:.2f} (Actual)")
    print(f"  Depth Threshold (threshold_dark): {threshold_dark_norm:.2f} (Normalized) -> {threshold_dark_norm:.2f} (Actual)")
    print(f"  Contrast Sensitivity (contrast_sensitivity): {contrast_sensitivity_norm:.2f} (Normalized) -> {actual_contrast_sensitivity:.2f} (Actual)")
    print(f"  Flow Brightness: {flow_brightness} (Range: 0.5 to 3.0)")
    print(f"  Target Width: {target_width} pixels")
    print(f"  Batch Size for Floor Exclusion: {batch_size} frames")
    print(f"  Preset: '{preset}'")

    analyzer = MotionAnalyzer(
        threshold_flow=flow_threshold_norm,         
        threshold_dark=threshold_dark_norm,         
        contrast_sensitivity=contrast_sensitivity_norm,
        flow_brightness=flow_brightness,
        preset=preset
    )

    prev_gray = None

    try:
        # Get total frame count for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing Frames")

        frame_idx = 0
        depth_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for processing
            frame_small = cv2.resize(frame, target_size)

            # Read corresponding depth map
            if depth_idx < len(depth_files):
                depth_map = cv2.imread(depth_files[depth_idx], cv2.IMREAD_UNCHANGED)
                if depth_map is not None:
                    depth_map = cv2.resize(depth_map, target_size)
                else:
                    depth_map = np.zeros((target_height, target_width), dtype=np.uint8)
            else:
                depth_map = np.zeros((target_height, target_width), dtype=np.uint8)

            depth_idx += 1

            # Convert to grayscale
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = analyzer.estimate_global_motion(prev_gray, gray)
                flow_mask = analyzer.generate_flow_mask(flow, frame_small)
                depth_mask = analyzer.extract_foreground_mask(depth_map)
                combined_mask = cv2.bitwise_and(flow_mask, depth_mask)

                # Create optical flow visualization
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
                hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
                hsv[..., 1] = 255
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                # Create 3-channel mask for visualization
                mask_3ch = cv2.merge([combined_mask, combined_mask, combined_mask])
                
                # Apply mask to flow visualization
                flow_rgb_masked = flow_rgb * mask_3ch
                
                # Enhance brightness of the masked flow visualization
                flow_rgb_masked = cv2.convertScaleAbs(flow_rgb_masked, alpha=analyzer.flow_brightness, beta=30)

                contrast_map = analyzer.analyze_color_gradients(frame_small, combined_mask)
                highlight_mask = analyzer.create_highlight_mask(contrast_map)

                # Create white overlay based on highlight mask
                white_overlay = np.ones_like(frame_small, dtype=np.uint8) * 255
                highlight_mask_3d = cv2.merge([highlight_mask, highlight_mask, highlight_mask])
                highlight_mask_3d = np.clip(highlight_mask_3d, 0, 1)
                highlighted = (frame_small.astype(np.float32) * (1 - highlight_mask_3d) +
                               white_overlay.astype(np.float32) * highlight_mask_3d)
                highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)

                # Convert depth mask to displayable format
                depth_mask_display = (depth_mask * 255).astype(np.uint8)
                depth_mask_display = cv2.cvtColor(depth_mask_display, cv2.COLOR_GRAY2BGR)

                # Stack the four panels horizontally
                combined = np.hstack((frame_small, depth_mask_display, flow_rgb_masked, highlighted))

            else:
                # For the first frame
                blank = np.zeros_like(frame_small)
                combined = np.hstack((frame_small, blank, blank, blank))

            # Write the combined frame
            out.write(combined)
            prev_gray = gray.copy()
            frame_idx += 1
            pbar.update(1)

        pbar.close()

    finally:
        cap.release()
        out.release()

def main():
    """
    Main function to parse arguments and initiate video processing.
    """
    parser = argparse.ArgumentParser(description="Process a video to generate motion highlights based on optical flow and depth maps.")
    
    # Input/Output Configuration
    parser.add_argument('--input_video', type=str, default='data/input_video.mp4', help='Path to the input video file.')
    parser.add_argument('--depth_folder', type=str, default='data/source_depth', help='Path to the folder containing depth maps.')
    parser.add_argument('--output_video', type=str, default='outputs/output_visualization.mp4', help='Path to save the output visualization video.')
    
    # Motion Detection Fine-Tuning
    parser.add_argument('--flow_threshold_norm', type=float, default=0.25, help='Normalized flow threshold [0.0, 1.0].')
    
    # Depth-Based Segmentation
    parser.add_argument('--threshold_dark_norm', type=float, default=0.9, help='Normalized depth threshold [0.0, 1.0].')
    
    # Visual Effect Control
    parser.add_argument('--contrast_sensitivity_norm', type=float, default=1.0, help='Normalized contrast sensitivity [0.0, 1.0].')
    
    # Optical Flow Visualization
    parser.add_argument('--flow_brightness', type=float, default=1.0, help='Controls brightness of optical flow visualization (0.5 to 3.0).')
    
    # Performance Optimization
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing.')
    
    # Preset Selection
    parser.add_argument('--preset', type=str, default='balanced', choices=['speed', 'balanced', 'quality', 'max_quality'], help='Preset configuration.')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_video).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        process_video(
            input_video_path=args.input_video,
            depth_folder_path=args.depth_folder,
            output_path=args.output_video,
            flow_threshold_norm=args.flow_threshold_norm,
            threshold_dark_norm=args.threshold_dark_norm,
            contrast_sensitivity_norm=args.contrast_sensitivity_norm,
            flow_brightness=args.flow_brightness,
            target_width=MotionAnalyzer.PRESETS[args.preset]['target_width'],
            batch_size=MotionAnalyzer.PRESETS[args.preset]['batch_size'],
            preset=args.preset
        )
        print(f"Video processing completed successfully using '{args.preset}' preset!")
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
