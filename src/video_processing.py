# src/video_processing.py

import os
import cv2
import numpy as np
from .optical_flow import OpticalFlowEstimator
from .brightening import Brightener
from .depth_refinement import DepthRefiner
from .utils import extract_frames, load_depth_maps, save_video
import time
import argparse
from typing import List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Depth Guided Flow Brightening Visual Effect")
    parser.add_argument('--input', type=str, required=True, help='Path to input video file.')
    parser.add_argument('--depth_dir', type=str, required=True, help='Path to depth maps directory.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output video.')
    parser.add_argument('--optical_flow_method', type=str, default='RAFT', choices=['RAFT', 'Farneback'], help='Optical flow estimation method.')
    parser.add_argument('--raft_model', type=str, default='RAFT/models/raft-things.pth', help='Path to RAFT model weights.')
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
    raft_model_path = args.raft_model
    alpha = args.alpha
    beta = args.beta

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    # Initialize components
    optical_flow_estimator = OpticalFlowEstimator(method=optical_flow_method, model_path=raft_model_path)
    brightener = Brightener(alpha=alpha, beta=beta)
    depth_refiner = DepthRefiner(near=0.0, far=20.0)

    with console.status("[bold green]Initializing...", spinner="dots") as status:
        # Extract frames from the video
        status.update("[bold blue]🎥 Extracting frames from video...")
        frames = extract_frames(input_video)
        num_frames = len(frames)
        console.print(f"[green]✅ Total frames extracted: {num_frames}")

        # Load depth maps
        status.update("[bold blue]🗺️ Loading depth maps...")
        depth_map_paths = sorted([os.path.join(depth_dir, fname) for fname in os.listdir(depth_dir) if fname.endswith('.png')])
        if len(depth_map_paths) != num_frames:
            raise ValueError("Number of depth maps does not match number of video frames.")
        depth_maps = [depth_refiner.load_depth_map(path) for path in depth_map_paths]
        console.print("[green]✅ Depth maps loaded successfully.")

    # Process each frame
    console.print("\n[bold cyan]🚀 Processing frames...")
    processed_frames: List[np.ndarray] = []
    optical_flow_times = []
    brightening_times = []
    refinement_times = []
    total_start = time.time()

    progress_table = Table.grid(expand=True)
    progress_table.add_column(justify="right", style="cyan", no_wrap=True)
    progress_table.add_column(style="magenta")

    job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    task_id = job_progress.add_task("[cyan]Frames", total=num_frames-1)

    frame_info = Table.grid(expand=True)
    frame_info.add_column(justify="right", style="green")
    frame_info.add_column(style="yellow")

    live = Live(Panel(progress_table, title="[b]Video Processing", border_style="blue", padding=(1, 1)), refresh_per_second=10)

    with live:
        for i in range(num_frames - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            depth_map = depth_maps[i + 1]  # Assuming depth map corresponds to frame2

            # Optical Flow Estimation
            start_time = time.time()
            flow = optical_flow_estimator.estimate_flow(frame1, frame2)
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
            refined_mask = depth_refiner.refine_brightening(flow_magnitude, depth_map)
            bright_mask_3ch = np.repeat(refined_mask[:, :, np.newaxis], 3, axis=2)
            refined_frame = cv2.convertScaleAbs(frame2 * (1 + bright_mask_3ch))
            refinement_time = time.time() - start_time
            refinement_times.append(refinement_time)

            processed_frames.append(refined_frame)

            # Update progress
            job_progress.update(task_id, advance=1)
            
            frame_info = Table.grid(expand=True)
            frame_info.add_column(justify="right", style="green")
            frame_info.add_column(style="yellow")
            frame_info.add_row("Frame", f"{i+1}/{num_frames-1}")
            frame_info.add_row("Optical Flow Time", f"{optical_flow_time:.4f}s")
            frame_info.add_row("Brightening Time", f"{brightening_time:.4f}s")
            frame_info.add_row("Refinement Time", f"{refinement_time:.4f}s")

            progress_table = Table.grid(expand=True)
            progress_table.add_row(job_progress)
            progress_table.add_row(frame_info)

            live.update(Panel(progress_table, title="[b]Video Processing", border_style="blue", padding=(1, 1)))

    total_end = time.time()
    total_time = total_end - total_start

    # Save the processed frames as video
    console.print("\n[bold green]💾 Saving output video...")
    try:
        save_video(processed_frames, output_video, fps=60)
        console.print(f"[bold green]✅ Output video saved at {output_video}")
    except Exception as e:
        console.print(f"[bold red]❌ Error saving video: {str(e)}")
        console.print("[yellow]Attempting to save frames as individual images...")
        
        output_dir = os.path.splitext(output_video)[0] + "_frames"
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(processed_frames):
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.png"), frame)
        console.print(f"[green]✅ Frames saved as images in {output_dir}")

    # Print Performance Metrics
    avg_optical_flow_time = np.mean(optical_flow_times)
    avg_brightening_time = np.mean(brightening_times)
    avg_refinement_time = np.mean(refinement_times)

    performance_table = Table(title="Performance Metrics (per frame)")
    performance_table.add_column("Metric", style="cyan", no_wrap=True)
    performance_table.add_column("Time", style="magenta")
    performance_table.add_row("Optical Flow Estimation", f"{avg_optical_flow_time:.4f} seconds")
    performance_table.add_row("Brightening", f"{avg_brightening_time:.4f} seconds")
    performance_table.add_row("Depth Refinement", f"{avg_refinement_time:.4f} seconds")
    performance_table.add_row("Total Processing", f"{total_time:.2f} seconds for {num_frames-1} frames")

    console.print(performance_table)

if __name__ == "__main__":
    main()
