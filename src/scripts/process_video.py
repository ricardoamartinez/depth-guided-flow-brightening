# src/scripts/process_video.py

"""
Motion Analysis and Highlight Generation System
Copyright (c) 2024 WISTLABS
Author: Ricardo Alexander Martinez <martricardo.a@gmail.com>

A very basic toy example of a video processing system that combines optical flow analysis,
depth mapping, and highlight generation to create dynamic
visual effects that emphasize motion and depth in video sequences.

Licensed under the terms of the MIT license.
"""

import os  # ✅ Added to set environment variables
import cv2
import numpy as np
from pathlib import Path
import glob
import sys
import argparse
from motion_analysis import MotionAnalyzer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn  # ✅ Added SpinnerColumn
)
import emoji
from rich.layout import Layout
from rich.live import Live
from rich.syntax import Syntax
from rich.traceback import install
from rich.markdown import Markdown
from rich.columns import Columns
import psutil
import time
import datetime  # ✅ Changed to import datetime module directly
from queue import Queue
from threading import Thread
import json

# ✅ Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize rich Console
console = Console()

# Install rich traceback handler
install(show_locals=True)

def create_header() -> Panel:
    """Create header with project info."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_row(f"[bold cyan]Motion Analysis and Highlight Generation System[/bold cyan]")
    grid.add_row("[yellow]WISTLABS © 2024[/yellow]")
    return Panel(grid, style="bright_blue")

def create_settings_panel(settings: dict) -> Panel:
    """Create a panel displaying processing settings."""
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in settings.items():
        table.add_row(key, str(value))
    
    return Panel(
        table,
        title="[bold cyan]Processing Settings[/bold cyan] 🎬",
        border_style="bright_blue",
        padding=(1, 2)
    )

def create_stats_panel(stats: dict) -> Panel:
    """Create a panel displaying processing statistics."""
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    for key, value in stats.items():
        table.add_row(key, str(value))
    
    return Panel(
        table,
        title="[bold cyan]Processing Statistics[/bold cyan] 📊",
        border_style="bright_blue",
        padding=(1, 2)
    )

def save_processing_stats(stats, output_path="outputs/processing_stats.json"):
    """Save processing statistics to JSON file"""
    stats_dict = {
        'summary': {
            'Total Frames': stats['frames_processed'],
            'Average FPS': stats['average_fps'],
            'Peak Memory (MB)': stats['peak_memory_mb'],
            'Total Processing Time': str(stats['total_time']),
            'Average CPU Load (%)': stats['avg_cpu_load'],
            'Average GPU Util (%)': stats['avg_gpu_util']
        },
        'frame_metrics': stats['frame_metrics'],
        'function_profile': stats['function_profile']
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)

def process_video(input_video_path, depth_folder_path, output_path, flow_threshold_norm=0.25, threshold_dark_norm=0.5, contrast_sensitivity_norm=1.0, flow_brightness=1.0, target_width=480, batch_size=32, preset='balanced', flow_method='farneback'):
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
        flow_method (str): One of 'farneback', 'fastflownet', or 'deepflow'.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        console.print(f"{emoji.emojize(':warning:')} [bold red]Error:[/bold red] Could not open video file: {input_video_path}")
        sys.exit(1)  # ✅ Exit if video cannot be opened

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_height = int(target_width * original_height / original_width)
    target_size = (target_width, target_height)

    # Get depth maps sorted
    depth_files = sorted(glob.glob(str(Path(depth_folder_path) / "*.png")))
    if not depth_files:
        console.print(f"{emoji.emojize(':warning:')} [bold yellow]Warning:[/bold yellow] No depth maps found in {depth_folder_path}")

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
        console.print(f"{emoji.emojize(':warning:')} [bold red]Error:[/bold red] Could not create output video: {output_path}")
        sys.exit(1)  # ✅ Exit if video writer cannot be created

    # Map normalized settings to actual values for printing
    actual_flow_threshold = 1.0 + flow_threshold_norm * (4.0 - 1.0)  # [1.0,4.0]
    actual_contrast_sensitivity = 0.8 + contrast_sensitivity_norm * (3.0 - 0.8)  # [0.8,3.0]

    # Print the settings with normalized and actual values
    console.print(Panel.fit(f"[bold underline cyan]Video Processing Settings[/bold underline cyan] 🎬", border_style="bright_blue"))
    settings_table = Table.grid(expand=True)
    settings_table.add_column(justify="left", style="magenta")
    settings_table.add_column(justify="right")
    settings_table.add_row("Motion Detection Sensitivity (flow_threshold):", f"[green]{flow_threshold_norm:.2f} (Normalized) -> {actual_flow_threshold:.2f} (Actual)[/green]")
    settings_table.add_row("Depth Threshold (threshold_dark):", f"[green]{threshold_dark_norm:.2f} (Normalized) -> {threshold_dark_norm:.2f} (Actual)[/green]")
    settings_table.add_row("Contrast Sensitivity (contrast_sensitivity):", f"[green]{contrast_sensitivity_norm:.2f} (Normalized) -> {actual_contrast_sensitivity:.2f} (Actual)[/green]")
    settings_table.add_row("Flow Brightness:", f"[green]{flow_brightness} (Range: 0.5 to 3.0)[/green]")
    settings_table.add_row("Target Width:", f"[green]{target_width} pixels[/green]")
    settings_table.add_row("Batch Size for Floor Exclusion:", f"[green]{batch_size} frames[/green]")
    settings_table.add_row("Preset:", f"[green]{preset}[/green]")
    console.print(settings_table)

    analyzer = MotionAnalyzer(
        threshold_flow=flow_threshold_norm,         
        threshold_dark=threshold_dark_norm,         
        contrast_sensitivity=contrast_sensitivity_norm,
        flow_brightness=flow_brightness,
        preset=preset,
        flow_method=flow_method
    )

    prev_gray = None

    # Create frame queue and start frame reader thread
    frame_queue = Queue(maxsize=32)
    stop_flag = False

    def frame_reader():
        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((ret, frame))
        frame_queue.put((False, None))  # Signal end of video

    reader_thread = Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    try:
        # Get total frame count for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = time.time()  # Add start time for statistics
        frame_idx = 0  # Initialize frame_idx here
        depth_idx = 0  # Initialize depth_idx here
        
        # Create layout with adjusted sizes
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=10)  # ✅ Increased footer size
        )
        layout["main"].split_row(
            Layout(name="settings"),
            Layout(name="progress")
        )
        
        # Initialize statistics with better formatting
        stats = {
            "Frames Processed": "0",
            "Average FPS": "0.00",
            "Current Memory Usage": "0 MB",
            "Elapsed Time": "0:00:00"
        }
        
        # Create initial stats panel
        stats_panel = create_stats_panel(stats)
        layout["footer"].update(stats_panel)

        # Create settings dict
        settings = {
            "Motion Detection Sensitivity": f"{flow_threshold_norm:.2f} (Normalized)",
            "Depth Threshold": f"{threshold_dark_norm:.2f}",
            "Contrast Sensitivity": f"{contrast_sensitivity_norm:.2f}",
            "Flow Brightness": f"{flow_brightness}",
            "Target Width": f"{target_width}px",
            "Preset": preset.upper()
        }
        
        with Live(layout, refresh_per_second=4, screen=True) as live:
            # Update header
            layout["header"].update(create_header())
            
            # Update settings panel
            layout["settings"].update(create_settings_panel(settings))
            
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                expand=True,
                transient=False,
                refresh_per_second=4
            )
            
            # Add tasks
            process_task = progress.add_task(
                f"[cyan]Processing Video[/cyan] 🎥",
                total=total_frames,
                start=True
            )
            flow_task = progress.add_task(
                "[magenta]Optical Flow Analysis[/magenta] 🌊", 
                total=total_frames,
                start=True
            )
            depth_task = progress.add_task(
                "[yellow]Depth Processing[/yellow] 📏", 
                total=total_frames,
                start=True
            )
            highlight_task = progress.add_task(
                "[green]Generating Highlights[/green] ✨", 
                total=total_frames,
                start=True
            )

            # Create progress panel once
            progress_panel = Panel(progress, title="Progress", border_style="blue")
            layout["progress"].update(progress_panel)
            
            # Batch updates for smoother display
            update_batch_size = 5  # Update every 5 frames
            frames_since_update = 0
            last_update = time.time()
            
            # Initialize metrics tracking
            cpu_loads = []
            gpu_utils = []
            memory_usage = []
            frame_metrics = []
            process = psutil.Process()

            # Main processing loop
            while True:
                ret, frame = frame_queue.get()
                if not ret:
                    break

                # Track system metrics
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage.append(current_memory)
                cpu_loads.append(psutil.cpu_percent())
                
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except:
                    gpu_util = 0
                gpu_utils.append(gpu_util)

                # Collect frame-level metrics
                frame_metrics.append([
                    time.time() - start_time,  # processing time
                    current_memory,            # memory usage
                    frame_idx / (time.time() - start_time),  # current fps
                    psutil.cpu_percent(),      # cpu load
                    gpu_util                   # gpu utilization
                ])

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
                    # Choose flow computation method
                    if flow_method == 'fastflownet':
                        # FastFlowNet without camera motion compensation
                        flow = analyzer.compute_optical_flow(prev_gray, gray)
                        if not hasattr(analyzer, '_method_logged'):
                            console.print("[cyan]Using FastFlowNet for optical flow computation[/cyan]")
                            analyzer._method_logged = True
                    else:
                        # Farneback with camera motion compensation
                        flow = analyzer.estimate_global_motion(prev_gray, gray)
                        if not hasattr(analyzer, '_method_logged'):
                            console.print("[cyan]Using Farneback with camera motion compensation[/cyan]")
                            analyzer._method_logged = True

                    flow_mask = analyzer.generate_flow_mask(flow, frame_small)
                    depth_mask = analyzer.extract_foreground_mask(depth_map)

                    # Convert masks to float32 for operations
                    flow_mask = flow_mask.astype(np.float32)
                    depth_mask = depth_mask.astype(np.float32)
                    combined_mask = cv2.multiply(flow_mask, depth_mask)
                    
                    # Create optical flow visualization
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
                    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
                    hsv[..., 1] = 255
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    
                    # Create 3-channel mask for visualization (as float32)
                    mask_3ch = cv2.merge([combined_mask, combined_mask, combined_mask])
                    
                    # Convert flow_rgb to float32 for multiplication
                    flow_rgb = flow_rgb.astype(np.float32) / 255.0
                    
                    # Apply mask to flow visualization
                    flow_rgb_masked = cv2.multiply(flow_rgb, mask_3ch)
                    
                    # Convert back to uint8 for display
                    flow_rgb_masked = (flow_rgb_masked * 255).astype(np.uint8)
                    
                    # Enhance brightness of the masked flow visualization
                    flow_rgb_masked = cv2.convertScaleAbs(flow_rgb_masked, alpha=analyzer.flow_brightness, beta=30)

                    contrast_map = analyzer.analyze_color_gradients(frame_small, combined_mask)
                    highlight_mask = analyzer.create_highlight_mask(contrast_map)

                    # Create white overlay based on highlight mask
                    white_overlay = np.ones_like(frame_small, dtype=np.float32) * 255
                    highlight_mask_3d = cv2.merge([highlight_mask, highlight_mask, highlight_mask])
                    
                    # Calculate alpha values for blending
                    alpha = 1 - highlight_mask_3d.astype(np.float32)
                    beta = highlight_mask_3d.astype(np.float32)
                    
                    # Ensure frame is float32
                    frame_small_float = frame_small.astype(np.float32)
                    
                    # Perform weighted addition manually
                    highlighted = (frame_small_float * alpha + white_overlay * beta)
                    
                    # Convert back to uint8
                    highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)

                    # Convert depth mask to displayable format
                    depth_mask_display = (depth_mask * 255).astype(np.uint8)
                    depth_mask_display = cv2.cvtColor(depth_mask_display, cv2.COLOR_GRAY2BGR)

                    # Stack the four panels horizontally
                    try:
                        combined = np.hstack((frame_small, depth_mask_display, flow_rgb_masked, highlighted))
                    except cv2.error as e:
                        console.print(f"{emoji.emojize(':cross_mark:')} [bold red]OpenCV Error:[/bold red] {e}")
                        continue

                else:
                    # For the first frame
                    blank = np.zeros_like(frame_small)
                    combined = np.hstack((frame_small, blank, blank, blank))

                # Write the combined frame
                out.write(combined)
                prev_gray = gray.copy()
                frame_idx += 1
                frames_since_update += 1  # ✅ Add counter increment
                current_time = time.time()
                
                # Update progress and collect stats
                if frames_since_update >= update_batch_size:
                    stats = {
                        'frames_processed': frame_idx,
                        'average_fps': frame_idx / (time.time() - start_time),
                        'peak_memory_mb': max(memory_usage) if memory_usage else 0,
                        'total_time': datetime.timedelta(seconds=int(time.time() - start_time)),
                        'avg_cpu_load': np.mean(cpu_loads) if cpu_loads else 0,
                        'avg_gpu_util': np.mean(gpu_utils) if gpu_utils else 0,
                        'frame_metrics': frame_metrics,
                        'function_profile': {
                            'functions': ['process_frame', 'optical_flow', 'depth_processing', 'highlight_gen'],
                            'callers': ['', 'process_frame', 'process_frame', 'process_frame'],
                            'time_spent': [40, 20, 20, 20]  # Example values
                        }
                    }
                    
                    # Save stats
                    save_processing_stats(stats)
                    
                    # Update display
                    layout["footer"].update(create_stats_panel({
                        "Frames Processed": f"{frame_idx:,}",
                        "Average FPS": f"{stats['average_fps']:.2f}",
                        "Current Memory Usage": f"{current_memory:.1f} MB",
                        "Peak Memory Usage": f"{stats['peak_memory_mb']:.1f} MB",
                        "CPU Load": f"{stats['avg_cpu_load']:.1f}%",
                        "GPU Utilization": f"{stats['avg_gpu_util']:.1f}%",
                        "Elapsed Time": str(stats['total_time'])
                    }))
                    
                    frames_since_update = 0
                    live.refresh()

    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/bold red] {str(e)}\n\n[dim]See traceback below:[/dim]",
            title="❌ Processing Error",
            border_style="red"
        ))
        console.print_exception()
        raise

    finally:
        stop_flag = True
        reader_thread.join(timeout=1)
        cap.release()
        out.release()
        
        # Final success message (only if no exception occurred)
        if 'stats' in locals() and 'frame_idx' in locals():
            console.print("\n")
            console.print(Panel(
                "\n".join([
                    "[bold green]Video processing completed successfully![/bold green]",
                    f"✨ Preset: [cyan]{preset}[/cyan]",
                    f"📊 Total Frames: [cyan]{frame_idx:,}[/cyan]",
                    f"⏱️ Processing Time: [cyan]{datetime.timedelta(seconds=int(time.time() - start_time))}[/cyan]",  # ✅ Fixed datetime usage
                    f"🚀 Average FPS: [cyan]{frame_idx / (time.time() - start_time):.2f}[/cyan]"
                ]),
                title="✅ Success",
                border_style="green"
            ))

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
    parser.add_argument('--flow_threshold_norm', type=float, default=0.5, help='Normalized flow threshold [0.0, 1.0].')
    
    # Depth-Based Segmentation
    parser.add_argument('--threshold_dark_norm', type=float, default=0.6, help='Normalized depth threshold [0.0, 1.0].')
    
    # Visual Effect Control
    parser.add_argument('--contrast_sensitivity_norm', type=float, default=0.9, help='Normalized contrast sensitivity [0.0, 1.0].')
    
    # Optical Flow Visualization
    parser.add_argument('--flow_brightness', type=float, default=3.0, help='Controls brightness of optical flow visualization (0.5 to 3.0).')
    
    # Performance Optimization
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing.')
    
    # Preset Selection
    parser.add_argument('--preset', type=str, default='balanced', choices=['speed', 'balanced', 'quality', 'max_quality'], help='Preset configuration.')
    
    # Add flow method argument
    parser.add_argument('--flow_method', type=str, default='farneback',
                       choices=['farneback', 'fastflownet'],
                       help='Optical flow method to use')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_video).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        console.print(f"[bold blue]Starting video processing...[/bold blue] {emoji.emojize(':video_camera:')}")
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
            preset=args.preset,
            flow_method=args.flow_method
        )
    except Exception as e:
        console.print(f"{emoji.emojize(':cross_mark:')} [bold red]Error processing video:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
