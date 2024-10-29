import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
import json
from rich.console import Console
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
import time

class TestVisualizer:
    def __init__(self, stats_file="outputs/processing_stats.json", test_results_path="test_results"):
        self.results_path = Path(test_results_path)
        self.results_path.mkdir(exist_ok=True)
        self.outputs_path = Path("outputs")
        self.outputs_path.mkdir(exist_ok=True)
        self.console = Console()
        
        # Load real processing stats
        try:
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
        except FileNotFoundError:
            self.console.print(f"[red]Error: Stats file not found at {stats_file}[/red]")
            self.stats = None
            
    def create_performance_plot(self):
        """Creates a comprehensive performance visualization plot."""
        if not self.stats:
            self.console.print("[red]No stats available for plotting[/red]")
            return

        # Extract frame metrics from the list format
        frame_data = []
        if 'frame_metrics' in self.stats:
            for i, metrics in enumerate(self.stats['frame_metrics']):
                frame_data.append({
                    'Frame': i,
                    'Time (s)': metrics[0],
                    'Memory (MB)': metrics[1],  # Already in MB
                    'FPS': metrics[2],
                    'CPU Load (%)': metrics[3],
                    'GPU Util (%)': metrics[4]
                })
        
        if not frame_data:
            self.console.print("[red]No valid frame data found for plotting[/red]")
            return
            
        frame_df = pd.DataFrame(frame_data)
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Memory usage plot
        ax.plot(frame_df['Time (s)'], frame_df['Memory (MB)'], 
                label='Memory (MB)', color='purple', linewidth=2)
        
        # CPU/GPU overlay
        ax2 = ax.twinx()
        ax2.plot(frame_df['Time (s)'], frame_df['CPU Load (%)'], 
                label='CPU Load (%)', color='orange', linewidth=1.5, linestyle='--')
        ax2.plot(frame_df['Time (s)'], frame_df['GPU Util (%)'], 
                label='GPU Util (%)', color='blue', linewidth=1.5, linestyle=':')
        
        # FPS scatter overlay
        scatter = ax.scatter(frame_df['Time (s)'], frame_df['Memory (MB)'],
                           c=frame_df['FPS'], cmap='plasma', 
                           s=60, alpha=0.6, label='FPS')
        
        # Styling
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Memory (MB)', fontsize=12, color='purple')
        ax2.set_ylabel('CPU/GPU Load (%)', fontsize=12)
        plt.title("Performance Metrics Over Time")
        
        # Legends
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.colorbar(scatter, ax=ax, label='FPS')
        
        try:
            # Save the plot with correct matplotlib options
            plt.savefig(
                self.results_path / 'performance_metrics.png',
                dpi=300,
                bbox_inches='tight',
                format='png'
            )
            plt.close()
            
            output_file = self.results_path / 'performance_metrics.png'
            if output_file.exists():
                self.console.print(f"[green]✓ Successfully saved plot to {output_file}[/green]")
                self.console.print(f"[blue]File size: {output_file.stat().st_size / 1024:.2f} KB[/blue]")
            else:
                self.console.print("[red]Failed to save plot - file not found[/red]")
                
        except Exception as e:
            self.console.print(f"[red]Error saving plot: {str(e)}[/red]")
            return False
            
        return True
        
if __name__ == "__main__":
    visualizer = TestVisualizer()
    visualizer.create_performance_plot()
        
       