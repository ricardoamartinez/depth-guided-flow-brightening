import os
from pathlib import Path
from rich.console import Console
import shutil
import subprocess
import sys

console = Console()

def get_project_root():
    """Get the path to the project root directory."""
    current_path = Path(__file__).parent
    while not (current_path / 'src').exists():
        if current_path == current_path.parent:
            raise RuntimeError("Could not find project root")
        current_path = current_path.parent
    return current_path

def setup_fastflownet():
    """Setup FastFlowNet repository and model files"""
    project_root = get_project_root()
    
    # Create necessary directories
    models_dir = project_root / 'src' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    fastflownet_dir = models_dir / 'FastFlowNet'
    if not fastflownet_dir.exists():
        console.print("[cyan]Cloning FastFlowNet repository...[/cyan]")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/ltkong218/FastFlowNet.git',
            str(fastflownet_dir)
        ], check=True)
    
    # Create models and checkpoints directories
    (models_dir / 'models').mkdir(exist_ok=True)
    (models_dir / 'checkpoints').mkdir(exist_ok=True)
    
    # Copy model files
    console.print("[cyan]Copying model files...[/cyan]")
    for file in (fastflownet_dir / 'models').glob('*.py'):
        shutil.copy2(file, models_dir / 'models' / file.name)
    
    console.print("[green]FastFlowNet setup completed![/green]")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Run: [cyan]python src/scripts/download_weights.py[/cyan]")
    console.print("2. Try FastFlowNet with: [cyan]python src/scripts/process_video.py --flow_method fastflownet[/cyan]")

if __name__ == "__main__":
    try:
        setup_fastflownet()
    except Exception as e:
        console.print(f"[red]Error setting up FastFlowNet: {e}[/red]")
        sys.exit(1) 