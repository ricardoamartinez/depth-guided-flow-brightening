import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import os
from rich.console import Console

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
    """Setup FastFlowNet and return the model class if available"""
    try:
        # Add models directory to path
        project_root = get_project_root()
        models_path = project_root / 'src' / 'models'
        if str(models_path) not in sys.path:
            sys.path.append(str(models_path))

        # Try importing FastFlowNet
        from models.FastFlowNet_v2 import FastFlowNet
        console.print("[green]Successfully imported FastFlowNet[/green]")
        return FastFlowNet
    except ImportError as e:
        console.print(f"[yellow]Could not import FastFlowNet. Error: {e}[/yellow]")
        console.print("[yellow]Checking if FastFlowNet repository exists...[/yellow]")
        
        # Check if FastFlowNet repo exists
        if not (models_path / 'models' / 'FastFlowNet_v2.py').exists():
            console.print("[red]FastFlowNet repository not found. Please clone it first:[/red]")
            console.print("\n[cyan]Run these commands from your project root:[/cyan]")
            console.print("git clone https://github.com/ltkong218/FastFlowNet.git src/models/FastFlowNet")
            console.print("cp src/models/FastFlowNet/models/* src/models/models/")
            console.print("mkdir -p src/models/checkpoints")
            console.print("wget -O src/models/checkpoints/fastflownet_ft_mix.pth https://github.com/ltkong218/FastFlowNet/raw/master/checkpoints/fastflownet_ft_mix.pth")
        return None

def load_fastflownet_model(device='cuda'):
    """Load FastFlowNet model with weights"""
    FastFlowNet = setup_fastflownet()
    if FastFlowNet is None:
        return None
        
    try:
        model = FastFlowNet()
        weights_path = get_project_root() / 'src' / 'models' / 'checkpoints' / 'fastflownet_ft_mix.pth'
        
        if not weights_path.exists():
            console.print(f"[red]Weights not found at {weights_path}[/red]")
            return None
            
        model.load_state_dict(torch.load(str(weights_path), weights_only=True))
        model.eval()
        
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            console.print("[green]FastFlowNet loaded on GPU[/green]")
        else:
            model = model.cpu()
            console.print("[yellow]FastFlowNet loaded on CPU[/yellow]")
            
        return model
    except Exception as e:
        console.print(f"[red]Error loading FastFlowNet model: {e}[/red]")
        return None 