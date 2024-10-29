import torch
from rich.console import Console

console = Console()

def check_cuda_setup():
    """Check CUDA availability and setup"""
    console.print("\n[bold cyan]Checking CUDA Setup...[/bold cyan]")
    
    # Check PyTorch installation
    console.print(f"PyTorch version: [green]{torch.__version__}[/green]")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    console.print(f"CUDA available: [{'green' if cuda_available else 'red'}]{cuda_available}[/{'green' if cuda_available else 'red'}]")
    
    if cuda_available:
        # Get CUDA version
        console.print(f"CUDA version: [green]{torch.version.cuda}[/green]")
        
        # Get device count and properties
        device_count = torch.cuda.device_count()
        console.print(f"GPU devices found: [green]{device_count}[/green]")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            console.print(f"\nGPU {i}: [cyan]{props.name}[/cyan]")
            console.print(f"  Memory: [green]{props.total_memory / 1024**2:.0f} MB[/green]")
            console.print(f"  Compute capability: [green]{props.major}.{props.minor}[/green]")
    else:
        console.print("\n[yellow]CUDA is not available. Please install CUDA and PyTorch with CUDA support.[/yellow]")
        console.print("\nInstallation instructions:")
        console.print("1. Install NVIDIA GPU drivers")
        console.print("2. Install CUDA Toolkit 11.3")
        console.print("3. Install PyTorch with CUDA support:")
        console.print("[cyan]pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113[/cyan]")

if __name__ == "__main__":
    check_cuda_setup() 