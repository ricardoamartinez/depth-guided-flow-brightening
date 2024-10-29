import requests
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

console = Console()

def download_weights():
    """Download FastFlowNet weights"""
    # Direct link to weights file on GitHub LFS
    urls = [
        "https://media.githubusercontent.com/media/ltkong218/FastFlowNet/main/checkpoints/fastflownet_ft_mix.pth",
        "https://raw.githubusercontent.com/ltkong218/FastFlowNet/main/checkpoints/fastflownet_ft_mix.pth"
    ]
    
    # Define destination path
    destination = Path(__file__).parent.parent / 'models' / 'checkpoints' / 'fastflownet_ft_mix.pth'
    
    # Ensure directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold cyan]Downloading FastFlowNet weights...[/bold cyan]")
    
    for url in urls:
        try:
            console.print(f"\nTrying URL: [cyan]{url}[/cyan]")
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with Progress(
                *Progress.get_default_columns(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Downloading...", total=total_size)
                
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
            
            if destination.exists() and destination.stat().st_size > 1000000:
                console.print(f"[green]Successfully downloaded weights to {destination}[/green]")
                return True
                
        except Exception as e:
            console.print(f"[yellow]Failed with error: {str(e)}[/yellow]")
            continue
    
    # If all downloads fail, provide manual instructions
    console.print("\n[red]All automatic downloads failed.[/red]")
    console.print("\n[yellow]Please download manually:[/yellow]")
    console.print("1. Visit: [cyan]https://github.com/ltkong218/FastFlowNet/tree/main/checkpoints[/cyan]")
    console.print("2. Download: [cyan]fastflownet_ft_mix.pth[/cyan]")
    console.print("3. Create directory: [cyan]mkdir -p src/models/checkpoints[/cyan]")
    console.print("4. Move the file to: [cyan]src/models/checkpoints/fastflownet_ft_mix.pth[/cyan]")
    
    return False

def main():
    download_weights()

if __name__ == "__main__":
    main() 