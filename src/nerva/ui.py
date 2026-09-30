"""
NERVA-ML Terminal UI & Logging Setup
Rich console formatting with file-based detailed debug logs.
"""

import os
import sys
import logging
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

def setup_logging(log_dir: str = "logs") -> str:
    """Configures detailed file logging and returns the log file path."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return log_path

def print_banner(mode: str, profile: str, seed: int):
    """Prints the NERVA-ML terminal banner."""
    mode_text = "DEMO / SYNTHETIC DATA ONLY" if "synthetic" in mode.lower() or "demo" in mode.lower() else "REAL"
    banner_content = Text()
    banner_content.append("NERVA-ML\n", style="bold cyan")
    banner_content.append("Neuro-Evolutionary Regional Vulnerability & Robustness Framework\n", style="white")
    banner_content.append(f"Mode: {mode_text}   Profile: {profile}   Seed: {seed}", style="dim yellow")
    
    console.print(Panel(banner_content, border_style="cyan", expand=False))

def stage_header(num: int, title: str):
    """Prints clean stage headers."""
    console.print(f"\n[bold green][{num}/8] {title}[/bold green]")
    console.print("[dim]" + "-" * 60 + "[/dim]")
