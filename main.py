#!/usr/bin/env python3
"""
NERVA-ML: Neuro-Evolutionary Regional Vulnerability & Robustness Framework
CLI Entrypoint
"""

import sys
import os
import argparse
import logging
import traceback

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure src is at the top of module lookup path
current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from nerva.config import load_config
from nerva.pipeline import run_pipeline
from nerva.ui import console, setup_logging
from nerva.utils.errors import NervaError


def parse_args():
    parser = argparse.ArgumentParser(
        description="NERVA-ML: Neuro-Evolutionary Regional Vulnerability & Robustness Framework"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in offline synthetic demo mode (uses demo profile and synthetic data, watermarked outputs)."
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["demo", "hackathon", "full"],
        help="Configuration profile override ('demo', 'hackathon', 'full')."
    )
    parser.add_argument(
        "--disease",
        nargs="+",
        default=None,
        help="Subset of disease IDs to analyze (e.g., --disease PD AD)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "config.yaml"),
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override project random seed."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine profile override
    profile = args.profile
    if args.demo and profile is None:
        profile = "demo"

    # Setup file logging
    log_file = setup_logging()

    try:
        cfg = load_config(args.config, profile_override=profile)
        if args.seed is not None:
            cfg.project.seed = args.seed

        logging.info("Starting NERVA-ML with profile=%s, seed=%d, demo=%s", cfg.project.profile, cfg.project.seed, args.demo)
        
        # Run master 8-stage pipeline
        out_dir = run_pipeline(cfg, is_demo=args.demo, diseases_filter=args.disease)
        
        logging.info("NERVA-ML pipeline completed successfully. Results saved to: %s", out_dir)
        sys.exit(0)

    except NervaError as e:
        console.print(f"\n[bold red]ERROR: {e.message}[/bold red]")
        if e.hint:
            console.print(f"[bold yellow]HINT: {e.hint}[/bold yellow]\n")
        logging.error("NERVA-ML error: %s", e, exc_info=True)
        sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Execution interrupted by user.[/yellow]")
        sys.exit(130)

    except Exception as e:
        console.print(f"\n[bold red]FATAL UNEXPECTED ERROR: {e}[/bold red]")
        console.print(f"[dim]Traceback logged to: {log_file}[/dim]\n")
        logging.critical("Unexpected error: %s", e, exc_info=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
