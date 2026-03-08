#!/usr/bin/env python3
"""
Download VerilogDB and related Verilog datasets for ChipMind.

Tries HuggingFace first (VerilogDB, MG-Verilog, etc.), then falls back to
cloning GitHub repos: NVlabs/verilog-eval and shailja-thakur/VGen.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Project root: data/scripts/ -> go up 2 levels
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "verilogdb"

# HuggingFace repo IDs to try (VerilogDB paper mentions 20K+ verified modules)
HF_REPO_IDS = [
    "GaTech-EIC/verilogdb",
    "FICS-LLM/VerilogDB",
    "GaTech-EIC/MG-Verilog",
    "shailja/Verilog_GitHub",
    "davide221/verilog-raw-100k",
    "dakies/nvlabs-verilogeval",
]

# GitHub fallback repos
GITHUB_REPOS = [
    ("https://github.com/NVlabs/verilog-eval.git", "verilog-eval"),
    ("https://github.com/shailja-thakur/VGen.git", "VGen"),
]

console = Console()


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    """Run command, return (success, stderr_or_stdout)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0, result.stderr or result.stdout
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, "git not found - install git to use GitHub fallback"
    except Exception as e:
        return False, str(e)


def try_huggingface() -> bool:
    """Try downloading from HuggingFace. Returns True if any dataset was downloaded."""
    try:
        from huggingface_hub import HfApi, hf_hub_download, list_datasets
    except ImportError:
        console.print("[yellow]huggingface_hub not installed. Run: pip install huggingface-hub[/yellow]")
        return False

    api = HfApi()
    downloaded = False

    for repo_id in HF_REPO_IDS:
        try:
            # Check if it's a dataset (repo_id format: org/name)
            if "/" not in repo_id:
                continue
            parts = repo_id.split("/")
            if len(parts) != 2:
                continue

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Trying {repo_id}...", total=None)
                try:
                    # Try snapshot_download for full dataset
                    from huggingface_hub import snapshot_download

                    dest = RAW_DIR / repo_id.replace("/", "_")
                    dest.mkdir(parents=True, exist_ok=True)
                    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dest))
                    downloaded = True
                    progress.update(task, description=f"[green]Downloaded {repo_id}[/green]")
                except Exception as e:
                    err_str = str(e).lower()
                    if "404" in err_str or "not found" in err_str:
                        progress.update(task, description=f"[dim]Not found: {repo_id}[/dim]")
                    else:
                        progress.update(task, description=f"[yellow]Failed {repo_id}: {e}[/yellow]")
        except Exception as e:
            console.print(f"[dim]Skipped {repo_id}: {e}[/dim]")

    return downloaded


def try_github_fallback() -> bool:
    """Clone GitHub repos as fallback. Returns True if any was cloned."""
    any_ok = False
    for url, name in GITHUB_REPOS:
        dest = RAW_DIR / name
        if dest.exists():
            console.print(f"[dim]Already exists: {dest}[/dim]")
            any_ok = True
            continue

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Cloning {url}...", total=None)
            ok, msg = _run_cmd(["git", "clone", "--depth", "1", url, str(dest)])
            if ok:
                progress.update(task, description=f"[green]Cloned {name}[/green]")
                any_ok = True
            else:
                progress.update(task, description=f"[red]Failed: {msg[:60]}[/red]")
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)

    return any_ok


def collect_stats() -> tuple[int, int, list[tuple[Path, str]]]:
    """Count files, total size, and collect sample previews."""
    verilog_extensions = {".v", ".sv", ".vh", ".svh"}
    total_files = 0
    total_bytes = 0
    samples: list[tuple[Path, str]] = []

    for path in RAW_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in verilog_extensions:
            total_files += 1
            try:
                total_bytes += path.stat().st_size
                if len(samples) < 3:
                    content = path.read_text(errors="replace")[:500]
                    samples.append((path, content))
            except OSError:
                pass

    return total_files, total_bytes, samples


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    console.print(Panel("[bold]ChipMind — VerilogDB Download[/bold]", style="blue"))

    # Try HuggingFace first
    console.print("\n[bold]1. Trying HuggingFace datasets...[/bold]")
    hf_ok = try_huggingface()

    # Fallback to GitHub
    console.print("\n[bold]2. GitHub fallback...[/bold]")
    gh_ok = try_github_fallback()

    if not hf_ok and not gh_ok:
        console.print("[red]No data was downloaded. Check network and dependencies.[/red]")
        return 1

    # Stats
    console.print("\n[bold]3. Statistics[/bold]")
    total_files, total_bytes, samples = collect_stats()

    table = Table(title="Download Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Verilog files", str(total_files))
    table.add_row("Total size", f"{total_bytes / (1024*1024):.2f} MB")
    table.add_row("Output directory", str(RAW_DIR))
    console.print(table)

    if samples:
        console.print("\n[bold]Sample file previews:[/bold]")
        for path, content in samples:
            rel = path.relative_to(RAW_DIR)
            console.print(Panel(content, title=str(rel), border_style="dim"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
