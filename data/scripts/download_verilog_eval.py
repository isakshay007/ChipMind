#!/usr/bin/env python3
"""
Download and parse NVlabs/verilog-eval benchmark for ChipMind.

Clones the repo to data/raw/verilog-eval/ and parses problem files to display
stats, sample problems, and categories.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

# Project root: data/scripts/ -> go up 2 levels
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "raw" / "verilog-eval"

VERILOG_EVAL_URL = "https://github.com/NVlabs/verilog-eval.git"

console = Console()


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    """Run command, return (success, stderr_or_stdout)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0, result.stderr or result.stdout
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, "git not found"
    except Exception as e:
        return False, str(e)


def clone_repo() -> bool:
    """Clone verilog-eval to EVAL_DIR. Returns True on success."""
    if EVAL_DIR.exists():
        console.print("[dim]Already exists: verilog-eval. Skipping clone.[/dim]")
        return True

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Cloning NVlabs/verilog-eval...", total=None)
        ok, msg = _run_cmd(["git", "clone", "--depth", "1", VERILOG_EVAL_URL, str(EVAL_DIR)])
        if ok:
            progress.update(task, description="[green]Cloned verilog-eval[/green]")
        else:
            progress.update(task, description=f"[red]Failed: {msg}[/red]")
            if EVAL_DIR.exists():
                shutil.rmtree(EVAL_DIR, ignore_errors=True)
            return False
    return True


def parse_problems() -> tuple[list[dict], dict[str, list[str]]]:
    """
    Parse problem files from dataset_spec-to-rtl and dataset_code-complete-iccad2023.
    Returns (list of problem dicts, categories -> problem_ids).
    """
    problems: list[dict] = []
    categories: dict[str, list[str]] = {}

    dataset_dirs = [
        EVAL_DIR / "dataset_spec-to-rtl",
        EVAL_DIR / "dataset_code-complete-iccad2023",
    ]

    # Pattern: Prob001_zero_prompt.txt, Prob001_zero_ref.sv, Prob001_zero_test.sv
    # Each dataset has same problem IDs but different prompt formats (spec vs interface)
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue

        dataset_name = dataset_dir.name
        if dataset_name not in categories:
            categories[dataset_name] = []

        # Find all _prompt.txt files
        prompt_files = list(dataset_dir.glob("*_prompt.txt"))
        if not prompt_files:
            continue
        for prompt_file in prompt_files:
            stem = prompt_file.stem  # e.g. Prob001_zero_prompt
            match = re.match(r"(Prob\d+)_(\w+)_prompt", stem)
            if not match:
                continue
            prob_id, suffix = match.groups()

            ref_file = dataset_dir / f"{prob_id}_{suffix}_ref.sv"
            test_file = dataset_dir / f"{prob_id}_{suffix}_test.sv"

            try:
                description = prompt_file.read_text(errors="replace")
                solution = ref_file.read_text(errors="replace") if ref_file.exists() else ""
                testbench = test_file.read_text(errors="replace") if test_file.exists() else ""
            except OSError:
                continue

            problem = {
                "id": f"{prob_id}_{suffix}",
                "dataset": dataset_name,
                "description": description.strip(),
                "solution": solution.strip(),
                "testbench": testbench.strip(),
            }
            problems.append(problem)
            categories[dataset_name].append(problem["id"])
            problems.append(problem)
            categories[dataset_name].append(problem["id"])

    return problems, categories


def main() -> int:
    EVAL_DIR.parent.mkdir(parents=True, exist_ok=True)

    console.print(Panel("[bold]ChipMind — VerilogEval Download[/bold]", style="blue"))

    if not clone_repo():
        console.print("[red]Failed to clone verilog-eval.[/red]")
        return 1

    console.print("\n[bold]Parsing problems...[/bold]")
    problems, categories = parse_problems()

    if not problems:
        console.print("[yellow]No problems found. Check dataset structure.[/yellow]")
        return 0

    # Summary
    console.print(f"\n[bold]Total problems:[/bold] {len(problems)}")

    # Categories
    tree = Tree("[bold]Categories[/bold]")
    for cat, ids in categories.items():
        branch = tree.add(f"[cyan]{cat}[/cyan] ({len(ids)} problems)")
        for pid in ids[:5]:
            branch.add(f"[dim]{pid}[/dim]")
        if len(ids) > 5:
            branch.add(f"[dim]... and {len(ids) - 5} more[/dim]")
    console.print(tree)

    # Sample problem
    sample = problems[0]
    console.print("\n[bold]Sample problem[/bold]")
    console.print(Panel(sample["description"], title="Description", border_style="green"))
    console.print(Panel(sample["solution"][:800] + ("..." if len(sample["solution"]) > 800 else ""),
                     title="Reference solution", border_style="blue"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
