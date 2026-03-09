"""
VerilogEval benchmark entry point.

Usage:
    # Quick test (10 problems)
    python -m chipmind.evaluation.run_eval --max-problems 10

    # Full benchmark
    python -m chipmind.evaluation.run_eval

    # Single mode only
    python -m chipmind.evaluation.run_eval --modes baseline rag_only

    # Resume from crash (reads existing results file, skips completed)
    python -m chipmind.evaluation.run_eval --resume

    # Use a different model
    python -m chipmind.evaluation.run_eval --model llama-3.1-8b-instant
"""

import argparse
import json
from pathlib import Path

from chipmind.evaluation.verilog_eval_runner import (
    DETAILS_FILE,
    RESULTS_FILE,
    VerilogEvalRunner,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ChipMind on VerilogEval benchmark"
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit number of problems",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Max debug iterations for agentic mode",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "rag_only", "chipmind_agentic"],
        help="Modes to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results (skip completed problem+mode combos)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="Model for evaluation (default: llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["groq", "nvidia"],
        default="groq",
        help="LLM provider: groq or nvidia",
    )
    args = parser.parse_args()

    # Clear old results when not resuming
    if not args.resume:
        for p in [RESULTS_FILE, DETAILS_FILE]:
            if p.exists():
                p.unlink()

    runner = VerilogEvalRunner(provider=args.provider, eval_model=args.model)

    existing: list[dict] = []
    if args.resume and RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        print("Skipping corrupted line in results file")
        print(f"Resuming: {len(existing)} existing results found")

    results = runner.run_benchmark(
        modes=args.modes,
        max_problems=args.max_problems,
        max_iterations=args.max_iterations,
        existing_results=existing if existing else None,
    )

    metrics = runner.compute_metrics(results)
    runner.print_report(metrics)
    runner.save_report(metrics, results)


if __name__ == "__main__":
    main()
