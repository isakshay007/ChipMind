"""Build the full retrieval index from chunks."""

import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Project root: chipmind/ingestion/ -> go up 2 levels
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "all_chunks.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "indexes"

console = Console()


def main() -> int:
    from chipmind.retrieval.hybrid_retriever import HybridRetriever
    from chipmind.config import settings

    console.print(Panel("[bold]ChipMind — Build Retrieval Index[/bold]", style="blue"))

    if not CHUNKS_PATH.exists():
        console.print(f"[red]Chunks file not found: {CHUNKS_PATH}[/red]")
        console.print("Run 'make chunk' first.")
        return 1

    t0 = time.perf_counter()
    retriever = HybridRetriever.build_and_save(
        chunks_path=str(CHUNKS_PATH),
        output_dir=str(OUTPUT_DIR),
        embedding_model=settings.EMBEDDING_MODEL,
    )
    elapsed = time.perf_counter() - t0
    console.print(f"\n[green]Index built successfully in {elapsed:.1f}s[/green]")
    console.print(f"Output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
