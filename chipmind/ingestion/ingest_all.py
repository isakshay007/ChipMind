"""Master ingestion script: runs VerilogChunker and DocChunker, saves combined chunks."""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Project root: chipmind/ingestion/ -> go up 2 levels
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODULES_PATH = PROJECT_ROOT / "data" / "processed" / "all_modules.jsonl"
DOCS_DIR = PROJECT_ROOT / "data" / "raw" / "eda_docs"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "all_chunks.jsonl"

console = Console()


def main() -> int:
    from chipmind.ingestion.verilog_chunker import VerilogChunker
    from chipmind.ingestion.doc_chunker import DocChunker
    from dataclasses import asdict

    console.print(Panel("[bold]ChipMind — Chunking Pipeline[/bold]", style="blue"))

    # 1. Verilog chunks
    console.print("\n[bold]1. Processing Verilog modules...[/bold]")
    verilog_chunker = VerilogChunker(MODULES_PATH)
    verilog_chunks = verilog_chunker.process_all()

    # 2. Doc chunks
    console.print("\n[bold]2. Processing EDA documentation...[/bold]")
    doc_chunker = DocChunker(DOCS_DIR)
    doc_chunks = doc_chunker.process_all()

    # 3. Combine and save
    all_chunks: list[dict] = []
    for c in verilog_chunks:
        d = asdict(c)
        all_chunks.append(d)
    for c in doc_chunks:
        d = asdict(c)
        all_chunks.append(d)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    console.print(f"\n[green]Saved {len(all_chunks)} chunks to {OUTPUT_PATH}[/green]")

    # 4. Combined stats
    verilog_with_desc = sum(1 for c in verilog_chunks if getattr(c, "has_description", False))
    table = Table(title="Combined Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Verilog chunks", str(len(verilog_chunks)))
    table.add_row("Total doc chunks", str(len(doc_chunks)))
    table.add_row("Total combined", str(len(all_chunks)))
    table.add_row("Verilog with descriptions", str(verilog_with_desc))
    table.add_row("Description coverage", f"{100 * verilog_with_desc / len(verilog_chunks):.1f}%" if verilog_chunks else "0%")
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
