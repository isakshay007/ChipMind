#!/usr/bin/env python3
"""
Extract and consolidate Verilog modules from all downloaded sources into a
unified format for the RAG pipeline.
"""

import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Project root: data/scripts/ -> go up 2 levels
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "verilogdb"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Module extraction regex: captures full module block (module name ... endmodule)
MODULE_PATTERN = re.compile(
    r"(module\s+\w+[\s\S]*?endmodule)",
    re.IGNORECASE | re.MULTILINE,
)

# Minimum code length to consider valid
MIN_CODE_LEN = 20

console = Console()


@dataclass
class VerilogModule:
    """Unified representation of a Verilog module."""

    module_id: str
    code: str
    description: str
    source: str
    has_description: bool


def _normalize_code(code: str) -> str:
    """Normalize code for hashing (strip, collapse whitespace)."""
    return " ".join(code.split())


def _code_hash(code: str) -> str:
    """Generate short hash of normalized code."""
    return hashlib.md5(_normalize_code(code).encode()).hexdigest()[:12]


def _extract_modules_from_text(text: str) -> list[str]:
    """Extract all module blocks from text using regex."""
    if not text or len(text.strip()) < MIN_CODE_LEN:
        return []
    matches = MODULE_PATTERN.findall(text)
    return [m.strip() for m in matches if len(m.strip()) >= MIN_CODE_LEN]


def extract_davide221() -> list[VerilogModule]:
    """Extract from davide221/verilog-raw-100k parquet files."""
    modules: list[VerilogModule] = []
    parquet_dir = RAW_DIR / "davide221_verilog-raw-100k" / "data"
    if not parquet_dir.exists():
        return modules

    parquet_files = sorted(parquet_dir.glob("train-*.parquet"))
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            text = row.get("text", "")
            if not isinstance(text, str):
                continue
            extracted = _extract_modules_from_text(text)
            for code in extracted:
                h = _code_hash(code)
                modules.append(
                    VerilogModule(
                        module_id=h,
                        code=code,
                        description="",
                        source="davide221",
                        has_description=False,
                    )
                )
    return modules


def extract_mg_verilog() -> list[VerilogModule]:
    """Extract from GaTech-EIC/MG-Verilog arrow file."""
    modules: list[VerilogModule] = []
    arrow_path = RAW_DIR / "GaTech-EIC_MG-Verilog" / "merged_dataset" / "data-00000-of-00001.arrow"
    if not arrow_path.exists():
        return modules

    rows: list[dict] = []
    try:
        from datasets import Dataset

        ds = Dataset.from_file(str(arrow_path))
        rows = [ds[i] for i in range(len(ds))]
    except ImportError:
        try:
            import pyarrow as pa

            with pa.memory_map(str(arrow_path), "r") as f:
                reader = pa.ipc.open_stream(f)
                table = reader.read_all()
            for i in range(table.num_rows):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                rows.append(row)
        except Exception as e:
            console.print(f"[yellow]Could not load MG-Verilog: {e}[/yellow]")
            return modules

    for row in rows:
        code = row.get("code", "")
        if not code or len(code.strip()) < MIN_CODE_LEN:
            continue

        desc_obj = row.get("description") or {}
        if isinstance(desc_obj, dict):
            desc = (
                desc_obj.get("high_level_global_summary")
                or desc_obj.get("detailed_global_summary")
                or desc_obj.get("block_summary")
                or ""
            )
        else:
            desc = ""

        if isinstance(desc, dict):
            desc = str(desc)
        desc = (desc or "").strip()

        h = _code_hash(code)
        modules.append(
            VerilogModule(
                module_id=h,
                code=code.strip(),
                description=desc,
                source="mg_verilog",
                has_description=bool(desc),
            )
        )
    return modules


def extract_vgen() -> list[VerilogModule]:
    """Extract from VGen .v files."""
    modules: list[VerilogModule] = []
    vgen_dir = RAW_DIR / "VGen"
    if not vgen_dir.exists():
        return modules

    for vfile in vgen_dir.rglob("*.v"):
        try:
            text = vfile.read_text(errors="replace")
        except OSError:
            continue
        extracted = _extract_modules_from_text(text)
        for code in extracted:
            h = _code_hash(code)
            modules.append(
                VerilogModule(
                    module_id=h,
                    code=code,
                    description="",
                    source="vgen",
                    has_description=False,
                )
            )
    return modules


def deduplicate(modules: list[VerilogModule]) -> tuple[list[VerilogModule], int]:
    """Remove exact duplicates by code hash. Returns (deduped_list, removed_count)."""
    seen: set[str] = set()
    deduped: list[VerilogModule] = []
    for m in modules:
        norm = _normalize_code(m.code)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(m)
    return deduped, len(modules) - len(deduped)


def _module_name_prefix(code: str) -> str:
    """Extract module name (first word after 'module') from code."""
    match = re.search(r"module\s+(\w+)", code, re.IGNORECASE)
    return match.group(1).lower() if match else "unknown"


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    console.print(Panel("[bold]ChipMind — Verilog Module Extraction[/bold]", style="blue"))

    # Extract from all sources
    console.print("\n[bold]Extracting from sources...[/bold]")
    davide = extract_davide221()
    mg = extract_mg_verilog()
    vgen = extract_vgen()

    all_modules = davide + mg + vgen
    before_dedup = len(all_modules)

    # Deduplicate
    all_modules, dup_count = deduplicate(all_modules)

    # Save all_modules.jsonl
    output_path = PROCESSED_DIR / "all_modules.jsonl"
    with open(output_path, "w") as f:
        for m in all_modules:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
    console.print(f"[green]Saved {len(all_modules)} modules to {output_path}[/green]")

    # Save sample
    sample_path = PROCESSED_DIR / "sample_modules.jsonl"
    with open(sample_path, "w") as f:
        for m in all_modules[:100]:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
    console.print(f"[green]Saved sample (100 modules) to {sample_path}[/green]")

    # Stats
    with_desc = sum(1 for m in all_modules if m.has_description)
    code_lines = [len(m.code.splitlines()) for m in all_modules]
    avg_lines = sum(code_lines) / len(code_lines) if code_lines else 0
    name_prefixes = Counter(_module_name_prefix(m.code) for m in all_modules)
    top_prefixes = name_prefixes.most_common(20)

    table = Table(title="Extraction Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("davide221 (100K parquet)", str(len(davide)))
    table.add_row("mg_verilog (MG-Verilog)", str(len(mg)))
    table.add_row("vgen (.v files)", str(len(vgen)))
    table.add_row("Total before dedup", str(before_dedup))
    table.add_row("Duplicates removed", str(dup_count))
    table.add_row("Total after dedup", str(len(all_modules)))
    table.add_row("With descriptions", str(with_desc))
    table.add_row("Without descriptions", str(len(all_modules) - with_desc))
    table.add_row("Avg code length (lines)", f"{avg_lines:.1f}")
    console.print(table)

    tree = Tree("[bold]Top 20 module name prefixes[/bold]")
    for name, count in top_prefixes:
        tree.add(f"[cyan]{name}[/cyan]: {count}")
    console.print(tree)

    return 0


if __name__ == "__main__":
    sys.exit(main())
