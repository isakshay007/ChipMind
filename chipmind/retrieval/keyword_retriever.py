"""Keyword retriever using BM25."""

import pickle
import random
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Verilog keywords to preserve (case-insensitive)
VERILOG_KEYWORDS = {
    "module", "endmodule", "input", "output", "inout", "wire", "reg", "assign",
    "always", "posedge", "negedge", "initial", "parameter", "localparam",
    "if", "else", "case", "casez", "casex", "endcase", "default",
    "for", "while", "begin", "end", "function", "endfunction",
    "task", "endtask", "generate", "endgenerate", "integer", "real",
    "and", "or", "not", "xor", "xnor", "nand", "nor",
    "buf", "bufif0", "bufif1", "notif0", "notif1",
    "signed", "unsigned", "logic", "bit", "byte", "shortint", "longint",
}

# Max docs for BM25Okapi (rank_bm25 segfaults on 80K+; 50K with optimizations)
BM25_MAX_DOCS = 50_000
TOKENIZE_BATCH_SIZE = 10_000


def _apply_verilog_patterns(text: str) -> str:
    """Replace Verilog compound patterns with single tokens. Uses str.replace for speed."""
    if not text:
        return ""
    t = text.lower()
    # str.replace is much faster than regex; order matters (longer first)
    # Include both compound and component tokens so "posedge clk" query matches
    t = t.replace("always @(posedge clk", "always_at_posedge_clk posedge_clk")
    t = t.replace("always @(negedge clk", "always_at_negedge_clk negedge_clk")
    t = t.replace("always @(posedge reset", "always_at_posedge_reset posedge_reset")
    t = t.replace("always @(negedge reset", "always_at_negedge_reset negedge_reset")
    t = t.replace("always @(posedge", "always_at_posedge")
    t = t.replace("always @(negedge", "always_at_negedge")
    t = t.replace("always @(*)", "always_at_star")
    t = t.replace("posedge clk", "posedge_clk")
    t = t.replace("negedge clk", "negedge_clk")
    t = t.replace("posedge reset", "posedge_reset")
    t = t.replace("negedge reset", "negedge_reset")
    return t


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25.

    - Apply Verilog pattern replacements (keep compound tokens together)
    - Lowercase (done in _apply_verilog_patterns)
    - Split on whitespace and Verilog delimiters
    - Keep tokens >= 2 chars
    - Preserve Verilog keywords
    """
    if not text:
        return []
    text = _apply_verilog_patterns(text)
    tokens = re.split(r"[\s()\[\]{}=+\-*/&|^~!<>@#,;:]+", text)
    result = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if len(t) >= 2 or t in VERILOG_KEYWORDS:
            result.append(t)
    return result


def _get_bm25_text(chunk: dict) -> str:
    """Get text to tokenize for BM25: code for verilog_code, text for eda_doc."""
    chunk_type = chunk.get("chunk_type", "")
    if chunk_type == "verilog_code":
        return chunk.get("code", "") or chunk.get("embedding_text", "")
    if chunk_type == "eda_doc":
        return chunk.get("text", "") or chunk.get("embedding_text", "")
    return chunk.get("embedding_text", "")


class KeywordRetriever:
    """BM25-based keyword search."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.metadata: list[dict] = []

    def build_index(self, chunks: list[dict]):
        """Build BM25 index from chunks. Uses code for verilog_code, text for eda_doc."""
        # Collect chunks with text to index
        to_index: list[dict] = []
        for c in chunks:
            text = _get_bm25_text(c)
            if not text or not text.strip():
                continue
            to_index.append(c)

        if not to_index:
            console.print("[yellow]No documents to index[/yellow]")
            return

        # Subsample if too large (BM25Okapi can segfault on 80K+ docs)
        if len(to_index) > BM25_MAX_DOCS:
            rng = random.Random(42)
            to_index = rng.sample(to_index, BM25_MAX_DOCS)
            console.print(
                f"[yellow]BM25: subsampled to {BM25_MAX_DOCS} docs (avoids segfault on large corpus)[/yellow]"
            )

        # Tokenize in batches with progress bar
        tokenized_docs: list[list[str]] = []
        self.metadata = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Tokenizing for BM25...", total=len(to_index))
            for i in range(0, len(to_index), TOKENIZE_BATCH_SIZE):
                batch = to_index[i : i + TOKENIZE_BATCH_SIZE]
                for c in batch:
                    text = _get_bm25_text(c)
                    tokens = _tokenize(text)
                    if tokens:
                        tokenized_docs.append(tokens)
                        self.metadata.append({k: v for k, v in c.items()})
                    progress.advance(task)

        if not tokenized_docs:
            console.print("[yellow]No documents to index after tokenization[/yellow]")
            return

        console.print(f"[dim]Building BM25Okapi with {len(tokenized_docs)} docs...[/dim]")
        self.bm25 = BM25Okapi(tokenized_docs)
        console.print(f"[green]BM25 index: {len(tokenized_docs)} documents[/green]")

    def save(self, path: str):
        """Save BM25 index and metadata using pickle (protocol=4 for large objects)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "metadata": self.metadata},
                f,
                protocol=4,
            )

    def load(self, path: str):
        """Load BM25 index and metadata."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.metadata = data["metadata"]

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Search using BM25. Returns list of {**chunk_metadata, "score": float}."""
        if self.bm25 is None or not self.metadata:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            meta = dict(self.metadata[idx])
            meta["score"] = float(scores[idx])
            results.append(meta)
        return results
