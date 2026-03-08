"""Keyword retriever using BM25."""

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console

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


def _tokenize(text: str) -> list[str]:
    """Tokenize embedding_text for BM25.

    - Lowercase
    - Split on whitespace and Verilog delimiters: ( ) ; , [ ] { } = + - * / & | ^ ~ ! < > @ #
    - Keep tokens >= 2 chars
    - Preserve Verilog keywords
    """
    if not text:
        return []
    text = text.lower()
    tokens = re.split(r"[\s()\[\]{}=+\-*/&|^~!<>@#,;:]+", text)
    result = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if len(t) >= 2 or t in VERILOG_KEYWORDS:
            result.append(t)
    return result


class KeywordRetriever:
    """BM25-based keyword search."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.metadata: list[dict] = []

    def build_index(self, chunks: list[dict]):
        """Build BM25 index from chunks."""
        tokenized_docs = []
        self.metadata = []
        for c in chunks:
            text = c.get("embedding_text", "")
            if not text:
                continue
            tokens = _tokenize(text)
            if not tokens:
                continue
            tokenized_docs.append(tokens)
            self.metadata.append({k: v for k, v in c.items() if k != "embedding_text"})

        if not tokenized_docs:
            console.print("[yellow]No documents to index[/yellow]")
            return

        self.bm25 = BM25Okapi(tokenized_docs)
        console.print(f"[green]BM25 index: {len(tokenized_docs)} documents[/green]")

    def save(self, path: str):
        """Save BM25 index and metadata using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "metadata": self.metadata}, f)

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
            if scores[idx] <= 0:
                break
            meta = dict(self.metadata[idx])
            meta["score"] = float(scores[idx])
            results.append(meta)
        return results
