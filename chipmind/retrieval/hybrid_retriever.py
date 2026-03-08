"""Hybrid retriever combining semantic (FAISS) and keyword (BM25) with RRF."""

import pickle
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from chipmind.retrieval.semantic_retriever import SemanticRetriever
from chipmind.retrieval.keyword_retriever import KeywordRetriever

console = Console()

RRF_K = 60


class HybridRetriever:
    """Hybrid search using Reciprocal Rank Fusion."""

    def __init__(self, semantic: SemanticRetriever, keyword: KeywordRetriever):
        self.semantic = semantic
        self.keyword = keyword

    def search(
        self,
        query: str,
        k: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        fetch_k: int = 20,
    ) -> list[dict]:
        """Hybrid search using Reciprocal Rank Fusion.

        RRF: rrf_score(d) = sum(weight_i / (k_constant + rank_i))
        """
        sem_results = self.semantic.search(query, k=fetch_k)
        kw_results = self.keyword.search(query, k=fetch_k)

        # Build doc_id -> chunk mapping (use chunk_id)
        scores: dict[str, float] = {}
        sem_ranks: dict[str, int] = {}
        kw_ranks: dict[str, int] = {}
        doc_store: dict[str, dict] = {}

        for rank, r in enumerate(sem_results, start=1):
            doc_id = r.get("chunk_id", str(id(r)))
            doc_store[doc_id] = dict(r)
            scores[doc_id] = scores.get(doc_id, 0) + semantic_weight / (RRF_K + rank)
            sem_ranks[doc_id] = rank

        for rank, r in enumerate(kw_results, start=1):
            doc_id = r.get("chunk_id", str(id(r)))
            if doc_id not in doc_store:
                doc_store[doc_id] = dict(r)
            scores[doc_id] = scores.get(doc_id, 0) + keyword_weight / (RRF_K + rank)
            kw_ranks[doc_id] = rank

        # Sort by RRF score (descending)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Deduplicate: for verilog_code by module_name, for eda_doc by section_title
        # Keep the one with highest rrf_score (first in sorted order)
        seen: set[str] = set()
        results = []
        for doc_id in sorted_ids:
            r = doc_store[doc_id]
            chunk_type = r.get("chunk_type", "")
            if chunk_type == "verilog_code":
                dedup_key = r.get("module_name") or r.get("chunk_id") or doc_id
            elif chunk_type == "eda_doc":
                dedup_key = r.get("section_title") or r.get("chunk_id") or doc_id
            else:
                dedup_key = r.get("chunk_id") or doc_id

            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            out = dict(r)
            out["rrf_score"] = scores[doc_id]
            out["semantic_rank"] = sem_ranks.get(doc_id)
            out["keyword_rank"] = kw_ranks.get(doc_id)
            results.append(out)
            if len(results) >= k:
                break
        return results

    def search_code(self, query: str, k: int = 5) -> list[dict]:
        """Search only verilog_code chunks."""
        results = self.search(query, k=k * 3)
        return [r for r in results if r.get("chunk_type") == "verilog_code"][:k]

    def search_docs(self, query: str, k: int = 5) -> list[dict]:
        """Search only eda_doc chunks."""
        results = self.search(query, k=k * 3)
        return [r for r in results if r.get("chunk_type") == "eda_doc"][:k]

    @classmethod
    def build_and_save(
        cls,
        chunks_path: str,
        output_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> "HybridRetriever":
        """One-shot: load chunks, build both indexes, save everything."""
        import json

        path = Path(chunks_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        console.print("[bold]Loading chunks...[/bold]")
        chunks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        console.print(f"Loaded {len(chunks)} chunks")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Build semantic index
        console.print("\n[bold]Building semantic index (FAISS)...[/bold]")
        t0 = time.perf_counter()
        semantic = SemanticRetriever(embedding_model=embedding_model)
        semantic.build_index(chunks)
        semantic.save(str(out / "semantic"))
        t1 = time.perf_counter()
        console.print(f"Semantic index built in {t1 - t0:.1f}s")

        # Build keyword index
        console.print("\n[bold]Building keyword index (BM25)...[/bold]")
        t2 = time.perf_counter()
        keyword = KeywordRetriever()
        keyword.build_index(chunks)
        keyword.save(str(out / "bm25.pkl"))
        t3 = time.perf_counter()
        console.print(f"Keyword index built in {t3 - t2:.1f}s")

        # Save metadata for load
        with open(out / "hybrid_meta.pkl", "wb") as f:
            pickle.dump({"embedding_model": embedding_model}, f)

        total = t3 - t0
        faiss_size = (out / "semantic" / "faiss.index").stat().st_size / (1024 * 1024)
        bm25_size = (out / "bm25.pkl").stat().st_size / (1024 * 1024)
        table = Table(title="Index Build Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total build time", f"{total:.1f}s")
        table.add_row("FAISS index size", f"{faiss_size:.2f} MB")
        table.add_row("BM25 index size", f"{bm25_size:.2f} MB")
        console.print(table)

        return cls(semantic, keyword)

    @classmethod
    def load(cls, directory: str) -> "HybridRetriever":
        """Load both indexes from directory."""
        path = Path(directory)
        meta_path = path / "hybrid_meta.pkl"
        embedding_model = "all-MiniLM-L6-v2"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                embedding_model = meta.get("embedding_model", embedding_model)
        semantic = SemanticRetriever(embedding_model=embedding_model)
        semantic.load(str(path / "semantic"))
        keyword = KeywordRetriever()
        keyword.load(str(path / "bm25.pkl"))
        return cls(semantic, keyword)
