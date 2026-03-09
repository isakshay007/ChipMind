"""Semantic retriever using FAISS and sentence-transformers."""

import pickle
from pathlib import Path

import faiss
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


class SemanticRetriever:
    """FAISS-based semantic search using local sentence-transformers embeddings."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        import os
        import sys
        import logging
        import warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        
        from sentence_transformers import SentenceTransformer

        # Redirect stdout/stderr to completely silence hardcoded HuggingFace C-bindings
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                self.model = SentenceTransformer(embedding_model)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []
        self.dim: int = 0

    def build_index(self, chunks: list[dict], batch_size: int = 256):
        """Build FAISS index from chunks.

        1. Extract embedding_text from each chunk
        2. Generate embeddings in batches with progress bar
        3. Build FAISS IndexFlatIP (inner product / cosine similarity)
        4. Store chunk metadata in parallel list
        """
        texts = []
        self.metadata = []
        for c in chunks:
            text = c.get("embedding_text", "")
            if not text:
                continue
            texts.append(text)
            self.metadata.append({k: v for k, v in c.items() if k != "embedding_text"})

        if not texts:
            console.print("[yellow]No texts to embed[/yellow]")
            return

        embeddings_list: list[np.ndarray] = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding chunks...", total=n_batches)
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                emb = self.model.encode(batch, show_progress_bar=False)
                embeddings_list.append(emb.astype(np.float32))
                progress.advance(task)

        embeddings = np.vstack(embeddings_list)
        self.dim = embeddings.shape[1]

        # Normalize for cosine similarity (inner product of normalized = cosine)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        index_size_mb = self.index.ntotal * self.dim * 4 / (1024 * 1024)  # float32
        console.print(
            f"[green]FAISS index: {self.index.ntotal} vectors, "
            f"dim={self.dim}, size≈{index_size_mb:.1f} MB[/green]"
        )

    def save(self, directory: str):
        """Save FAISS index and metadata to directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, directory: str):
        """Load FAISS index and metadata from directory."""
        path = Path(directory)
        self.index = faiss.read_index(str(path / "faiss.index"))
        self.dim = self.index.d
        with open(path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Search the index. Returns list of {**chunk_metadata, "score": float}."""
        if self.index is None or not self.metadata:
            return []
        q_emb = self.model.encode([query], show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, min(k, len(self.metadata)))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)
            results.append(meta)
        return results
