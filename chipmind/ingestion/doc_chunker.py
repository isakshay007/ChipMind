"""EDA documentation chunker for RAG pipeline."""

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Approximate: 4 chars per token for English/technical text
CHARS_PER_TOKEN = 4
MAX_CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN

console = Console()


@dataclass
class DocChunk:
    """Chunk representing EDA documentation for embedding."""

    chunk_id: str
    chunk_type: str = "eda_doc"
    text: str = ""
    source_file: str = ""
    source_tool: str = ""
    section_title: str = ""
    embedding_text: str = ""


def _infer_source_tool(filename: str) -> str:
    """Infer source tool from filename."""
    name = filename.lower()
    if name.startswith("yosys"):
        return "yosys"
    if name.startswith("chipverify"):
        return "chipverify"
    if name.startswith("hdlbits"):
        return "hdlbits"
    if name.startswith("asic_world"):
        return "asic_world"
    if "concepts" in name:
        return "concepts"
    return "unknown"


def _extract_section_title(text: str, filename: str) -> str:
    """Extract section title from content or filename."""
    lines = text.strip().split("\n")
    for line in lines[:10]:
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()[:100]
        if line and len(line) < 80 and line.isupper():
            return line[:100]
        if line and not line.startswith("#") and len(line) < 60:
            return line[:100]
    return Path(filename).stem.replace("_", " ").title()[:100]


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split text by section headers. Returns list of (title, content)."""
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_content: list[str] = []
    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+", line):
            if current_content:
                sections.append((current_title, "\n".join(current_content)))
            current_title = stripped.lstrip("#").strip()
            current_content = []
        elif stripped and stripped.isupper() and len(stripped) < 80:
            if current_content:
                sections.append((current_title, "\n".join(current_content)))
            current_title = stripped
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append((current_title or "Content", "\n".join(current_content)))
    return sections


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, avoiding mid-sentence splits."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Chunk text by paragraphs/sentences, never splitting mid-sentence."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para) + 2  # +2 for \n\n

        if current_len + para_len <= max_chars:
            current.append(para)
            current_len += para_len
        else:
            if current:
                chunks.append("\n\n".join(current))
            if len(para) > max_chars:
                sentences = _split_into_sentences(para)
                current = []
                current_len = 0
                for sent in sentences:
                    sent_len = len(sent) + 1
                    if current_len + sent_len <= max_chars:
                        current.append(sent)
                        current_len += sent_len
                    else:
                        if current:
                            chunks.append("\n".join(current))
                        current = [sent]
                        current_len = sent_len
                if current:
                    chunks.append("\n".join(current))
                    overlap_sentences = []
                    overlap_len = 0
                    for s in reversed(current):
                        if overlap_len + len(s) <= overlap_chars:
                            overlap_sentences.insert(0, s)
                            overlap_len += len(s)
                        else:
                            break
                    current = overlap_sentences
                    current_len = overlap_len
            else:
                current = [para]
                current_len = para_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _chunk_content(text: str, section_title: str) -> list[str]:
    """Apply chunking strategy: headers first, then paragraphs, max size with overlap."""
    sections = _split_by_headers(text)
    if len(sections) > 1:
        result: list[str] = []
        for title, content in sections:
            if len(content) <= MAX_CHUNK_CHARS:
                result.append(content)
            else:
                result.extend(_chunk_text(content, MAX_CHUNK_CHARS, OVERLAP_CHARS))
        return result
    if len(text) <= MAX_CHUNK_CHARS:
        return [text] if text.strip() else []
    return _chunk_text(text, MAX_CHUNK_CHARS, OVERLAP_CHARS)


class DocChunker:
    """Process EDA documentation files into chunks."""

    def __init__(self, docs_dir: Path | str):
        self.docs_dir = Path(docs_dir)

    def process_all(self) -> list[DocChunk]:
        """Read all .txt files, create DocChunk(s) for each. Returns all chunks."""
        chunks: list[DocChunk] = []
        source_counts: Counter[str] = Counter()

        if not self.docs_dir.exists():
            console.print(f"[yellow]Docs directory not found: {self.docs_dir}[/yellow]")
            return chunks

        txt_files = sorted(self.docs_dir.glob("*.txt"))
        for fpath in txt_files:
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                console.print(f"[dim]Could not read {fpath}: {e}[/dim]")
                continue

            if not text.strip():
                continue

            source_tool = _infer_source_tool(fpath.name)
            section_title = _extract_section_title(text, fpath.name)
            sub_chunks = _chunk_content(text, section_title)

            for i, sub_text in enumerate(sub_chunks):
                chunk_id = hashlib.md5(sub_text.encode()).hexdigest()[:16]
                if len(sub_chunks) > 1:
                    chunk_id = f"{chunk_id}_{i}"
                embedding_text = (
                    f"EDA Documentation: {source_tool}\n"
                    f"Section: {section_title}\n"
                    f"{sub_text}"
                )
                chunk = DocChunk(
                    chunk_id=chunk_id,
                    chunk_type="eda_doc",
                    text=sub_text,
                    source_file=fpath.name,
                    source_tool=source_tool,
                    section_title=section_title,
                    embedding_text=embedding_text,
                )
                chunks.append(chunk)
                source_counts[source_tool] += 1

        table = Table(title="Doc Chunker Stats")
        table.add_column("Source", style="cyan")
        table.add_column("Chunks", style="green")
        for tool, count in source_counts.most_common():
            table.add_row(tool, str(count))
        table.add_row("Total", str(len(chunks)))
        console.print(table)

        return chunks
