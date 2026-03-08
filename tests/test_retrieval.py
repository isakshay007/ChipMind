"""Tests for the hybrid retrieval system."""

import pytest
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "indexes"


@pytest.fixture(scope="module")
def retriever():
    """Load hybrid retriever if index exists."""
    if not INDEX_DIR.exists():
        pytest.skip("Index not built. Run 'make build-index' first.")
    from chipmind.retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever.load(str(INDEX_DIR))


def _print_results(results: list, title: str, top_n: int = 3):
    """Helper to print results for verification."""
    from rich.console import Console
    from rich.table import Table
    console = Console()
    console.print(f"\n[bold]{title}[/bold]")
    table = Table()
    table.add_column("Rank", style="dim")
    table.add_column("Name/Title", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Source", style="yellow")
    for i, r in enumerate(results[:top_n], 1):
        name = (r.get("module_name") or r.get("section_title") or r.get("source_file") or "")[:50] or r.get("chunk_id", "?")
        score = r.get("rrf_score") or r.get("score", 0)
        src = []
        if r.get("semantic_rank"):
            src.append(f"sem#{r['semantic_rank']}")
        if r.get("keyword_rank"):
            src.append(f"kw#{r['keyword_rank']}")
        table.add_row(str(i), str(name)[:50], f"{score:.4f}", ", ".join(src) or "-")
    console.print(table)


def test_1_natural_language_alu(retriever):
    """Test 1: Natural language - 4-bit ALU with addition and subtraction."""
    query = "4-bit ALU with addition and subtraction"
    results = retriever.search_code(query, k=5)
    _print_results(results, "Test 1: Natural language (ALU)", top_n=3)
    assert len(results) > 0
    # Should return ALU-related modules
    names = [r.get("module_name", "").lower() for r in results]
    assert any("alu" in n or "add" in n for n in names)


def test_2_exact_verilog_syntax(retriever):
    """Test 2: Exact Verilog syntax - BM25 should find posedge/clk."""
    query = "always @(posedge clk or posedge reset)"
    results = retriever.search(query, k=5)
    _print_results(results, "Test 2: Exact Verilog syntax", top_n=3)
    assert len(results) > 0
    # For exact syntax, BM25 should contribute; keyword_rank populated when in keyword top-20
    has_kw = any(r.get("keyword_rank") for r in results)
    assert has_kw, "Expected keyword_rank populated for exact syntax (BM25 match)"


def test_3_concept_fsm(retriever):
    """Test 3: Concept query - finite state machine."""
    query = "finite state machine with three states"
    results = retriever.search_code(query, k=5)
    _print_results(results, "Test 3: Concept (FSM)", top_n=3)
    assert len(results) > 0
    # Should return FSM modules
    tags = [r.get("tags", []) for r in results]
    names = [r.get("module_name", "").lower() for r in results]
    assert any("fsm" in str(t).lower() or "state" in n for t, n in zip(tags, names))


def test_4_doc_query(retriever):
    """Test 4: Doc query - synthesis in chip design."""
    query = "What is synthesis in chip design"
    results = retriever.search_docs(query, k=5)
    _print_results(results, "Test 4: Doc query (synthesis)", top_n=3)
    assert len(results) > 0
    # Should return eda_doc chunks
    assert all(r.get("chunk_type") == "eda_doc" for r in results)


def test_5_compare_retrievers(retriever):
    """Test 5: Compare semantic-only vs keyword-only vs hybrid."""
    query = "counter with reset"
    semantic = retriever.semantic.search(query, k=5)
    keyword = retriever.keyword.search(query, k=5)
    hybrid = retriever.search(query, k=5)
    _print_results(
        [{"module_name": r.get("module_name"), "score": r.get("score"), "semantic_rank": 1} for r in semantic],
        "Semantic-only (top 3)",
    )
    _print_results(
        [{"module_name": r.get("module_name"), "score": r.get("score"), "keyword_rank": 1} for r in keyword],
        "Keyword-only (top 3)",
    )
    _print_results(hybrid, "Hybrid (combined) (top 3)")
    assert len(hybrid) > 0
    assert len(semantic) > 0
    assert len(keyword) > 0
