"""End-to-end tests using ChipMindGraph.run()."""

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "indexes"


@pytest.fixture(scope="module")
def graph():
    """Create ChipMindGraph. Skip if index or deps missing."""
    if not INDEX_DIR.exists():
        pytest.skip("Index not built. Run 'make build-index' first.")
    try:
        from chipmind.agents.graph import ChipMindGraph
        return ChipMindGraph(str(INDEX_DIR))
    except RuntimeError as e:
        if "GROQ_API_KEY" in str(e):
            pytest.skip("GROQ_API_KEY not set in .env")
        if "iverilog" in str(e).lower():
            pytest.skip("iverilog not installed")
        raise
    except FileNotFoundError as e:
        if "Index directory" in str(e) or "index" in str(e).lower():
            pytest.skip("Index not built. Run 'make build-index' first.")
        raise


def test_simple_design(graph):
    """Test 1: Simple design - half adder."""
    result = graph.run(
        "Design a half adder with inputs a and b, outputs sum and carry"
    )
    print(f"\n--- Simple Design (Half Adder) ---")
    print(f"Status: {result.get('final_status')}")
    print(f"Iterations: {result.get('iteration', 0)}")
    print(f"Final code (first 500 chars):\n{result.get('final_code', '')[:500]}")

    assert result.get("final_status") in ["success", "compile_success_sim_fail"]
    assert "module" in result.get("final_code", "")
    assert "endmodule" in result.get("final_code", "")


def test_medium_design(graph):
    """Test 2: Medium design - 4-bit counter (may need debug loop)."""
    result = graph.run(
        "Design a 4-bit up counter with synchronous reset and enable"
    )
    print(f"\n--- Medium Design (4-bit Counter) ---")
    print(f"Status: {result.get('final_status')}")
    print(f"Iterations: {result.get('iteration', 0)}")
    history = result.get("iteration_history", [])
    for i, entry in enumerate(history):
        err_count = len(entry.get("errors", []))
        compiled = entry.get("compiled", False)
        print(f"  Iteration {i}: compiled={compiled}, errors={err_count}")
    if history:
        last = history[-1]
        if last.get("errors"):
            print(f"  Last errors: {[e.get('message', '')[:50] for e in last['errors'][:3]]}")

    assert result.get("final_status") in [
        "success",
        "compile_success_sim_fail",
        "max_iterations_reached",
    ]
    assert "module" in result.get("final_code", "")


def test_max_iterations(graph):
    """Test 3: Complex design with low max_iterations."""
    result = graph.run(
        "Design a complex pipelined RISC-V processor",
        max_iterations=2,
    )
    print(f"\n--- Max Iterations Test ---")
    print(f"Status: {result.get('final_status')}")
    print(f"Iterations used: {result.get('iteration', 0)}")
    print(f"Max iterations: {result.get('max_iterations', 5)}")

    assert result.get("final_status") in [
        "success",
        "compile_success_sim_fail",
        "max_iterations_reached",
    ]
