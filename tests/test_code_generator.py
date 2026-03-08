"""Tests for CodeGeneratorAgent: spec → retrieve → generate → compile."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "indexes"


@pytest.fixture(scope="module")
def retriever():
    """Load hybrid retriever if index exists."""
    if not INDEX_DIR.exists():
        pytest.skip("Index not built. Run 'make build-index' first.")
    from chipmind.retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever.load(str(INDEX_DIR))


@pytest.fixture(scope="module")
def spec_agent():
    """Create SpecAnalyzerAgent, skip if GROQ_API_KEY not set."""
    try:
        from chipmind.agents.spec_analyzer import SpecAnalyzerAgent
        return SpecAnalyzerAgent()
    except RuntimeError as e:
        if "GROQ_API_KEY" in str(e):
            pytest.skip("GROQ_API_KEY not set in .env")
        raise


@pytest.fixture(scope="module")
def code_agent(retriever):
    """Create CodeGeneratorAgent with retriever."""
    from chipmind.agents.code_generator import CodeGeneratorAgent
    return CodeGeneratorAgent(retriever=retriever)


@pytest.fixture(scope="module")
def compiler():
    """Create CompilerGate, skip if iverilog not found."""
    try:
        from chipmind.agents.compiler_gate import CompilerGate
        return CompilerGate()
    except RuntimeError as e:
        if "iverilog not found" in str(e):
            pytest.skip("iverilog not installed")
        raise


def test_generate_half_adder(spec_agent, code_agent, compiler):
    """Full pipeline: NL → spec → RAG → generate → compile."""
    from chipmind.agents.state import create_initial_state

    state = create_initial_state("Design a half adder with inputs a and b")

    # Spec
    updates = spec_agent.analyze(state)
    state.update(updates)
    print(f"\nSpec: {json.dumps(state['spec'], indent=2)}")

    # Generate with RAG
    updates = code_agent.generate(state)
    state.update(updates)
    print(f"\nGenerated code:\n{state['generated_code']}")
    print(f"\nRetrieved {len(state['retrieved_modules'])} reference modules")

    # Compile
    result = compiler.compile(state["generated_code"])
    print(f"\nCompiled: {result.success}")
    if not result.success:
        for e in result.errors:
            print(f"  Error: {e.message}")

    assert state["generated_code"] != ""
    assert "module" in state["generated_code"]
    assert "endmodule" in state["generated_code"]


def test_generate_counter(spec_agent, code_agent, compiler):
    """Test sequential design generation."""
    from chipmind.agents.state import create_initial_state

    state = create_initial_state("Create a 4-bit up counter with synchronous reset")

    updates = spec_agent.analyze(state)
    state.update(updates)

    updates = code_agent.generate(state)
    state.update(updates)
    print(f"\nCounter code:\n{state['generated_code']}")

    result = compiler.compile(state["generated_code"])
    print(f"Compiled: {result.success}")

    assert "module" in state["generated_code"]
    assert "always" in state["generated_code"]


def test_rag_vs_no_rag(spec_agent, code_agent, compiler):
    """Compare: does RAG context improve compilation success?"""
    from chipmind.agents.state import create_initial_state

    query = "Design a 4-bit ALU with add, subtract, AND, OR"
    state = create_initial_state(query)
    updates = spec_agent.analyze(state)
    state.update(updates)

    # With RAG
    updates = code_agent.generate(state)
    state.update(updates)
    rag_result = compiler.compile(state["generated_code"])

    print(f"\nWith RAG - Compiled: {rag_result.success}")
    print(f"Retrieved modules: {[m.get('module_name') for m in state['retrieved_modules'][:3]]}")
    print(f"Code:\n{state['generated_code'][:300]}...")


# --- Unit tests (no retriever/LLM required) ---


@pytest.fixture
def code_agent_unit():
    """CodeGeneratorAgent with mocked retriever and settings (for unit tests)."""
    mock_retriever = MagicMock()
    mock_retriever.search_code.return_value = []
    with patch("chipmind.agents.code_generator.settings", MagicMock(GROQ_API_KEY="test-key")):
        from chipmind.agents.code_generator import CodeGeneratorAgent
        return CodeGeneratorAgent(retriever=mock_retriever)


def test_clean_verilog_strips_markdown(code_agent_unit):
    """_clean_verilog strips ```verilog blocks."""
    raw = '```verilog\nmodule foo(input a, output b);\n  assign b = a;\nendmodule\n```'
    out = code_agent_unit._clean_verilog(raw)
    assert "module" in out
    assert "endmodule" in out
    assert "```" not in out


def test_clean_verilog_extracts_module_block(code_agent_unit):
    """_clean_verilog extracts module...endmodule from extra text."""
    raw = "Here is the code:\n\nmodule foo(input a, output b);\n  assign b = a;\nendmodule\n\nDone."
    out = code_agent_unit._clean_verilog(raw)
    assert out.startswith("module")
    assert out.endswith("endmodule")


def test_build_retrieval_query(code_agent_unit):
    """_build_retrieval_query combines spec fields."""
    spec = {
        "description": "4-bit ALU",
        "complexity_hint": "combinational",
        "inputs": [{"name": "a"}, {"name": "b"}],
        "outputs": [{"name": "out"}],
        "constraints": ["synthesizable"],
    }
    q = code_agent_unit._build_retrieval_query(spec)
    assert "4-bit ALU" in q
    assert "combinational" in q
    assert "a" in q
    assert "b" in q
    assert "out" in q
    assert "synthesizable" in q


def test_format_examples(code_agent_unit):
    """_format_examples formats modules with truncation."""
    modules = [
        {"module_name": "half_adder", "code": "module half_adder(input a, input b, output sum, output carry);\n  assign sum = a ^ b;\n  assign carry = a & b;\nendmodule"},
        {"module_name": "full_adder", "code": "module full_adder(...);\nendmodule"},
    ]
    out = code_agent_unit._format_examples(modules, max_examples=2)
    assert "Example 1: half_adder" in out
    assert "Example 2: full_adder" in out
    assert "half_adder" in out
    assert "full_adder" in out


def test_format_errors_dict(code_agent_unit):
    """_format_errors handles dict errors."""
    errors = [
        {"line": 5, "message": "missing semicolon", "error_type": "syntax"},
        {"line": 10, "message": "unknown signal", "error_type": "undeclared"},
    ]
    out = code_agent_unit._format_errors(errors)
    assert "Line 5: [syntax] missing semicolon" in out
    assert "Line 10: [undeclared] unknown signal" in out


def test_format_errors_compiler_error(code_agent_unit):
    """_format_errors handles CompilerError objects."""
    from chipmind.agents.compiler_gate import CompilerError
    errors = [CompilerError("file.v", 3, "syntax", "unexpected token", "raw")]
    out = code_agent_unit._format_errors(errors)
    assert "Line 3: [syntax]" in out
    assert "unexpected token" in out
