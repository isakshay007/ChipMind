"""Tests for SpecAnalyzerAgent."""

import json
import pytest

from chipmind.agents.state import create_initial_state


@pytest.fixture
def agent():
    """Create SpecAnalyzerAgent, skip if GROQ_API_KEY not set."""
    try:
        from chipmind.agents.spec_analyzer import SpecAnalyzerAgent
        return SpecAnalyzerAgent()
    except RuntimeError as e:
        if "GROQ_API_KEY" in str(e):
            pytest.skip("GROQ_API_KEY not set in .env")
        raise


def test_alu_spec(agent):
    """Test: 4-bit ALU."""
    state = create_initial_state(
        "Design a 4-bit ALU that supports add, subtract, AND, OR operations"
    )
    result = agent.analyze(state)
    spec = result["spec"]

    assert "module_name" in spec
    assert "alu" in spec["module_name"].lower()
    assert len(spec["inputs"]) >= 2
    assert len(spec["outputs"]) >= 1
    assert spec["complexity_hint"] in ["combinational", "sequential", "fsm"]
    print(f"\nALU Spec: {json.dumps(spec, indent=2)}")


def test_counter_spec(agent):
    """Test: 8-bit counter."""
    state = create_initial_state(
        "Create an 8-bit up counter with synchronous reset and enable"
    )
    result = agent.analyze(state)
    spec = result["spec"]

    assert spec["complexity_hint"] in ["sequential", "fsm"]
    input_names = [i["name"] for i in spec["inputs"]]
    assert "clk" in input_names or "clock" in input_names
    print(f"\nCounter Spec: {json.dumps(spec, indent=2)}")


def test_vague_query(agent):
    """Test: vague description should still produce valid spec."""
    state = create_initial_state("make a counter")
    result = agent.analyze(state)
    spec = result["spec"]

    assert "module_name" in spec
    assert len(spec["inputs"]) > 0
    assert len(spec["outputs"]) > 0
    print(f"\nVague Spec: {json.dumps(spec, indent=2)}")
