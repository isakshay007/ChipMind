"""Tests for ChipMindState and helpers."""

import pytest

from chipmind.agents.state import (
    ChipMindState,
    create_initial_state,
    get_error_summary,
)


def test_create_initial_state_returns_correct_defaults():
    """Test create_initial_state returns correct defaults."""
    state = create_initial_state("Create a 4-bit adder")
    assert state["user_query"] == "Create a 4-bit adder"
    assert state["spec"] == {}
    assert state["retrieved_modules"] == []
    assert state["retrieved_docs"] == []
    assert state["generated_code"] == ""
    assert state["generated_testbench"] == ""
    assert state["compile_result"] == {}
    assert state["sim_result"] == {}
    assert state["is_compiled"] is False
    assert state["is_functionally_correct"] is False
    assert state["errors"] == []
    assert state["error_classifications"] == []
    assert state["debug_context"] == []
    assert state["iteration"] == 0
    assert state["max_iterations"] == 5
    assert state["iteration_history"] == []
    assert state["final_code"] == ""
    assert state["final_status"] == ""
    assert state["total_tokens_used"] == 0
    assert state["total_time_seconds"] == 0.0


def test_create_initial_state_custom_max_iterations():
    """Test create_initial_state with custom max_iterations."""
    state = create_initial_state("Query", max_iterations=10)
    assert state["max_iterations"] == 10


def test_get_error_summary_no_errors():
    """Test get_error_summary with no errors."""
    state: ChipMindState = {"errors": []}
    assert get_error_summary(state) == "No errors"


def test_get_error_summary_empty_state():
    """Test get_error_summary when errors key is missing."""
    state: ChipMindState = {}
    assert get_error_summary(state) == "No errors"


def test_get_error_summary_with_errors():
    """Test get_error_summary with sample errors."""
    state: ChipMindState = {
        "errors": [
            {"line": 5, "error_type": "syntax", "message": "syntax error"},
            {"line": 10, "error_type": "undeclared", "message": "Unable to bind wire 'xyz'"},
        ]
    }
    summary = get_error_summary(state)
    assert "Line 5: [syntax] syntax error" in summary
    assert "Line 10: [undeclared] Unable to bind wire 'xyz'" in summary


def test_get_error_summary_partial_error_dict():
    """Test get_error_summary with errors missing some fields."""
    state: ChipMindState = {
        "errors": [
            {"message": "Unknown error"},
        ]
    }
    summary = get_error_summary(state)
    assert "Line ?: [unknown] Unknown error" in summary
