"""Shared state for the ChipMind LangGraph agent pipeline."""

from typing import TypedDict


class ChipMindState(TypedDict, total=False):
    """State flowing through the entire LangGraph agent pipeline."""

    # --- User Input ---
    user_query: str  # Original natural language request

    # --- Spec Analyzer Output ---
    spec: dict  # Structured hardware specification:
    # {
    #   "module_name": str,
    #   "description": str,
    #   "inputs": [{"name": str, "width": int, "description": str}],
    #   "outputs": [{"name": str, "width": int, "description": str}],
    #   "functionality": str,
    #   "complexity_hint": str,  # "combinational" | "sequential" | "fsm"
    #   "constraints": list[str]
    # }

    # --- RAG Retrieval Output ---
    retrieved_modules: list[dict]  # Retrieved Verilog examples
    retrieved_docs: list[dict]  # Retrieved EDA docs

    # --- Code Generation Output ---
    generated_code: str  # Current Verilog code
    generated_testbench: str  # Auto-generated testbench

    # --- Compiler Results ---
    compile_result: dict  # From CompilerGate
    sim_result: dict  # Simulation result
    is_compiled: bool  # Did it compile?
    is_functionally_correct: bool  # Did simulation pass?

    # --- Debug Loop ---
    errors: list[dict]  # Current errors
    error_classifications: list[dict]  # Classified errors
    debug_context: list[dict]  # Retrieved bug-fix examples
    iteration: int  # Current debug iteration (starts at 0)
    max_iterations: int  # Max allowed (default 5)
    iteration_history: list[dict]  # Log of each attempt:
    # [{"iteration": 0, "code": str, "compiled": bool, "errors": [...],
    #   "action": "initial" | "debug_fix"}, ...]

    # --- Final Output ---
    final_code: str  # Final Verilog (best attempt)
    final_status: str  # "success" | "compile_success_sim_fail" |
    # "max_iterations_reached" | "error"
    total_tokens_used: int
    total_time_seconds: float


def create_initial_state(user_query: str, max_iterations: int = 5) -> ChipMindState:
    """Create initial state for a new ChipMind run."""
    return {
        "user_query": user_query,
        "spec": {},
        "retrieved_modules": [],
        "retrieved_docs": [],
        "generated_code": "",
        "generated_testbench": "",
        "compile_result": {},
        "sim_result": {},
        "is_compiled": False,
        "is_functionally_correct": False,
        "errors": [],
        "error_classifications": [],
        "debug_context": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "iteration_history": [],
        "final_code": "",
        "final_status": "",
        "total_tokens_used": 0,
        "total_time_seconds": 0.0,
    }


def get_error_summary(state: ChipMindState) -> str:
    """Get a human-readable summary of current errors."""
    if not state.get("errors"):
        return "No errors"
    lines = []
    for e in state["errors"]:
        lines.append(
            f"Line {e.get('line', '?')}: [{e.get('error_type', 'unknown')}] {e.get('message', '')}"
        )
    return "\n".join(lines)
