"""Error classifier: maps compiler errors to actionable fix strategies (no LLM)."""

from chipmind.agents.state import ChipMindState

FIX_STRATEGIES = {
    "syntax": "Check for missing semicolons, parentheses, or begin/end blocks",
    "undeclared": "Declare the signal as wire or reg, or check for typos",
    "width_mismatch": "Adjust signal width to match the port or assignment target",
    "type_error": "Change wire to reg if assigned inside always block, or vice versa",
    "missing_module": "Check module instantiation name and ensure module is defined",
    "simulation_fail": "Review simulation output for FAIL/ERROR; fix logic to match expected behavior",
    "other": "Review the error message and surrounding code context",
}

PRIORITY = {
    "syntax": 1,
    "undeclared": 2,
    "width_mismatch": 2,
    "type_error": 2,
    "missing_module": 2,
    "simulation_fail": 2,
    "other": 3,
}


class ErrorClassifier:
    """Classifies compiler errors into actionable categories (pure logic, no LLM)."""

    def classify(self, state: ChipMindState) -> dict:
        """Classify errors from compile_result into error_classifications."""
        compile_result = state.get("compile_result", {})
        errors = compile_result.get("errors", [])

        if not errors:
            return {"error_classifications": []}

        classifications = []
        for e in errors:
            if isinstance(e, dict):
                line = e.get("line", 0)
                error_type = e.get("error_type", "other")
                message = e.get("message", "")
            else:
                line = getattr(e, "line", 0)
                error_type = getattr(e, "error_type", "other")
                message = getattr(e, "message", "")

            fix_strategy = FIX_STRATEGIES.get(error_type, FIX_STRATEGIES["other"])
            priority = PRIORITY.get(error_type, 3)

            classifications.append(
                {
                    "line": line,
                    "error_type": error_type,
                    "message": message,
                    "fix_strategy": fix_strategy,
                    "priority": priority,
                }
            )

        classifications.sort(key=lambda x: x["priority"])

        return {"error_classifications": classifications}
