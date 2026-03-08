"""LangGraph orchestrator for the ChipMind pipeline."""

import time
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from chipmind.agents.code_generator import CodeGeneratorAgent
from chipmind.agents.compiler_gate import CompilerGate
from chipmind.agents.error_classifier import ErrorClassifier
from chipmind.agents.state import ChipMindState, create_initial_state
from chipmind.agents.spec_analyzer import SpecAnalyzerAgent
from chipmind.agents.testbench_generator import TestbenchGeneratorAgent
from chipmind.retrieval.hybrid_retriever import HybridRetriever


class ChipMindGraph:
    """LangGraph state machine orchestrating the full ChipMind pipeline."""

    def __init__(self, index_dir: str = "data/processed/indexes"):
        path = Path(index_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_dir}. Run 'make build-index' first."
            )
        self.retriever = HybridRetriever.load(str(path))
        self.spec_agent = SpecAnalyzerAgent()
        self.code_agent = CodeGeneratorAgent(retriever=self.retriever)
        self.tb_agent = TestbenchGeneratorAgent()
        self.error_cls = ErrorClassifier()
        self.compiler = CompilerGate()
        self.graph = self._build_graph()

    def run(self, query: str, max_iterations: int = 5) -> dict:
        """Run the full pipeline. Returns the final state dict."""
        initial = create_initial_state(query, max_iterations)
        start = time.time()
        result = self.graph.invoke(initial)
        result["total_time_seconds"] = time.time() - start
        return result

    def _build_graph(self):
        """Build the LangGraph state machine."""
        graph = StateGraph(ChipMindState)

        graph.add_node("spec_analyzer", self._spec_node)
        graph.add_node("code_generator", self._code_gen_node)
        graph.add_node("testbench_generator", self._tb_node)
        graph.add_node("compiler_gate", self._compile_node)
        graph.add_node("error_classifier", self._error_node)
        graph.add_node("debug_fix", self._debug_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "spec_analyzer")
        graph.add_edge("spec_analyzer", "code_generator")
        graph.add_edge("code_generator", "testbench_generator")
        graph.add_edge("testbench_generator", "compiler_gate")

        graph.add_conditional_edges(
            "compiler_gate",
            self._should_continue,
            {"success": "finalize", "retry": "error_classifier", "give_up": "finalize"},
        )

        graph.add_edge("error_classifier", "debug_fix")
        graph.add_edge("debug_fix", "compiler_gate")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _spec_node(self, state: ChipMindState) -> dict:
        """Parse NL query → structured spec."""
        return self.spec_agent.analyze(state)

    def _code_gen_node(self, state: ChipMindState) -> dict:
        """Generate fresh Verilog code (only called once at start)."""
        return self.code_agent.generate(state)

    def _tb_node(self, state: ChipMindState) -> dict:
        """Generate testbench."""
        return self.tb_agent.generate(state)

    def _compile_node(self, state: ChipMindState) -> dict:
        """Compile and simulate. Store results in state."""
        code = state.get("generated_code", "")
        tb = state.get("generated_testbench", "")

        if not code:
            return {
                "compile_result": {
                    "success": False,
                    "errors": [{"message": "No code to compile", "line": 0, "error_type": "other", "file": "", "raw": ""}],
                },
                "is_compiled": False,
                "is_functionally_correct": False,
            }

        if tb:
            try:
                sim_result = self.compiler.compile_and_simulate(code, tb)
            except Exception as e:
                errors_list = [
                    {
                        "file": "",
                        "line": 0,
                        "error_type": "other",
                        "message": str(e),
                        "raw": str(e),
                    }
                ]
                return {
                    "compile_result": {"success": False, "errors": errors_list},
                    "is_compiled": False,
                    "is_functionally_correct": False,
                    "errors": errors_list,
                    "iteration_history": state.get("iteration_history", [])
                    + [
                        {
                            "iteration": state.get("iteration", 0),
                            "code": code,
                            "compiled": False,
                            "errors": errors_list,
                        }
                    ],
                }
            errors_list = [
                {
                    "file": e.file,
                    "line": e.line,
                    "error_type": e.error_type,
                    "message": e.message,
                    "raw": e.raw,
                }
                for e in sim_result.compile_errors
            ]

            history_entry = {
                "iteration": state.get("iteration", 0),
                "code": code,
                "compiled": sim_result.compiled,
                "simulated": sim_result.simulated,
                "passed": sim_result.passed,
                "errors": errors_list,
                "sim_output": sim_result.sim_output if sim_result.simulated else "",
            }
            iteration_history = list(state.get("iteration_history", []))
            iteration_history.append(history_entry)

            # When compiled but sim failed, pass sim output as "errors" for debug loop
            errors_for_debug = errors_list
            if sim_result.compiled and not sim_result.passed and sim_result.sim_output:
                errors_for_debug = [
                    {
                        "line": 0,
                        "error_type": "simulation_fail",
                        "message": f"Simulation failed. Output: {sim_result.sim_output[:500]}",
                        "file": "",
                        "raw": sim_result.sim_output[:300],
                    }
                ]

            return {
                "compile_result": {"success": sim_result.compiled, "errors": errors_list},
                "sim_result": {
                    "simulated": sim_result.simulated,
                    "output": sim_result.sim_output,
                    "passed": sim_result.passed,
                },
                "is_compiled": sim_result.compiled,
                "is_functionally_correct": sim_result.compiled and sim_result.passed,
                "errors": errors_for_debug if not (sim_result.compiled and sim_result.passed) else [],
                "iteration_history": iteration_history,
            }
        else:
            try:
                compile_result = self.compiler.compile(code)
            except Exception as e:
                errors_list = [
                    {
                        "file": "",
                        "line": 0,
                        "error_type": "other",
                        "message": str(e),
                        "raw": str(e),
                    }
                ]
                return {
                    "compile_result": {"success": False, "errors": errors_list},
                    "is_compiled": False,
                    "is_functionally_correct": False,
                    "errors": errors_list,
                    "iteration_history": state.get("iteration_history", [])
                    + [
                        {
                            "iteration": state.get("iteration", 0),
                            "code": code,
                            "compiled": False,
                            "errors": errors_list,
                        }
                    ],
                }
            errors_list = [
                {
                    "file": e.file,
                    "line": e.line,
                    "error_type": e.error_type,
                    "message": e.message,
                    "raw": e.raw,
                }
                for e in compile_result.errors
            ]

            history_entry = {
                "iteration": state.get("iteration", 0),
                "code": code,
                "compiled": compile_result.success,
                "errors": errors_list,
            }
            iteration_history = list(state.get("iteration_history", []))
            iteration_history.append(history_entry)

            return {
                "compile_result": {"success": compile_result.success, "errors": errors_list},
                "is_compiled": compile_result.success,
                "is_functionally_correct": False,
                "errors": errors_list if not compile_result.success else [],
                "iteration_history": iteration_history,
            }

    def _error_node(self, state: ChipMindState) -> dict:
        """Classify errors."""
        return self.error_cls.classify(state)

    def _debug_node(self, state: ChipMindState) -> dict:
        """Fix code using compiler feedback + RAG."""
        updates = self.code_agent.debug_fix(state)
        updates["iteration"] = state.get("iteration", 0) + 1
        return updates

    def _should_continue(self, state: ChipMindState) -> str:
        """Routing function for conditional edge after compile."""
        compiled = state.get("is_compiled", False)
        correct = state.get("is_functionally_correct", False)
        has_tb = bool(state.get("generated_testbench"))
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 5)

        # Success ONLY when: compiled AND (functionally correct OR no testbench)
        if compiled and (correct or not has_tb):
            return "success"
        # Retry when: compile failed OR (compiled but sim failed)
        if iteration >= max_iter:
            return "give_up"
        return "retry"

    def _finalize_node(self, state: ChipMindState) -> dict:
        """Set final output fields."""
        code = state.get("generated_code", "")
        compiled = state.get("is_compiled", False)
        correct = state.get("is_functionally_correct", False)

        if compiled and correct:
            status = "success"
        elif compiled:
            status = "compile_success_sim_fail"
        elif state.get("iteration", 0) >= state.get("max_iterations", 5):
            status = "max_iterations_reached"
        else:
            status = "error"

        return {
            "final_code": code,
            "final_status": status,
        }
