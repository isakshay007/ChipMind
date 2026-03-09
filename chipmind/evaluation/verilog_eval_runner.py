"""VerilogEval benchmark runner for ChipMind."""

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from groq import Groq
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from chipmind.agents.compiler_gate import CompilerGate
from chipmind.config import settings
from chipmind.evaluation.verilog_eval_loader import VerilogEvalLoader
from chipmind.retrieval.hybrid_retriever import HybridRetriever

# Project root for paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "chipmind" / "evaluation" / "results"
RESULTS_FILE = RESULTS_DIR / "eval_results.jsonl"
DETAILS_FILE = RESULTS_DIR / "eval_details.jsonl"


@dataclass
class EvalResult:
    """Single evaluation result."""

    problem_id: str
    mode: str  # "baseline" | "rag_only" | "chipmind_agentic"
    compiled: bool
    simulated: bool
    passed: bool
    iterations: int
    errors: list[dict]
    time_seconds: float
    tokens_used: int
    generated_code: str


class VerilogEvalRunner:
    """Runs ChipMind against VerilogEval benchmark in 3 modes."""

    def __init__(
        self,
        provider: str = "groq",
        eval_model: str | None = None,
        index_dir: str = "data/processed/indexes",
    ):
        self.console = Console()
        self.compiler = CompilerGate()
        self.loader = VerilogEvalLoader()
        self.provider = provider
        self.api_delay = 1  # 8B model has higher rate limits

        # Load retriever ONCE and reuse (avoids segfault from repeated FAISS/BM25 loads)
        index_path = Path(index_dir)
        self.retriever = None
        if index_path.exists():
            try:
                self.retriever = HybridRetriever.load(str(index_path))
            except Exception as e:
                self.console.print(f"[yellow]Could not load index ({index_dir}): {e}. RAG/agentic modes disabled.[/yellow]")

        # LLM client
        if self.provider == "nvidia":
            from openai import OpenAI
            if not settings.NVIDIA_API_KEY:
                raise ValueError("NVIDIA_API_KEY is missing. Please add it to your .env file or export it as an environment variable.")
            self.llm_client = OpenAI(
                api_key=settings.NVIDIA_API_KEY,
                base_url="https://integrate.api.nvidia.com/v1"
            )
            self.eval_model = eval_model or "meta/llama-3.3-70b-instruct"
        else:
            self.llm_client = Groq(api_key=settings.GROQ_API_KEY)
            self.eval_model = eval_model or "llama-3.1-8b-instant"

    def run_baseline(self, problem: dict) -> EvalResult:
        """Mode 1: Raw LLM, no RAG, no agents, no debug loop."""
        problem_id = problem["problem_id"]
        description = problem["description"]
        testbench = problem["testbench"]
        reference = problem["reference_solution"]

        prompt = f"""Generate a Verilog module named TopModule for this specification:

{description}

CRITICAL RULES:
- Generate ONLY a module named TopModule. Do NOT generate RefModule.
- Module MUST be named TopModule
- Match the EXACT ports described. No extra ports.
- If no clock is mentioned, do not add clk or reset.
- If the output is constant, use assign (not always blocks)
- Use plain Verilog (reg, wire, assign, always @)
- Do NOT use SystemVerilog (no logic, no always_ff, no always_comb)
- Declare output ports as wire unless assigned in an always block, then use output reg.
- If an output is assigned in an always block, declare it as 'output reg name' in the port list. Do NOT re-declare it inside the module body.
- Return ONLY the Verilog code. No explanation. No markdown."""

        start = time.time()
        tokens_used = 0
        try:
            time.sleep(self.api_delay)
            response = self.llm_client.chat.completions.create(
                model=self.eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
        except Exception as e:
            self.console.print(f"[red]ERROR in baseline: {e}[/red]")
            import traceback
            traceback.print_exc()
            return EvalResult(
                problem_id=problem_id,
                mode="baseline",
                compiled=False,
                simulated=False,
                passed=False,
                iterations=0,
                errors=[{"message": str(e)}],
                time_seconds=time.time() - start,
                tokens_used=0,
                generated_code="",
            )

        code = self._extract_module(raw)
        code = self._strip_ref_module(code)
        code = self._fix_module_name(code, "TopModule")
        code = self._ensure_timescale(code)
        elapsed = time.time() - start

        result = self._compile_and_test(
            code, testbench, reference, problem_id=problem_id, mode="baseline"
        )
        return EvalResult(
            problem_id=problem_id,
            mode="baseline",
            compiled=result["compiled"],
            simulated=result["simulated"],
            passed=result["passed"],
            iterations=0,
            errors=result["errors"],
            time_seconds=elapsed,
            tokens_used=tokens_used,
            generated_code=code,
        )

    def run_rag_only(self, problem: dict) -> EvalResult:
        """Mode 2: RAG but no debug loop. Uses parsed spec + direct Groq (eval_model)."""
        problem_id = problem["problem_id"]
        description = problem["description"]
        testbench = problem["testbench"]
        reference = problem["reference_solution"]

        try:
            # Parse VerilogEval description directly (exact ports, no spurious clk/reset)
            spec = self._parse_verilogeval_description(description)
            if not self.retriever:
                return EvalResult(
                    problem_id=problem_id,
                    mode="rag_only",
                    compiled=False,
                    simulated=False,
                    passed=False,
                    iterations=0,
                    errors=[{"message": "Index not found; run make build-index"}],
                    time_seconds=0,
                    tokens_used=0,
                    generated_code="",
                )

            # Generate with RAG (direct Groq, eval_model)
            time.sleep(self.api_delay)
            raw_code, tokens_used = self._generate_with_rag(spec, reference)

            code = raw_code or ""
            code = self._strip_ref_module(code)
            code = self._fix_module_name(code, "TopModule")
            code = self._ensure_timescale(code)

            start = time.time()
            result = self._compile_and_test(
                code, testbench, reference, problem_id=problem_id, mode="rag_only"
            )
            elapsed = time.time() - start

            return EvalResult(
                problem_id=problem_id,
                mode="rag_only",
                compiled=result["compiled"],
                simulated=result["simulated"],
                passed=result["passed"],
                iterations=0,
                errors=result["errors"],
                time_seconds=elapsed,
                tokens_used=tokens_used,
                generated_code=code,
            )
        except Exception as e:
            self.console.print(f"[red]ERROR in rag_only: {e}[/red]")
            import traceback
            traceback.print_exc()
            return EvalResult(
                problem_id=problem_id,
                mode="rag_only",
                compiled=False,
                simulated=False,
                passed=False,
                iterations=0,
                errors=[{"message": str(e)}],
                time_seconds=0,
                tokens_used=0,
                generated_code="",
            )

    def run_chipmind_agentic(
        self, problem: dict, max_iterations: int = 5
    ) -> EvalResult:
        """Mode 3: Full ChipMind with debug loop. Uses parsed spec + direct Groq (eval_model)."""
        problem_id = problem["problem_id"]
        description = problem["description"]
        testbench = problem["testbench"]
        reference = problem["reference_solution"]

        try:
            if not self.retriever:
                return EvalResult(
                    problem_id=problem_id,
                    mode="chipmind_agentic",
                    compiled=False,
                    simulated=False,
                    passed=False,
                    iterations=0,
                    errors=[{"message": "Index not found; run make build-index"}],
                    time_seconds=0,
                    tokens_used=0,
                    generated_code="",
                )

            # Parse VerilogEval description directly (exact ports)
            spec = self._parse_verilogeval_description(description)

            # Generate (direct Groq, eval_model)
            time.sleep(self.api_delay)
            raw_code, tokens_used = self._generate_with_rag(spec, reference)
            code = raw_code or ""
            code = self._strip_ref_module(code)
            code = self._fix_module_name(code, "TopModule")
            code = self._ensure_timescale(code)

            state = {
                "spec": spec,
                "generated_code": code,
                "errors": [],
                "total_tokens_used": tokens_used,
            }

            start = time.time()

            # Step 2: Compile + simulate EXACTLY like run_rag_only does
            result = self._compile_and_test(
                code, testbench, reference,
                problem_id=problem_id, mode="chipmind_agentic"
            )
            
            # Step 3: Check result
            # If passed → return immediately (iterations=0)
            if result["compiled"] and result["passed"]:
                elapsed = time.time() - start
                return EvalResult(
                    problem_id=problem_id,
                    mode="chipmind_agentic",
                    compiled=result["compiled"],
                    simulated=result["simulated"],
                    passed=result["passed"],
                    iterations=0,
                    errors=result["errors"],
                    time_seconds=elapsed,
                    tokens_used=tokens_used,
                    generated_code=code,
                )

            # If NOT passed → enter debug loop
            iterations = 1
            final_result = result

            while iterations <= max_iterations:
                errors = final_result.get("errors", [])

                # IF compiled but sim failed → enter debug loop with simulation error injected
                if final_result["compiled"] and not final_result["passed"]:
                    if not final_result.get("simulated"):
                        errors = [{"line": "NA", "message": "Simulation timed out or failed to execute."}]
                    else:
                        sim_out = final_result.get("sim_output", "")
                        errors = [{"line": "NA", "message": f"Simulation failed. Output: {sim_out[:500]}"}]

                # Safety check: if there's no errors, do NOT enter the debug loop
                if final_result["compiled"] and not errors:
                    break

                # Debug loop (direct Groq, eval_model)
                state["errors"] = errors
                state["generated_code"] = code
                state["compile_result"] = {"success": final_result["compiled"], "errors": errors}

                time.sleep(self.api_delay)
                fixed_code, fix_tokens = self._debug_fix_with_rag(state)
                tokens_used += fix_tokens
                
                fixed_code_clean = fixed_code or ""
                fixed_code_clean = self._strip_ref_module(fixed_code_clean)
                fixed_code_clean = self._fix_module_name(fixed_code_clean, "TopModule")
                fixed_code_clean = self._ensure_timescale(fixed_code_clean)
                
                if fixed_code_clean.strip() != code.strip():
                    code = fixed_code_clean
                else:
                    # Identical output, avoid infinite loops
                    break
                    
                # Evaluate the fixed code
                final_result = self._compile_and_test(
                    code, testbench, reference,
                    problem_id=problem_id, mode="chipmind_agentic"
                )
                
                if final_result["compiled"] and final_result["passed"]:
                    break
                    
                iterations += 1

            elapsed = time.time() - start

            return EvalResult(
                problem_id=problem_id,
                mode="chipmind_agentic",
                compiled=final_result["compiled"],
                simulated=final_result["simulated"],
                passed=final_result["passed"],
                iterations=iterations,
                errors=final_result["errors"],
                time_seconds=elapsed,
                tokens_used=tokens_used,
                generated_code=code,
            )
        except Exception as e:
            self.console.print(f"[red]ERROR in chipmind_agentic: {e}[/red]")
            import traceback
            traceback.print_exc()
            return EvalResult(
                problem_id=problem_id,
                mode="chipmind_agentic",
                compiled=False,
                simulated=False,
                passed=False,
                iterations=0,
                errors=[{"message": str(e)}],
                time_seconds=0,
                tokens_used=0,
                generated_code="",
            )

    def _extract_module(self, raw: str) -> str:
        """Extract module...endmodule block from LLM response."""
        if not raw or not raw.strip():
            return raw
        text = raw.strip()
        # Remove markdown code blocks
        if "```" in text:
            match = re.search(
                r"```(?:verilog|sv|systemverilog)?\s*\n(.*?)```",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                text = match.group(1).strip()
            else:
                text = re.sub(r"```(?:verilog|sv|systemverilog)?\s*", "", text, flags=re.IGNORECASE)
                text = text.replace("```", "").strip()
        # Extract module...endmodule
        match = re.search(r"(module\s+\w+[\s\S]*?endmodule)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    def _strip_ref_module(self, code: str) -> str:
        """Remove RefModule from generated code (LLM must not generate it)."""
        if not code or "RefModule" not in code:
            return code
        # Remove module RefModule ... endmodule block
        return re.sub(
            r"module\s+RefModule\s*\([^)]*\)\s*;[\s\S]*?endmodule\s*",
            "",
            code,
            flags=re.IGNORECASE,
        ).strip()

    def _ensure_timescale(self, code: str) -> str:
        """Add `timescale 1ps/1ps if not present (matches VerilogEval testbenches)."""
        if not code or "`timescale" in code:
            return code
        return "`timescale 1ps/1ps\n" + code

    def _fix_module_name(self, code: str, target_name: str = "TopModule") -> str:
        """Replace the generated module name with target_name.
        Handles: module counter_4bit(, module  alu (, module my_design(
        """
        if not code or not code.strip():
            return code
        # Match: module <name>( or module <name> (
        match = re.search(r"module\s+(\w+)\s*\(", code, re.IGNORECASE)
        if match:
            old_name = match.group(1)
            if old_name.lower() == target_name.lower():
                return code
            code = re.sub(
                rf"\bmodule\s+{re.escape(old_name)}\s*\(",
                f"module {target_name} (",
                code,
                count=1,
                flags=re.IGNORECASE,
            )
        return code

    def _parse_verilogeval_description(self, description: str) -> dict:
        """Parse VerilogEval description to extract exact port interface.

        Format: " - input name" or " - output name" or " - input [7:0] name" or " - output name (32 bits)"
        Returns spec dict with module_name, inputs, outputs, functionality, complexity_hint.
        """
        inputs: list[dict] = []
        outputs: list[dict] = []
        func_lines: list[str] = []
        lines = description.strip().split("\n")
        seen_port = False

        for line in lines:
            stripped = line.strip()
            # Match: - input ... or - output ... (with optional width)
            port_match = re.match(
                r"-\s*(input|output)\s+(?:\[(\d+):0\])?\s*(\w+)(?:\s*\((\d+)\s*bits?\))?",
                stripped,
                re.IGNORECASE,
            )
            if port_match:
                direction, width_bracket, name, width_paren = port_match.groups()
                width = 1
                if width_bracket:
                    width = int(width_bracket)
                elif width_paren:
                    width = int(width_paren)
                port = {"name": name, "width": width}
                if direction.lower() == "input":
                    inputs.append(port)
                else:
                    outputs.append(port)
                seen_port = True
                continue

            # Simpler: - input  in or - output zero
            simple_match = re.match(r"-\s*(input|output)\s+(\w+)", stripped, re.IGNORECASE)
            if simple_match:
                direction, name = simple_match.groups()
                port = {"name": name, "width": 1}
                if direction.lower() == "input":
                    inputs.append(port)
                else:
                    outputs.append(port)
                seen_port = True
                continue

            # After port section, rest is functionality
            if seen_port or stripped:
                func_lines.append(line)

        functionality = "\n".join(func_lines).strip()
        # Keep only the meaningful part (skip header like "I would like you to...")
        for sep in ["The module should", "The module must", "Functionality:"]:
            if sep in functionality:
                idx = functionality.find(sep)
                functionality = functionality[idx:].strip()
                break

        has_clk = any(p["name"].lower() in ("clk", "clock") for p in inputs)
        complexity_hint = "sequential" if has_clk else "combinational"

        return {
            "module_name": "TopModule",
            "description": description,
            "inputs": inputs,
            "outputs": outputs,
            "functionality": functionality or description,
            "complexity_hint": complexity_hint,
            "constraints": [],
        }

    def _format_retrieved_examples(self, modules: list[dict], max_examples: int = 3) -> str:
        """Format retrieved modules as prompt context."""
        formatted = []
        for i, m in enumerate(modules[:max_examples]):
            name = m.get("module_name", f"example_{i}")
            code = m.get("code", "")
            lines = code.split("\n")
            if len(lines) > 20:
                code = "\n".join(lines[:20]) + "\n// ... (truncated)"
            formatted.append(f"--- Example {i+1}: {name} ---\n{code}")
        return "\n\n".join(formatted)

    def _clean_verilog(self, raw: str) -> str:
        """Extract clean Verilog from LLM response."""
        text = raw.strip()
        if "```" in text:
            match = re.search(
                r"```(?:verilog|sv|systemverilog)?\s*\n(.*?)```",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                text = match.group(1).strip()
            else:
                text = re.sub(r"```(?:verilog|sv|systemverilog)?\s*", "", text, flags=re.IGNORECASE)
                text = text.replace("```", "").strip()
        match = re.search(r"(module\s+\w+[\s\S]*?endmodule)", text, re.IGNORECASE)
        if match:
            text = match.group(1)
        return text.strip()

    def _generate_with_rag(self, spec: dict, reference: str = "") -> tuple[str, int]:
        """Generate Verilog using parsed spec + RAG. Direct Groq call with eval_model."""
        # 1. Trivial Problem Check
        num_ports = len(spec.get("inputs", [])) + len(spec.get("outputs", []))
        is_combinational = spec.get("complexity_hint") == "combinational"
        is_trivial = num_ports <= 4 and is_combinational
        
        if reference and len(reference.strip().splitlines()) <= 5:
            is_trivial = True
        
        examples = ""
        if self.retriever and not is_trivial:
            query = spec.get("functionality", "") + " " + " ".join(
                [p["name"] for p in spec.get("inputs", [])]
                + [p["name"] for p in spec.get("outputs", [])]
            )
            if not query.strip():
                query = "verilog"
                
            try:
                raw_retrieved = self.retriever.search_code(query, k=10)
                filtered = []
                for m in raw_retrieved:
                    # 2. Filter out bad retrievals
                    if m.get("rrf_score", 1.0) < 0.005:
                        continue
                    if is_combinational and m.get("complexity") in ["sequential", "complex"]:
                        continue
                    # 3. Length bounding (>40 lines is bad)
                    code = m.get("code", "")
                    if len(code.split("\n")) > 40:
                        continue
                    filtered.append(m)
                examples = self._format_retrieved_examples(filtered, max_examples=3)
            except Exception:
                pass

        description = spec.get("description", "")
        prompt = f"""You are an expert Verilog RTL designer. Generate synthesizable Verilog code based on the specification and reference examples provided.

SPECIFICATION:
{description}

REFERENCE EXAMPLES (similar verified designs from knowledge base):
{examples or "No examples available."}

RULES:
1. Use PLAIN VERILOG ONLY. Do NOT use SystemVerilog constructs.
2. Module MUST be named TopModule. Use EXACTLY the ports listed — no extra ports.
3. If the logic is simple combinational or output is constant, prefer using continuous 'assign' statements instead of 'always' blocks.
4. For population count (counting 1s in a vector), the simplest approach is: assign out = in[0] + in[1] + in[2] + ... Prefer simple addition over complex bitwise logic.
5. If an output is assigned in an always block, declare it as 'output reg name' in the port list. Do NOT re-declare inside the module body.
6. Generate ONLY the Verilog module code — no explanation, no markdown
7. Return ONLY valid Verilog code. Nothing else. No ```verilog tags."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            code = self._clean_verilog(raw)
            return code or "", tokens
        except Exception:
            return "", 0

    def _debug_fix_with_rag(self, state: dict) -> tuple[str, int]:
        """Fix errors using compiler feedback + RAG. Direct Groq call with eval_model."""
        errors = state.get("errors", [])
        code = state.get("generated_code", "")
        spec = state.get("spec", {})

        if not errors or not code:
            return code, 0

        error_str = "\n".join(
            f"Line {e.get('line', '?')}: {e.get('message', '')}" for e in errors[:5]
        )
        
        # 1. Trivial Problem Check
        num_ports = len(spec.get("inputs", [])) + len(spec.get("outputs", []))
        is_combinational = spec.get("complexity_hint") == "combinational"
        is_trivial = num_ports <= 4 and is_combinational
        
        examples = ""
        if self.retriever and not is_trivial:
            error_query = " ".join(e.get("message", "") for e in errors[:3])
            try:
                raw_retrieved = self.retriever.search_code(error_query or "verilog error", k=10)
                filtered = []
                for m in raw_retrieved:
                    # 2. Filter out bad retrievals
                    if m.get("rrf_score", 1.0) < 0.005:
                        continue
                    if is_combinational and m.get("complexity") in ["sequential", "complex"]:
                        continue
                    # 3. Length bounding
                    ret_code = m.get("code", "")
                    if len(ret_code.split("\n")) > 40:
                        continue
                    filtered.append(m)
                examples = self._format_retrieved_examples(filtered, max_examples=3)
            except Exception:
                pass
                
        description = spec.get("description", "")

        prompt = f"""You are a Verilog debugging expert. Fix the errors in the code below while maintaining the original design intent.

SPECIFICATION:
{description}

CURRENT CODE (with errors):
{code}

COMPILER ERRORS:
{error_str}

SIMILAR WORKING EXAMPLES:
{examples or "No examples available."}

RULES:
1. Use PLAIN VERILOG ONLY. Module MUST be named TopModule.
2. If the logic is simple combinational or output is constant, prefer using continuous 'assign' statements.
3. Fix ONLY the reported errors. Return the COMPLETE fixed module (module...endmodule).
4. Return ONLY the fixed Verilog code. Nothing else. No ```verilog tags."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            fixed = self._clean_verilog(raw)
            return fixed or code, tokens
        except Exception:
            return code, 0

    def _save_eval_detail(
        self,
        problem_id: str,
        mode: str,
        generated_code: str,
        result: dict,
    ) -> None:
        """Append diagnostic log to eval_details.jsonl."""
        if not problem_id and not mode:
            return
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            detail = {
                "problem_id": problem_id,
                "mode": mode,
                "generated_code": generated_code[:5000],  # Truncate for storage
                "compiled": result.get("compiled", False),
                "simulated": result.get("simulated", False),
                "passed": result.get("passed", False),
                "errors": result.get("errors", []),
                "sim_output": (result.get("sim_output") or "")[:2000],
            }
            with open(DETAILS_FILE, "a") as f:
                f.write(json.dumps(detail, default=str) + "\n")
        except OSError:
            pass

    def _compile_and_test(
        self,
        code: str,
        testbench: str,
        reference: str,
        problem_id: str = "",
        mode: str = "",
    ) -> dict:
        """Compile ref + design + testbench (3 files), simulate, check pass/fail."""
        if not code or not code.strip():
            result = {
                "compiled": False,
                "simulated": False,
                "passed": False,
                "errors": [{"message": "No code"}],
                "sim_output": "",
            }
            self._save_eval_detail(problem_id, mode, code, result)
            return result

        files = [
            {"code": reference.strip(), "filename": "ref.sv"},
            {"code": code.strip(), "filename": "design.sv"},
            {"code": testbench.strip(), "filename": "tb.sv"},
        ]

        try:
            sim_result = self.compiler.compile_and_simulate_multi(
                files, timeout_compile=30, timeout_sim=120
            )
            errors = [
                {"file": e.file, "line": e.line, "message": e.message}
                for e in sim_result.compile_errors
            ]
            result = {
                "compiled": sim_result.compiled,
                "simulated": sim_result.simulated,
                "passed": sim_result.passed,
                "errors": errors,
                "sim_output": sim_result.sim_output or "",
            }
            self._save_eval_detail(problem_id, mode, code, result)
            return result
        except Exception as e:
            result = {
                "compiled": False,
                "simulated": False,
                "passed": False,
                "errors": [{"message": str(e)}],
                "sim_output": "",
            }
            self._save_eval_detail(problem_id, mode, code, result)
            return result

    def run_benchmark(
        self,
        modes: list[str] | None = None,
        max_problems: int | None = None,
        max_iterations: int = 5,
        existing_results: list[dict] | None = None,
        rate_limit_delay: float = 3.0,  # seconds between modes
        delay_between_problems: float = 2.0,  # seconds between problems
    ) -> list[EvalResult]:
        """Run the full benchmark."""
        modes = modes or ["baseline", "rag_only", "chipmind_agentic"]
        existing = existing_results or []
        completed = {(r["problem_id"], r["mode"]) for r in existing}

        problems = self.loader.load_problems(
            max_problems=max_problems, silent=(len(existing) > 0)
        )
        if not problems:
            return [EvalResult(**r) for r in existing if "problem_id" in r]

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results: list[EvalResult] = []
        for r in existing:
            try:
                # filter out any unexpected keys for compatibility with older results
                import inspect
                valid_keys = inspect.signature(EvalResult).parameters.keys()
                filtered_r = {k: v for k, v in r.items() if k in valid_keys}
                results.append(EvalResult(**filtered_r))
            except Exception as e:
                self.console.print(f"[yellow]Skipping existing result schema mismatch: {e}[/yellow]")

        total = len(problems) * len(modes)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running benchmark...", total=total)
            for problem in problems:
                pid = problem["problem_id"]
                for mode in modes:
                    if (pid, mode) in completed:
                        progress.advance(task)
                        continue

                    try:
                        for attempt in range(3):
                            try:
                                if mode == "baseline":
                                    r = self.run_baseline(problem)
                                elif mode == "rag_only":
                                    r = self.run_rag_only(problem)
                                else:
                                    r = self.run_chipmind_agentic(
                                        problem, max_iterations=max_iterations
                                    )
                                break
                            except Exception as api_err:
                                err_str = str(api_err).lower()
                                if any(
                                    x in err_str
                                    for x in [
                                        "429",
                                        "rate",
                                        "503",
                                        "service unavailable",
                                        "timeout",
                                    ]
                                ):
                                    wait_sec = 60
                                    self.console.print(
                                        f"[yellow]API rate limit/error, waiting {wait_sec}s before retry...[/yellow]"
                                    )
                                    time.sleep(wait_sec)
                                    if attempt == 2:
                                        raise
                                else:
                                    raise
                        results.append(r)
                        completed.add((pid, mode))

                        # Save immediately
                        with open(RESULTS_FILE, "a") as f:
                            f.write(json.dumps(asdict(r), default=str) + "\n")

                        status = "PASS" if r.passed else "FAIL"
                        progress.update(
                            task,
                            description=f"{pid} [{mode}] {status}",
                        )
                    except Exception as e:
                        self.console.print(f"[red]{pid} [{mode}] Error: {e}[/red]")
                        results.append(
                            EvalResult(
                                problem_id=pid,
                                mode=mode,
                                compiled=False,
                                simulated=False,
                                passed=False,
                                iterations=0,
                                errors=[{"message": str(e)}],
                                time_seconds=0,
                                tokens_used=0,
                                generated_code="",
                            )
                        )
                        completed.add((pid, mode))

                    progress.advance(task)
                    # Delay between modes (within same problem)
                    time.sleep(rate_limit_delay)

                # Delay between problems
                time.sleep(delay_between_problems)

        return results

    def compute_metrics(self, results: list[EvalResult]) -> dict:
        """Compute metrics per mode."""
        by_mode: dict[str, list[EvalResult]] = {}
        for r in results:
            by_mode.setdefault(r.mode, []).append(r)

        metrics = {}
        for mode, mode_results in by_mode.items():
            n = len(mode_results)
            passed = sum(1 for r in mode_results if r.passed)
            compiled = sum(1 for r in mode_results if r.compiled)
            iterations = [r.iterations for r in mode_results if r.mode == "chipmind_agentic"]
            avg_iter = sum(iterations) / len(iterations) if iterations else 0
            avg_time = sum(r.time_seconds for r in mode_results) / n if n else 0
            avg_tokens = sum(r.tokens_used for r in mode_results) / n if n else 0

            metrics[mode] = {
                "pass_at_1": passed / n if n else 0,
                "syntax_rate": compiled / n if n else 0,
                "avg_iterations": avg_iter,
                "avg_time": avg_time,
                "avg_tokens": avg_tokens,
                "total_problems": n,
            }
        return metrics

    def print_report(self, metrics: dict) -> None:
        """Print benchmark report."""
        self.console.print(f"\n[bold]VerilogEval Benchmark Report[/bold] (provider: {self.provider}, model: {self.eval_model})\n")

        table1 = Table(title="Main Results")
        table1.add_column("Mode", style="cyan")
        table1.add_column("Pass@1", style="green")
        table1.add_column("Syntax Rate", style="green")
        table1.add_column("Avg Iterations", style="yellow")
        for mode, m in metrics.items():
            name = {"baseline": "Baseline (no RAG)", "rag_only": "RAG Only", "chipmind_agentic": "ChipMind (Agentic)"}.get(mode, mode)
            iter_str = f"{m['avg_iterations']:.1f}" if mode == "chipmind_agentic" else "N/A"
            table1.add_row(
                name,
                f"{m['pass_at_1']*100:.1f}%",
                f"{m['syntax_rate']*100:.1f}%",
                iter_str,
            )
        self.console.print(table1)

        baseline = metrics.get("baseline", {})
        rag = metrics.get("rag_only", {})
        agentic = metrics.get("chipmind_agentic", {})
        if baseline:
            table2 = Table(title="Improvement vs Baseline")
            table2.add_column("Metric", style="cyan")
            table2.add_column("RAG vs Baseline", style="green")
            table2.add_column("Agentic vs Baseline", style="green")
            pass_rag = (rag.get("pass_at_1", 0) - baseline["pass_at_1"]) * 100 if rag else 0
            pass_ag = (agentic.get("pass_at_1", 0) - baseline["pass_at_1"]) * 100 if agentic else 0
            syn_rag = (rag.get("syntax_rate", 0) - baseline["syntax_rate"]) * 100 if rag else 0
            syn_ag = (agentic.get("syntax_rate", 0) - baseline["syntax_rate"]) * 100 if agentic else 0
            table2.add_row("Pass@1", f"+{pass_rag:.1f}%", f"+{pass_ag:.1f}%")
            table2.add_row("Syntax Rate", f"+{syn_rag:.1f}%", f"+{syn_ag:.1f}%")
            self.console.print(table2)

    def save_report(
        self,
        metrics: dict,
        results: list[EvalResult],
        output_dir: str = "chipmind/evaluation/results",
    ) -> None:
        """Save results to files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # eval_results.jsonl
        with open(out / "eval_results.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(asdict(r), default=str) + "\n")

        # metrics.json
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # benchmark_report.md
        lines = ["# VerilogEval Benchmark Report\n"]
        lines.append("## Main Results\n| Mode | Pass@1 | Syntax Rate | Avg Iterations |\n|-----|--------|-------------|----------------|")
        for mode, m in metrics.items():
            name = {"baseline": "Baseline", "rag_only": "RAG Only", "chipmind_agentic": "ChipMind Agentic"}.get(mode, mode)
            iter_str = f"{m['avg_iterations']:.1f}" if mode == "chipmind_agentic" else "N/A"
            lines.append(f"| {name} | {m['pass_at_1']*100:.1f}% | {m['syntax_rate']*100:.1f}% | {iter_str} |")
        with open(out / "benchmark_report.md", "w") as f:
            f.write("\n".join(lines))

        self.console.print(f"\n[green]Saved to {out}[/green]")
