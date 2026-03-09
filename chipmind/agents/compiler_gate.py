"""
Compiler gate: wraps Icarus Verilog (iverilog) for real compilation and simulation.
Provides the LLM with actual compiler feedback for self-correction.
"""

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompilerError:
    """Structured compiler error from iverilog."""

    file: str
    line: int
    error_type: str  # syntax, undeclared, width_mismatch, type_error, missing_module, other
    message: str
    raw: str


@dataclass
class CompileResult:
    """Result of iverilog compilation."""

    success: bool
    errors: list[CompilerError]
    warnings: list[str]
    raw_stderr: str
    raw_stdout: str


@dataclass
class SimResult:
    """Result of compile + simulate (design + testbench)."""

    compiled: bool
    simulated: bool
    compile_errors: list[CompilerError]
    sim_output: str
    passed: bool
    raw_stderr: str


def _patch_verilogeval_testbench(tb_code: str) -> str:
    """Fix VerilogEval testbench use-before-declaration for iverilog compatibility.

    The testbench uses tb_mismatch in $dumpvars before declaring it.
    Swap: move 'wire tb_match; wire tb_mismatch = ~tb_match' before the initial block.
    See: https://github.com/NVlabs/verilog-eval/issues/9
    """
    # Match: initial begin ... $dumpfile ... $dumpvars ... end ... wire tb_match ... wire tb_mismatch
    pattern = re.compile(
        r"(initial begin\s+\$dumpfile\([^)]+\);\s*\$dumpvars\([^)]+\);\s*end)\s*"
        r"(wire\s+tb_match;\s*//\s*Verification\s*wire\s+tb_mismatch\s*=\s*~tb_match;)",
        re.DOTALL,
    )
    match = pattern.search(tb_code)
    if match:
        init_block, wire_decls = match.group(1), match.group(2)
        return tb_code[: match.start()] + wire_decls + "\n\n" + init_block + tb_code[match.end() :]
    # More flexible pattern
    pattern2 = re.compile(
        r"(initial begin\s+\$dumpfile\([^)]+\);\s*\$dumpvars\([^)]+\);\s*end)\s*"
        r"(wire\s+tb_match;.*?wire\s+tb_mismatch\s*=\s*~tb_match;)",
        re.DOTALL,
    )
    match2 = pattern2.search(tb_code)
    if match2:
        init_block, wire_decls = match2.group(1), match2.group(2)
        return tb_code[: match2.start()] + wire_decls + "\n\n\t" + init_block + tb_code[match2.end() :]
    return tb_code


def _classify_error_type(message: str) -> str:
    """Classify error message into error_type."""
    msg_lower = message.lower()
    if "syntax error" in msg_lower:
        return "syntax"
    if any(
        x in msg_lower
        for x in [
            "unable to bind",
            "unable to elaborate",
            "not defined",
            "undeclared",
            "unknown identifier",
        ]
    ):
        return "undeclared"
    if any(x in msg_lower for x in ["width", "port size", "size mismatch"]):
        return "width_mismatch"
    if any(x in msg_lower for x in ["not a valid l-value", "reg vs wire", "lvalue"]):
        return "type_error"
    if any(x in msg_lower for x in ["unknown module", "module type", "cannot find module"]):
        return "missing_module"
    return "other"


class CompilerGate:
    """Wraps iverilog for compilation and simulation."""

    COMPILE_TIMEOUT = 30
    SIM_TIMEOUT = 60

    def __init__(self):
        iverilog_path = shutil.which("iverilog")
        if not iverilog_path:
            raise RuntimeError(
                "iverilog not found. Install Icarus Verilog: "
                "https://github.com/steveicarus/iverilog"
            )

    def compile(self, verilog_code: str) -> CompileResult:
        """Compile Verilog code using iverilog."""
        if not verilog_code or not verilog_code.strip():
            return CompileResult(
                success=False,
                errors=[CompilerError("", 0, "other", "Empty code", "Empty code")],
                warnings=[],
                raw_stderr="Empty code",
                raw_stdout="",
            )

        input_path = None
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".v", delete=False, encoding="utf-8"
            ) as f:
                f.write(verilog_code)
                input_path = f.name
            output_path = input_path.replace(".v", ".vvp")

            result = subprocess.run(
                ["iverilog", "-g2012", "-o", output_path, input_path],
                capture_output=True,
                text=True,
                timeout=self.COMPILE_TIMEOUT,
            )
            errors, warnings = self._parse_errors(result.stderr)
            return CompileResult(
                success=result.returncode == 0,
                errors=errors,
                warnings=warnings,
                raw_stderr=result.stderr,
                raw_stdout=result.stdout,
            )
        except subprocess.TimeoutExpired:
            return CompileResult(
                success=False,
                errors=[
                    CompilerError(
                        input_path or "",
                        0,
                        "other",
                        "Compilation timed out",
                        "Compilation timed out",
                    )
                ],
                warnings=[],
                raw_stderr="Compilation timed out",
                raw_stdout="",
            )
        except FileNotFoundError as e:
            return CompileResult(
                success=False,
                errors=[
                    CompilerError(
                        "",
                        0,
                        "other",
                        f"iverilog not found: {e}",
                        str(e),
                    )
                ],
                warnings=[],
                raw_stderr=str(e),
                raw_stdout="",
            )
        finally:
            for p in [input_path, output_path]:
                if p and Path(p).exists():
                    try:
                        Path(p).unlink()
                    except OSError:
                        pass

    def compile_and_simulate(
        self, design_code: str, testbench_code: str
    ) -> SimResult:
        """Compile design + testbench and run simulation."""
        design_path = None
        tb_path = None
        vvp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".v", delete=False, prefix="design_", encoding="utf-8"
            ) as f:
                f.write(design_code)
                design_path = f.name
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".v", delete=False, prefix="tb_", encoding="utf-8"
            ) as f:
                f.write(testbench_code)
                tb_path = f.name
            vvp_path = str(Path(design_path).with_suffix(".vvp"))

            # Compile (use -g2012 for SystemVerilog support in testbenches)
            compile_result = subprocess.run(
                ["iverilog", "-g2012", "-o", vvp_path, design_path, tb_path],
                capture_output=True,
                text=True,
                timeout=self.COMPILE_TIMEOUT,
            )
            errors, _ = self._parse_errors(compile_result.stderr)

            if compile_result.returncode != 0:
                return SimResult(
                    compiled=False,
                    simulated=False,
                    compile_errors=errors,
                    sim_output="",
                    passed=False,
                    raw_stderr=compile_result.stderr,
                )

            # Simulate
            try:
                sim_result = subprocess.run(
                    ["vvp", vvp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.SIM_TIMEOUT,
                )
                sim_output = sim_result.stdout or ""
                raw_stderr = sim_result.stderr or ""

                # passed: no FAIL, ERROR, or MISMATCH in output
                output_upper = sim_output.upper()
                failed = (
                    "FAIL" in output_upper
                    or "ERROR" in output_upper
                    or "MISMATCH" in output_upper
                )
                passed = not failed and sim_result.returncode == 0

                return SimResult(
                    compiled=True,
                    simulated=True,
                    compile_errors=[],
                    sim_output=sim_output,
                    passed=passed,
                    raw_stderr=raw_stderr,
                )
            except subprocess.TimeoutExpired:
                return SimResult(
                    compiled=True,
                    simulated=False,
                    compile_errors=[],
                    sim_output="",
                    passed=False,
                    raw_stderr="Simulation timed out",
                )
        finally:
            for p in [design_path, tb_path, vvp_path]:
                if p and Path(p).exists():
                    try:
                        Path(p).unlink()
                    except OSError:
                        pass

    def compile_and_simulate_multi(
        self,
        files: list[dict],
        timeout_compile: int = 30,
        timeout_sim: int = 120,
    ) -> SimResult:
        """Compile and simulate multiple Verilog/SystemVerilog files.

        files: list of {"code": str, "filename": str}
        Example: [
            {"code": ref_solution, "filename": "ref.sv"},
            {"code": generated_code, "filename": "design.sv"},
            {"code": testbench, "filename": "tb.sv"},
        ]

        1. Write all files to a temp directory
        2. Compile: iverilog -g2012 -o sim.vvp file1 file2 file3
        3. Simulate: vvp sim.vvp (with timeout)
        4. Parse results
        5. Clean up
        """
        if not files:
            return SimResult(
                compiled=False,
                simulated=False,
                compile_errors=[
                    CompilerError("", 0, "other", "No files provided", "No files")
                ],
                sim_output="",
                passed=False,
                raw_stderr="No files",
            )

        temp_dir = None
        file_paths: list[str] = []
        vvp_path = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="verilog_eval_")
            temp_path = Path(temp_dir)

            for f in files:
                code = f.get("code", "")
                filename = f.get("filename", "design.v")
                if not filename.endswith((".v", ".sv")):
                    filename = filename + ".v"
                # Patch VerilogEval testbench for iverilog compatibility
                if "tb.sv" in filename or "tb.v" in filename:
                    code = _patch_verilogeval_testbench(code)
                path = temp_path / filename
                path.write_text(code, encoding="utf-8")
                file_paths.append(str(path))

            vvp_path = str(temp_path / "sim.vvp")

            # Compile
            compile_result = subprocess.run(
                ["iverilog", "-g2012", "-o", vvp_path] + file_paths,
                capture_output=True,
                text=True,
                timeout=timeout_compile,
                cwd=temp_dir,
            )
            errors, _ = self._parse_errors(compile_result.stderr)

            if compile_result.returncode != 0:
                return SimResult(
                    compiled=False,
                    simulated=False,
                    compile_errors=errors,
                    sim_output="",
                    passed=False,
                    raw_stderr=compile_result.stderr,
                )

            # Simulate
            try:
                sim_result = subprocess.run(
                    ["vvp", vvp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_sim,
                    cwd=temp_dir,
                )
                sim_output = sim_result.stdout or ""
                raw_stderr = sim_result.stderr or ""

                # Pass/fail: VerilogEval uses "Mismatches: 0 in N samples"
                output_upper = sim_output.upper()
                failed = (
                    "FAIL" in output_upper
                    or "ERROR" in output_upper
                    or "TIMEOUT" in output_upper
                )
                # VerilogEval: "Mismatches: 0 in N samples" = pass
                mismatch_match = re.search(
                    r"Mismatches:\s*(\d+)\s+in\s+\d+\s+samples",
                    sim_output,
                    re.IGNORECASE,
                )
                if mismatch_match:
                    mismatches = int(mismatch_match.group(1))
                    passed = not failed and mismatches == 0 and sim_result.returncode == 0
                else:
                    # Fallback: no FAIL/ERROR/MISMATCH
                    passed = (
                        not failed
                        and "MISMATCH" not in output_upper
                        and sim_result.returncode == 0
                    )

                return SimResult(
                    compiled=True,
                    simulated=True,
                    compile_errors=[],
                    sim_output=sim_output,
                    passed=passed,
                    raw_stderr=raw_stderr,
                )
            except subprocess.TimeoutExpired:
                return SimResult(
                    compiled=True,
                    simulated=False,
                    compile_errors=[],
                    sim_output="",
                    passed=False,
                    raw_stderr="Simulation timed out",
                )
        except subprocess.TimeoutExpired:
            return SimResult(
                compiled=False,
                simulated=False,
                compile_errors=[
                    CompilerError(
                        "", 0, "other", "Compilation timed out", "Compilation timed out"
                    )
                ],
                sim_output="",
                passed=False,
                raw_stderr="Compilation timed out",
            )
        except Exception as e:
            return SimResult(
                compiled=False,
                simulated=False,
                compile_errors=[
                    CompilerError("", 0, "other", str(e), str(e))
                ],
                sim_output="",
                passed=False,
                raw_stderr=str(e),
            )
        finally:
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except OSError:
                    pass

    def _parse_errors(self, stderr: str) -> tuple[list[CompilerError], list[str]]:
        """Parse iverilog stderr into structured errors and warnings."""
        errors: list[CompilerError] = []
        warnings: list[str] = []

        # Pattern: file:line: (warning|error)?: message
        pattern = re.compile(
            r"([^:]+):(\d+):\s*(warning|error)?:?\s*(.*)",
            re.IGNORECASE,
        )

        for line in stderr.splitlines():
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                file_part, line_str, level, message = match.groups()
                line_num = int(line_str) if line_str.isdigit() else 0
                message = (message or "").strip()

                if level and "warning" in level.lower():
                    warnings.append(line)
                else:
                    error_type = _classify_error_type(message)
                    errors.append(
                        CompilerError(
                            file=file_part.strip(),
                            line=line_num,
                            error_type=error_type,
                            message=message,
                            raw=line,
                        )
                    )
            elif "error" in line.lower() or "warning" in line.lower():
                # Fallback: treat as other error
                errors.append(
                    CompilerError(
                        file="",
                        line=0,
                        error_type="other",
                        message=line,
                        raw=line,
                    )
                )

        return errors, warnings
