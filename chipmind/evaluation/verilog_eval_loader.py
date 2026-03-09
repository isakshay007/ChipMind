"""Load and parse VerilogEval benchmark problems."""

import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


class VerilogEvalLoader:
    """Loads and parses VerilogEval problems."""

    def __init__(self, eval_dir: str | None = None):
        # Try multiple possible locations for VerilogEval data
        project_root = Path(__file__).resolve().parent.parent.parent
        candidates = [
            Path(eval_dir) if eval_dir else None,
            project_root / "data" / "raw" / "verilog-eval",
            project_root / "data" / "raw" / "verilogdb" / "verilog-eval",
        ]
        self.eval_dir = None
        self.spec_to_rtl_dir = None
        for c in candidates:
            if c and (c / "dataset_spec-to-rtl").exists():
                self.eval_dir = c
                self.spec_to_rtl_dir = c / "dataset_spec-to-rtl"
                break
        if not self.spec_to_rtl_dir:
            self.eval_dir = project_root / "data" / "raw" / "verilog-eval"
            self.spec_to_rtl_dir = self.eval_dir / "dataset_spec-to-rtl"

    def discover_format(self) -> None:
        """Print the file structure of 3 sample problems to understand the format."""
        if not self.spec_to_rtl_dir.exists():
            console.print(f"[red]Directory not found: {self.spec_to_rtl_dir}[/red]")
            return

        # Find unique problem prefixes (e.g. Prob001_zero, Prob002_m2014_q4i)
        prompt_files = sorted(self.spec_to_rtl_dir.glob("*_prompt.txt"))[:3]
        if not prompt_files:
            # Try alternate patterns
            prompt_files = sorted(self.spec_to_rtl_dir.glob("*.prompt"))[:3]
        if not prompt_files:
            prompt_files = sorted(self.spec_to_rtl_dir.glob("prompt*"))[:3]

        for pf in prompt_files:
            stem = pf.stem
            # Extract prefix: Prob001_zero from Prob001_zero_prompt
            prefix = re.sub(r"_prompt$", "", stem) if stem.endswith("_prompt") else stem
            prefix = re.sub(r"\.prompt$", "", prefix)

            console.print(f"\n[bold]=== {prefix} ===[/bold]")
            console.print(f"Directory: {self.spec_to_rtl_dir}")

            # Find all related files
            for f in sorted(self.spec_to_rtl_dir.iterdir()):
                if f.name.startswith(prefix + "_") or f.name == prefix:
                    size = f.stat().st_size
                    console.print(f"  [cyan]{f.name}[/cyan] ({size} bytes)")
                    try:
                        content = f.read_text(errors="replace")
                        preview = content[:500].replace("\n", "\\n ")
                        if len(content) > 500:
                            preview += "..."
                        console.print(f"    Preview: {preview}")
                    except OSError as e:
                        console.print(f"    [red]Error reading: {e}[/red]")

    def load_problems(
        self, max_problems: int | None = None, silent: bool = False
    ) -> list[dict]:
        """Load all spec-to-rtl problems.

        For each problem, load:
        - problem_id: str (e.g. "Prob001_zero")
        - description: str (prompt content)
        - reference_solution: str (ref .sv content)
        - testbench: str (test .sv content)

        Handles: *_prompt.txt, *_ref.sv, *_test.sv
        """
        if not self.spec_to_rtl_dir.exists():
            console.print(f"[red]Directory not found: {self.spec_to_rtl_dir}[/red]")
            return []

        problems: list[dict] = []
        skip_reasons: dict[str, int] = {}

        prompt_files = list(self.spec_to_rtl_dir.glob("*_prompt.txt"))
        if not prompt_files:
            prompt_files = list(self.spec_to_rtl_dir.glob("*.prompt"))

        for pf in sorted(prompt_files):
            stem = pf.stem
            prefix = re.sub(r"_prompt$", "", stem) if "_prompt" in stem else stem.replace(".prompt", "")

            # Try ref and test files
            ref_file = self.spec_to_rtl_dir / f"{prefix}_ref.sv"
            if not ref_file.exists():
                ref_file = self.spec_to_rtl_dir / f"{prefix}_reference.sv"
            if not ref_file.exists():
                ref_file = self.spec_to_rtl_dir / f"{prefix}_solution.sv"

            test_file = self.spec_to_rtl_dir / f"{prefix}_test.sv"
            if not test_file.exists():
                test_file = self.spec_to_rtl_dir / f"{prefix}_testbench.sv"
            if not test_file.exists():
                test_file = self.spec_to_rtl_dir / f"{prefix}_tb.sv"

            if not ref_file.exists():
                skip_reasons["missing_ref"] = skip_reasons.get("missing_ref", 0) + 1
                continue
            if not test_file.exists():
                skip_reasons["missing_test"] = skip_reasons.get("missing_test", 0) + 1
                continue

            try:
                description = pf.read_text(errors="replace").strip()
                reference_solution = ref_file.read_text(errors="replace").strip()
                testbench = test_file.read_text(errors="replace").strip()
            except OSError as e:
                skip_reasons["read_error"] = skip_reasons.get("read_error", 0) + 1
                continue

            if not description:
                skip_reasons["empty_description"] = skip_reasons.get("empty_description", 0) + 1
                continue

            problems.append({
                "problem_id": prefix,
                "description": description,
                "reference_solution": reference_solution,
                "testbench": testbench,
            })

            if max_problems and len(problems) >= max_problems:
                break

        # Stats
        if not silent:
            table = Table(title="VerilogEval Loader Stats")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Total loaded", str(len(problems)))
            for reason, count in sorted(skip_reasons.items()):
                table.add_row(f"Skipped ({reason})", str(count))
            console.print(table)

        return problems

    def get_problem(self, problem_id: str) -> dict | None:
        """Load a single problem by ID."""
        prefix = problem_id
        pf = self.spec_to_rtl_dir / f"{prefix}_prompt.txt"
        if not pf.exists():
            return None
        ref_file = self.spec_to_rtl_dir / f"{prefix}_ref.sv"
        test_file = self.spec_to_rtl_dir / f"{prefix}_test.sv"
        if not ref_file.exists() or not test_file.exists():
            return None
        try:
            return {
                "problem_id": prefix,
                "description": pf.read_text(errors="replace").strip(),
                "reference_solution": ref_file.read_text(errors="replace").strip(),
                "testbench": test_file.read_text(errors="replace").strip(),
            }
        except OSError:
            return None
