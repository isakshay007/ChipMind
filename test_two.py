import json
from chipmind.evaluation.verilog_eval_loader import VerilogEvalLoader
from chipmind.evaluation.verilog_eval_runner import VerilogEvalRunner
from rich.console import Console

console = Console()
loader = VerilogEvalLoader()
problems = loader.load_problems(max_problems=156, silent=True)
target = [p for p in problems if p["problem_id"] in ["Prob009_popcount3", "Prob015_vector1"]]

runner = VerilogEvalRunner(provider="nvidia")

for p in target:
    console.print(f"\n--- Testing {p['problem_id']} ---")
    b = runner.run_baseline(p)
    console.print(f"  Baseline: {b.passed}")
    r = runner.run_rag_only(p)
    console.print(f"  RAG: {r.passed}")
    a = runner.run_chipmind_agentic(p)
    console.print(f"  Agentic: {a.passed}")
