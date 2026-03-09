import sys
from pathlib import Path
from rich.console import Console

from chipmind.evaluation.verilog_eval_loader import VerilogEvalLoader
from chipmind.evaluation.verilog_eval_runner import VerilogEvalRunner

console = Console()

def main():
    problems_to_check = [
        "Prob001_zero", "Prob005_notgate", "Prob010_mt2015_q4a", 
        "Prob015_mux2to1", "Prob020_mux256to1", "Prob030_popcount255",
        "Prob050_vectorgates", "Prob070_gates4", "Prob090_mux2to1v", "Prob100_truthtable1"
    ]

    loader = VerilogEvalLoader()
    runner = VerilogEvalRunner()
    retriever = runner.retriever
    
    if not retriever:
        console.print("[red]Retriever not loaded. Make sure indexes exist.[/red]")
        sys.exit(1)

    total_ref_len = 0
    total_ret_len = 0
    num_problems = 0

    for pid in problems_to_check:
        prob = loader.get_problem(pid)
        if not prob:
            console.print(f"[yellow]Could not load {pid}[/yellow]")
            continue
            
        desc = prob.get("description", "")
        ref = prob.get("reference_solution", "")
        
        spec = runner._parse_verilogeval_description(desc)
        query = " ".join(
            [spec.get("functionality", "")]
            + [p["name"] for p in spec.get("inputs", [])]
            + [p["name"] for p in spec.get("outputs", [])]
        ) or "verilog"
        
        retrieved = retriever.search_code(query, k=3)
        
        console.print(f"\n[bold cyan]=== Problem: {pid} ===[/bold cyan]")
        console.print(f"[bold]DESCRIPTION (first 200 chars):[/bold]\n{desc[:200]}...")
        console.print(f"\n[bold]REFERENCE SOLUTION:[/bold]\n{ref}")
        
        console.print("\n[bold]WHAT RAG RETRIEVES (top 3):[/bold]")
        
        avg_ret_len_for_prob = 0
        
        for i, m in enumerate(retrieved):
            mod_name = m.get("module_name", "Unknown")
            rrf = m.get("rrf_score", 0)
            sem_rank = m.get("semantic_rank", "?")
            kw_rank = m.get("keyword_rank", "?")
            code = m.get("code", "")
            code_lines = code.split("\n")
            c_len = len(code_lines)
            tags = m.get("tags", [])
            complexity = m.get("complexity", "Unknown")
            
            avg_ret_len_for_prob += c_len
            
            console.print(f"\n- Module: [green]{mod_name}[/green]")
            console.print(f"  RRF Score: {rrf:.4f}")
            console.print(f"  Ranks: Semantic={sem_rank} / Keyword={kw_rank}")
            console.print(f"  Code Length: {c_len} lines")
            console.print(f"  Complexity: {complexity}, Tags: {tags}")
            first_10 = "\n    ".join(code_lines[:10])
            console.print(f"  First 10 lines:\n    {first_10}")

        if retrieved:
            avg_ret_len_for_prob /= len(retrieved)
            
        ref_lines = len(ref.split("\n"))
        total_ref_len += ref_lines
        total_ret_len += avg_ret_len_for_prob
        num_problems += 1
        
        console.print("\n[bold yellow]ANALYSIS:[/bold yellow]")
        console.print("- Is the retrieved code RELEVANT to this problem? (similar functionality?)")
        console.print("- Is the retrieved code TOO COMPLEX for this simple problem?")
        console.print(f"- Length comparison: Retrieved avg {avg_ret_len_for_prob:.1f} lines vs Reference {ref_lines} lines")
        console.print("- Would including this code in the prompt help or confuse a small LLM?")

    if num_problems > 0:
        console.print("\n[bold cyan]=== SUMMARY ===[/bold cyan]")
        console.print(f"Average Reference Code Length: {total_ref_len / num_problems:.1f} lines")
        console.print(f"Average Retrieved Code Length: {total_ret_len / num_problems:.1f} lines")
        console.print("Number of problems getting relevant retrievals: (Manual analysis required based on output)")
        console.print("Common issue patterns:")
        console.print("- (Manual analysis required based on output)")

if __name__ == '__main__':
    main()
