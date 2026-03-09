"""
Analyze VerilogEval format - run to understand dataset structure.
Usage: python -m chipmind.evaluation.analyze_format
"""

from chipmind.evaluation.verilog_eval_loader import VerilogEvalLoader


def main() -> None:
    loader = VerilogEvalLoader()
    problems = loader.load_problems(max_problems=5, silent=True)

    if not problems:
        print("No problems loaded. Check data path.")
        return

    for p in problems:
        print(f"\n=== {p['problem_id']} ===")
        print("Description (first 200 chars):", p["description"][:200])
        ref = p["reference_solution"]
        tb = p["testbench"]

        # What modules does testbench instantiate?
        if "RefModule" in tb:
            print("  Testbench instantiates: RefModule")
        if "TopModule" in tb:
            print("  Testbench instantiates: TopModule")

        # Reference module name
        if "module RefModule" in ref or "module RefModule" in ref:
            print("  Reference uses: RefModule")

        # Include?
        if "`include" in tb:
            print("  Testbench has `include")

        # Pass/fail patterns
        if "Mismatches:" in tb:
            print("  Pass/fail: Mismatches: N in M samples")
        if "PASS" in tb or "FAIL" in tb:
            print("  Has PASS/FAIL strings")

        # Timescale
        if "`timescale" in tb:
            print("  Testbench has `timescale")


if __name__ == "__main__":
    main()
