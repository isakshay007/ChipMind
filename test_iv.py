from chipmind.agents.compiler_gate import CompilerGate
import os
cg = CompilerGate()
with open("test_buggy.v", "r") as f:
    code = f.read()
res = cg.compile(code)
print("SUCCESS:", res.success)
print("ERRORS:", res.errors)
