"""Tests for CompilerGate (iverilog wrapper)."""

import pytest


@pytest.fixture
def gate():
    """Create CompilerGate, skip if iverilog not found."""
    try:
        from chipmind.agents.compiler_gate import CompilerGate
        return CompilerGate()
    except RuntimeError as e:
        if "iverilog not found" in str(e):
            pytest.skip("iverilog not installed")
        raise


def test_valid_code(gate):
    """Should compile successfully."""
    result = gate.compile("""
    module half_adder(input a, input b, output sum, output carry);
        assign sum = a ^ b;
        assign carry = a & b;
    endmodule
    """)
    assert result.success is True
    assert len(result.errors) == 0


def test_syntax_error(gate):
    """Missing semicolon — should fail with syntax error."""
    result = gate.compile("""
    module bad(input a, output b)
        assign b = a;
    endmodule
    """)
    assert result.success is False
    assert len(result.errors) > 0
    assert result.errors[0].error_type == "syntax"


def test_undeclared_signal(gate):
    """Using undeclared signal — should fail."""
    result = gate.compile("""
    module bad2(input a, output b);
        assign b = xyz;
    endmodule
    """)
    assert result.success is False
    assert any(e.error_type == "undeclared" for e in result.errors)


def test_compile_and_simulate_pass(gate):
    """Design + testbench that should pass simulation."""
    design = """
    module half_adder(input a, input b, output sum, output carry);
        assign sum = a ^ b;
        assign carry = a & b;
    endmodule
    """
    testbench = """
    module tb;
        reg a, b;
        wire sum, carry;
        half_adder uut(.a(a), .b(b), .sum(sum), .carry(carry));
        initial begin
            a=0; b=0; #10;
            $display("a=%b b=%b sum=%b carry=%b %s", a, b, sum, carry,
                     (sum===0 && carry===0) ? "PASS" : "FAIL");
            a=1; b=1; #10;
            $display("a=%b b=%b sum=%b carry=%b %s", a, b, sum, carry,
                     (sum===0 && carry===1) ? "PASS" : "FAIL");
            $finish;
        end
    endmodule
    """
    result = gate.compile_and_simulate(design, testbench)
    assert result.compiled is True
    assert result.simulated is True
    assert "FAIL" not in result.sim_output
    assert result.passed is True


def test_empty_code(gate):
    """Empty code should fail gracefully."""
    result = gate.compile("")
    assert result.success is False
    assert len(result.errors) > 0


def test_empty_code_whitespace(gate):
    """Whitespace-only code should fail gracefully."""
    result = gate.compile("   \n\t  ")
    assert result.success is False


@pytest.mark.slow
def test_timeout(gate):
    """Code with infinite loop (no $finish) — should timeout (may take ~60s)."""
    result = gate.compile_and_simulate(
        "module inf(input a, output b); assign b = a; endmodule",
        """
        module tb;
            reg a; wire b;
            inf uut(.a(a), .b(b));
            initial begin
                forever #1 a = ~a;
            end
        endmodule
        """,
    )
    # Should compile; simulation may timeout (simulated=False) or run
    assert result.compiled is True
    # Either timed out (simulated=False) or ran (simulated=True)
    assert result.passed is False  # No $finish, so output won't have PASS
