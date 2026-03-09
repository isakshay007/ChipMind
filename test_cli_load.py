from chipmind.cli import ChipMindCLI
import builtins

inputs = ['n']
def mock_input(prompt):
    print("MOCK INPUT PROMPT:", prompt)
    return inputs.pop(0)

builtins.input = mock_input

cli = ChipMindCLI()
cli._cmd_load("test_buggy.v")
