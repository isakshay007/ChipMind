# ChipMind CLI UI - Interactive Test Plan

This guide allows you to physically verify all 5 of the new "Quality of Life" UI upgrades we just integrated into `chipmind/cli.py`.

## 1. Interactive Help Auto-Complete
**Test:** Launch the CLI (`make cli`). Type `/` and press `<TAB>`.
**Expected:** A dropdown menu should immediately appear floating above your cursor, listing all 13 slash commands. You should be able to navigate it with your arrow keys and hit Enter to auto-complete.
*Verification of `prompt_toolkit` integration.*

## 2. Robust File Loading (`/load`)
**Test:** 
1. Open a new terminal and run: `echo "Build a parameterizable 4-bit gray code counter" > spec.txt`
2. Inside the ChipMind CLI, type: `/load spec.txt`
**Expected:** The CLI should print `Loaded specification from spec.txt (xx chars)` and immediately begin the generation flow without you having to manually type the query.

## 3. Multi-file Saving (`/save`)
**Test:** 
1. Wait for the gray code counter from the previous test to finish generating.
2. Type: `/save my_counter`
**Expected:** The CLI should now print **two** messages:
- `✓ Saved core to .../output/my_counter.v`
- `✓ Saved testbench to .../output/my_counter_tb.v`
Check your `output/` folder to confirm both the design and its exact testbench are safely exported.

## 4. Animated "Thinking" States (Live Streaming)
**Test:** 
1. Type: `Build an asynchronous FIFO but intentionally mess up the write pointer logic so it fails compilation.`
2. Wait for it to inevitably fail the `Icarus Verilog` sandbox check.
**Expected:** Instead of hanging with a spinner for 5 seconds waiting for the AI to fix it, you should see a cyan `Agent Generating Fix` window pop up, and you should physically watch the Verilog code being furiously typed out in real-time until the block is complete!

## 5. Graceful Model Fallbacks (NVIDIA -> Groq)
**Test:** 
1. *(Simulated)*: If you were to intentionally spam the backend until NVIDIA NIM rate-limits you (returning a `429` error).
**Expected:** Instead of explicitly crashing or hanging the CLI for 60 seconds (as it did previously), it will catch the exact error text, print `[yellow]⚠ NVIDIA NIM rate limited. Auto-falling back to Groq...[/]`, automatically trigger `self._setup_provider("groq")`, and silently re-submit the failed prompt using `llama-3.3-70b-versatile` without you ever losing your debug loop progress.
