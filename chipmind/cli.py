"""Main entry point for the ChipMind CLI."""

import os
import sys
import time
import json
import re
import readline  # Enables up-arrow history support automatically
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.status import Status
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.markdown import Markdown

from chipmind.config import settings
from chipmind.retrieval.hybrid_retriever import HybridRetriever
from chipmind.agents.compiler_gate import CompilerGate
from chipmind.agents.error_classifier import ErrorClassifier


class ChipMindCLI:
    """Impressive terminal CLI for ChipMind RTL design."""

    def __init__(self):
        self.console = Console()
        self.session_tokens = 0
        self.session_time = 0.0
        self.generations_count = 0
        self.session_history = []
        self.last_state = {}

        # 1. Setup Provider
        self._setup_provider()

        # 2. Load Prompts
        self._load_prompts()

        # 3. Print Welcome and Load Components
        self._print_welcome()
        self._load_components()

    def _setup_provider(self, name=None):
        """Initialize the LLM client directly (default NVIDIA NIM, fallback Groq)."""
        if name == "nvidia" or (name is None and settings.NVIDIA_API_KEY):
            if not settings.NVIDIA_API_KEY:
                self.console.print("[red]Missing NVIDIA_API_KEY in .env[/]")
                sys.exit(1)
            from openai import OpenAI
            self.llm_client = OpenAI(
                api_key=settings.NVIDIA_API_KEY,
                base_url="https://integrate.api.nvidia.com/v1"
            )
            self.provider = "nvidia"
            self.model = "meta/llama-3.3-70b-instruct"
        elif name == "groq" or (name is None and settings.GROQ_API_KEY):
            if not settings.GROQ_API_KEY:
                self.console.print("[red]Missing GROQ_API_KEY in .env[/]")
                sys.exit(1)
            from groq import Groq
            self.llm_client = Groq(api_key=settings.GROQ_API_KEY)
            self.provider = "groq"
            self.model = "llama-3.3-70b-versatile"
        else:
            self.console.print("[red]No API key found. Set NVIDIA_API_KEY or GROQ_API_KEY in .env[/]")
            sys.exit(1)

    def _load_prompts(self):
        """Load prompts directly from chipmind/agents/prompts/."""
        prompts_dir = Path(__file__).resolve().parent / "agents" / "prompts"
        try:
            self.prompt_spec = (prompts_dir / "spec_analyzer.txt").read_text()
            self.prompt_gen = (prompts_dir / "code_generator.txt").read_text()
            self.prompt_tb = (prompts_dir / "testbench_generator.txt").read_text()
            self.prompt_debug = (prompts_dir / "debug_fix.txt").read_text()
        except OSError as e:
            self.console.print(f"[red]Error loading prompts: {e}[/]")
            sys.exit(1)

    def _load_components(self):
        """Load the non-LLM components with a loading screen."""
        self.retriever = None
        t0 = time.time()
        
        self.console.print("Loading ChipMind...", style="bold")
        
        try:
            indexes_dir = Path(__file__).resolve().parent.parent / "data" / "processed" / "indexes"
            if indexes_dir.exists():
                self.retriever = HybridRetriever.load(str(indexes_dir))
                self.console.print(f"├── ✓ Knowledge base: {len(self.retriever.semantic.metadata) + len(self.retriever.keyword.metadata)} chunks")
                self.console.print(f"├── ✓ FAISS index: {self.retriever.semantic.index.ntotal if self.retriever.semantic.index else 0} vectors")
                self.console.print(f"├── ✓ BM25 index: {len(self.retriever.keyword.bm25.corpus_size) if hasattr(self.retriever.keyword.bm25, 'corpus_size') else len(self.retriever.keyword.metadata)} documents")
            else:
                self.console.print("[yellow]├── ✗ Knowledge base missing (run 'make pipeline')[/]")
        except Exception as e:
            self.console.print(f"[yellow]├── ✗ Failed to load indexes: {e}[/]")

        try:
            self.compiler = CompilerGate()
            self.console.print("├── ✓ Icarus Verilog")
        except RuntimeError:
            self.compiler = None
            self.console.print("[red]├── ✗ Icarus Verilog not found[/]")

        self.error_classifier = ErrorClassifier()
        
        self.console.print(f"├── ✓ LLM: {self.model} via {self.provider.upper()}")
        
        elapsed = time.time() - t0
        self.console.print(f"└── ✓ Ready in {elapsed:.1f}s\n")

    def _print_welcome(self):
        """Print the styling welcome banner."""
        banner = """
╭──────────────────────────────────────────────────────╮
│                                                      │
│   🧠 ChipMind v0.1.0                                 │
│   Multi-Agent RTL Design Assistant                   │
│                                                      │
│   "Describe a circuit. Watch it compile."            │
│                                                      │
│   Type /help for commands or just describe a circuit │
│                                                      │
╰──────────────────────────────────────────────────────╯
"""
        self.console.print(banner, style="bold cyan")

    def _llm_call(self, messages, temperature=0.3, max_tokens=2000):
        """Unified LLM call supporting GROQ and NVIDIA NIM."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            tokens = response.usage.total_tokens if response.usage else 0
            self.session_tokens += tokens
            return response.choices[0].message.content, tokens
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                self.console.print("[yellow]Rate limited. Waiting 60s...[/]")
                time.sleep(60)
                return self._llm_call(messages, temperature, max_tokens)
            raise

    def start(self):
        """Main REPL Loop."""
        while True:
            try:
                # rich string prompting
                query = Prompt.ask(f"[bold cyan]chipmind[/] ([dim]{self.provider}/{self.model.split('-')[1]}[/]) > ")
                
                if not query.strip():
                    continue

                if query.startswith("/"):
                    self._handle_slash_command(query)
                elif query.lower() in ["quit", "exit", "q"]:
                    self.console.print("\n[bold green]Goodbye![/]")
                    break
                else:
                    self._handle_generate_flow(query)
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted[/]")
                continue
            except EOFError:
                self.console.print("\n[bold green]Goodbye![/]")
                break
            except Exception as e:
                self.console.print(f"\n[red]Error: {e}[/]")

    def _handle_slash_command(self, query):
        parts = query.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            self._cmd_help()
        elif command == "/retrieve":
            self._cmd_retrieve(args)
        elif command == "/compile":
            self._cmd_compile(args)
        elif command == "/save":
            self._cmd_save(args)
        elif command == "/explain":
            self._cmd_explain()
        elif command == "/benchmark":
            self._cmd_benchmark(args)
        elif command == "/history":
            self._cmd_history()
        elif command == "/stats":
            self._cmd_stats()
        elif command == "/provider":
            self._cmd_provider(args)
        elif command == "/model":
            self._cmd_model(args)
        elif command == "/debug":
            self._cmd_debug()
        elif command == "/clear":
            self.console.clear()
        else:
            self.console.print(f"[red]Unknown command: {command}. Type /help.[/red]")

    # --- Slash Commands ---
    
    def _cmd_help(self):
        table = Table(title="ChipMind Slash Commands", show_header=False, box=None)
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        cmds = [
            ("/help", "Show this command reference"),
            ("/retrieve <q>", "Search knowledge base, show results"),
            ("/compile [file]", "Compile last generated code or file"),
            ("/save [name]", "Save last generated code to output/"),
            ("/explain", "LLM explains the last generated design"),
            ("/benchmark <n>", "Run VerilogEval with progress bar"),
            ("/history", "Show session generation history"),
            ("/stats", "Show system stats panel"),
            ("/provider <name>", "Switch provider (nvidia/groq)"),
            ("/model <name>", "Switch model"),
            ("/debug", "Show detailed debug info for last generation"),
            ("/clear", "Clear the screen"),
            ("quit/exit/q", "Exit ChipMind"),
        ]
        for cmd, desc in cmds:
            table.add_row(cmd, desc)
        self.console.print(Panel(table, border_style="cyan"))

    def _cmd_provider(self, name):
        name = name.strip().lower()
        if name not in ["nvidia", "groq"]:
            self.console.print("[red]Invalid provider. Use 'nvidia' or 'groq'.[/]")
            return
        self._setup_provider(name)
        self.console.print(f"  ✓ Switched to {name} ({self.model})", style="green")

    def _cmd_model(self, name):
        if not name.strip():
            self.console.print(f"Current model: {self.model}")
            return
        self.model = name.strip()
        self.console.print(f"  ✓ Switched model to {self.model}", style="green")

    def _cmd_save(self, args):
        code = self.last_state.get("code")
        if not code:
            self.console.print("[red]No code to save. Generate something first.[/]")
            return
        
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        filename = args.strip() if args.strip() else f"design_{self.generations_count}.v"
        if not filename.endswith(".v") and not filename.endswith(".sv"):
            filename += ".v"
            
        path = out_dir / filename
        path.write_text(code)
        self.console.print(f"  ✓ Saved to [green]{path.absolute()}[/]")

    def _cmd_explain(self):
        code = self.last_state.get("code")
        if not code:
            self.console.print("[red]No code to explain. Generate something first.[/]")
            return
            
        prompt = f"Explain this Verilog module in simple terms. What does it do? What are inputs and outputs? Keep it under 5 sentences.\n\n```verilog\n{code}\n```"
        messages = [
            {"role": "system", "content": "You are an expert hardware engineer."},
            {"role": "user", "content": prompt}
        ]
        
        with self.console.status("[bold cyan]Analyzing design...") as status:
            explanation, _ = self._llm_call(messages, temperature=0.2, max_tokens=500)
            
        self.console.print(Panel(Markdown(explanation), title="Explanation", border_style="cyan"))

    def _cmd_history(self):
        if not self.session_history:
            self.console.print("[dim]No generations yet in this session.[/]")
            return
            
        table = Table(title="Session History")
        table.add_column("Query", style="cyan", max_width=50)
        table.add_column("Status", justify="center")
        table.add_column("Iterations", justify="right")
        table.add_column("Time", justify="right")
        
        for h in self.session_history:
            status_color = "green" if "Success" in h['status'] else "red"
            status_text = f"[{status_color}]{h['status']}[/]"
            table.add_row(h['query'][:47] + "..." if len(h['query']) > 50 else h['query'], 
                          status_text, 
                          str(h['iterations']), 
                          f"{h['time']:.1f}s")
                          
        self.console.print(table)

    def _cmd_stats(self):
        tree = Tree("[bold]ChipMind System Stats[/]")
        
        kb = tree.add("Knowledge Base")
        if self.retriever:
            kb.add(f"Verilog modules: {len(self.retriever.semantic.metadata)}") # Approximation
            kb.add(f"Total chunks: {len(self.retriever.semantic.metadata) + len(self.retriever.keyword.metadata)}")
        else:
            kb.add("[red]Not loaded[/]")
            
        indices = tree.add("Indexes")
        if self.retriever:
            indices.add(f"FAISS: {self.retriever.semantic.index.ntotal if self.retriever.semantic.index else 0} vectors")
            indices.add(f"BM25: {len(self.retriever.keyword.bm25.corpus_size) if hasattr(self.retriever.keyword.bm25, 'corpus_size') else len(self.retriever.keyword.metadata)} documents")
        else:
            indices.add("[red]Not loaded[/]")
            
        comp = tree.add("Compiler")
        comp.add("Icarus Verilog" if self.compiler else "[red]Not installed[/]")
        
        llm = tree.add("LLM Provider")
        llm.add(f"Provider: {self.provider}")
        llm.add(f"Model: {self.model}")
        
        sess = tree.add("Session")
        sess.add(f"Generations: {self.generations_count}")
        sess.add(f"Total tokens: {self.session_tokens:,}")
        sess.add(f"Total generate time: {self.session_time:.1f}s")
        
        self.console.print(tree)

    def _cmd_debug(self):
        if not self.last_state:
            self.console.print("[dim]No debug info available.[/]")
            return
        try:
            tree = Tree("[bold]Last Debug Context[/]")
            
            spec_node = tree.add("Parsed Specification")
            spec_node.add(json.dumps(self.last_state.get('spec', {}), indent=2))
            
            hist_node = tree.add("Iteration History")
            for h in self.last_state.get('history', []):
                i_node = hist_node.add(f"Iteration {h.get('iteration')} - Compiled: {h.get('compiled')}")
                if h.get('errors'):
                    for e in h.get('errors'):
                        i_node.add(f"[red]Line {e.get('line')}:[/red] {e.get('message')}")
                        
            sim_node = tree.add("Simulation Details")
            sim_node.add(f"Passed: {self.last_state.get('sim_passed')}")
            if self.last_state.get('sim_output'):
                out = self.last_state.get('sim_output')
                sim_node.add(out[:300] + ("..." if len(out) > 300 else ""))
                
            self.console.print(tree)
        except Exception as e:
            self.console.print(f"[red]Error showing debug info: {e}[/]")

    def _cmd_retrieve(self, args):
        if not self.retriever:
            self.console.print("[red]Knowledge base not loaded.[/]")
            return
        if not args.strip():
            self.console.print("[yellow]Usage: /retrieve <query>[/]")
            return
            
        with self.console.status("[bold cyan]Searching knowledge base...") as status:
            results = self.retriever.search_code(args, k=3)
            
        if not results:
            self.console.print("[yellow]No results found.[/]")
            return
            
        table = Table(title=f"Search Results for '{args}'")
        table.add_column("Rank", style="cyan")
        table.add_column("Module")
        table.add_column("Score", style="green")
        table.add_column("Complexity", style="dim")
        
        for i, r in enumerate(results, 1):
            table.add_row(str(i), r.get('module_name', 'Unknown'), f"{r.get('rrf_score', 0):.3f}", r.get('complexity', 'unknown'))
            
        self.console.print(table)
        
        best = results[0]
        code = best.get('code', '')
        if code:
            self._print_verilog_code(code, f"Top Match: {best.get('module_name')}")

    def _cmd_compile(self, args):
        if not self.compiler:
            self.console.print("[red]Compiler not installed.[/]")
            return
            
        if args.strip():
            # Try to compile file
            path = Path(args.strip())
            if not path.exists():
                self.console.print(f"[red]File not found: {path}[/]")
                return
            code = path.read_text()
        else:
            code = self.last_state.get("code")
            if not code:
                self.console.print("[red]No code to compile.[/]")
                return
                
        with self.console.status("[bold cyan]Compiling...") as status:
            res = self.compiler.compile(code)
            
        if res.success:
            self.console.print("  ✓ [bold green]Compilation PASSED[/]")
        else:
            self.console.print(f"  ✗ [bold red]Compilation FAILED ({len(res.errors)} errors)[/]")
            for e in res.errors:
                self.console.print(f"    Line {e.line}: [[red]{e.error_type}[/]] {e.message}")

    def _cmd_benchmark(self, args):
        try:
            n = int(args.strip())
        except:
            n = 5
            
        self.console.print(f"[bold cyan]Starting VerilogEval Benchmark ({n} problems)...[/]")
        try:
            from chipmind.evaluation.verilog_eval_loader import VerilogEvalLoader
            from chipmind.evaluation.verilog_eval_runner import VerilogEvalRunner
        except ImportError:
            self.console.print("[red]Evaluation module not found![/]")
            return
            
        loader = VerilogEvalLoader()
        problems = loader.load_problems(max_problems=n, silent=True)
        if not problems:
            self.console.print("[red]Failed to load problems. Check data dir.[/]")
            return
            
        runner = VerilogEvalRunner(provider=self.provider)
        
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(problems))
            for p in problems:
                res = runner.run_chipmind_agentic(p)
                results.append(res)
                progress.advance(task)
                
        # Tabulate results
        table = Table(title="Benchmark Results")
        table.add_column("Problem", style="cyan")
        table.add_column("Passed", justify="center")
        table.add_column("Compile", justify="center")
        table.add_column("Iters", justify="right")
        
        passed_count = 0
        for r in results:
            pass_status = "[green]✓[/]" if r.get('passed', False) else "[red]✗[/]"
            comp_status = "[green]✓[/]" if r.get('compiled', False) else "[red]✗[/]"
            passed_count += 1 if r.get('passed', False) else 0
            table.add_row(r.get('problem_id', '?'), pass_status, comp_status, str(r.get('iterations', 0)))
            
        self.console.print(table)
        self.console.print(f"\n[bold green]Accuracy:[/] {passed_count}/{len(problems)} ({(passed_count/len(problems))*100:.1f}%)")

    # --- Core Generation Flow ---

    def _handle_generate_flow(self, query):
        """The main, step-by-step Verilog generation loop."""
        self.console.print()
        start_time = time.time()
        iter_count = 0
        total_tokens = 0
        
        # 1. Spec Analysis
        with self.console.status("[bold cyan]🔍 Analyzing specification...") as status:
            spec, toks = self._analyze_spec(query)
            total_tokens += toks
            module_name = spec.get('module_name', 'design')
            ports_len = len(spec.get('inputs', [])) + len(spec.get('outputs', []))
            comp = spec.get('complexity_hint', 'combinational')
            
        self.console.print(f"  ┃ Module: [white]{module_name}[/]", style="dim")
        self.console.print(f"  ┃ Ports: [white]{ports_len} total[/]", style="dim")
        self.console.print(f"  ┃ Type: [white]{comp}[/]", style="dim")
        self.console.print()

        # 2. RAG Retrieval
        retrieved = []
        if self.retriever:
            with self.console.status("[bold cyan]📚 Searching knowledge base...") as status:
                rag_query = self._build_retrieval_query(spec)
                retrieved = self.retriever.search_code(rag_query, k=3)
                
            self.console.print(f"  ┃ Retrieved {len(retrieved)} similar designs{':' if retrieved else ''}", style="dim")
            for i, r in enumerate(retrieved, 1):
                name = r.get('module_name', f'example_{i}')
                score = r.get('rrf_score', 0)
                lines = r.get('line_count', 0)
                self.console.print(f"  ┃   {i}. {name} (score: {score:.3f}) — {lines} lines", style="dim")
        self.console.print()

        # 3. Generate Code
        with self.console.status("[bold cyan]⚡ Generating Verilog...") as status:
            code, toks = self._generate_code(spec, retrieved)
            total_tokens += toks
        lines_gen = len(code.splitlines())
        self.console.print(f"  ┃ Generated {lines_gen} lines", style="dim")
        self.console.print()
        
        # 4. Generate Testbench
        with self.console.status("[bold cyan]📝 Generating testbench...") as status:
            tb_code, toks = self._generate_testbench(spec, code)
            total_tokens += toks
        self.console.print(f"  ┃ Generated {len(tb_code.splitlines())} lines", style="dim")
        self.console.print()

        # 5. Compile and Debug Loop
        is_success = False
        iteration_history = []
        sim_passed = False
        sim_output = ""
        
        for iteration in range(1, 6):
            if self.compiler:
                with self.console.status(f"[bold cyan]🔨 Compiling with Icarus Verilog... (Iter {iteration})") as status:
                    sim_result = self.compiler.compile_and_simulate(code, tb_code)
                    
                if not sim_result.compiled:
                    self.console.print(f"  ┃ ✗ Compilation FAILED ({len(sim_result.compile_errors)} errors)", style="red")
                    for i, e in enumerate(sim_result.compile_errors[:3]):
                        label = f"[[dim]{e.error_type}[/]]"
                        self.console.print(f"  ┃   Line {e.line}: {label} {e.message}", style="yellow")
                    if len(sim_result.compile_errors) > 3:
                        self.console.print(f"  ┃   ... and {len(sim_result.compile_errors)-3} more", style="dim")
                    self.console.print()
                    
                    history_entry = {"iteration": iteration, "compiled": False, "errors": [vars(e) for e in sim_result.compile_errors]}
                    iteration_history.append(history_entry)
                    
                    # Debug logic
                    with self.console.status("[bold cyan]🔧 Debugging... Analyzing errors & generating fix...") as status:
                        fixed_code, toks = self._debug_fix(spec, code, sim_result.compile_errors)
                        total_tokens += toks
                    
                    if fixed_code.strip() == code.strip():
                        self.console.print("  ┃ [yellow]LLM produced identical code. Aborting debug loop to save tokens.[/]")
                        self.console.print()
                        iter_count += 1
                        break
                        
                    code = fixed_code
                    self.console.print("  ┃ Regenerated code", style="dim")
                    self.console.print()
                    iter_count += 1
                    continue
                else:
                    self.console.print("  ┃ ✓ Compilation PASSED", style="green")
                    self.console.print()
                    
                    # Simulation check
                    with self.console.status("[bold cyan]🧪 Running simulation...") as status:
                        sim_passed = sim_result.passed
                        sim_output = sim_result.sim_output
                        
                    if sim_passed:
                        self.console.print("  ┃ ✓ All tests passed", style="green")
                        self.console.print()
                        is_success = True
                        history_entry = {"iteration": iteration, "compiled": True, "simulated": True, "passed": True}
                        iteration_history.append(history_entry)
                        iter_count += 1
                        break
                    else:
                        self.console.print("  ┃ ✗ Simulation FAILED", style="red")
                        self.console.print()
                        
                        history_entry = {"iteration": iteration, "compiled": True, "simulated": True, "passed": False}
                        iteration_history.append(history_entry)
                        
                        # Debug logic for Sim fail
                        mock_error = [{"line": 0, "error_type": "simulation_fail", "message": f"Simulation failed. Output: {sim_output[:300]}"}]
                        with self.console.status("[bold cyan]🔧 Debugging... Analyzing simulation mismatch...") as status:
                            fixed_code, toks = self._debug_fix(spec, code, mock_error)
                            total_tokens += toks
                            
                    if fixed_code.strip() == code.strip():
                        self.console.print("  ┃ [yellow]LLM produced identical code. Aborting debug loop to save tokens.[/]")
                        self.console.print()
                        iter_count += 1
                        break
                        
                    code = fixed_code
                    self.console.print("  ┃ Regenerated code", style="dim")
                    self.console.print()
                    iter_count += 1
                    continue
            else:
                # No compiler available
                is_success = True
                iter_count = 1
                break

        elapsed = time.time() - start_time
        self.session_time += elapsed
        self.generations_count += 1
        
        status_label = "Success" if is_success else "Partial/Failed"
        self.console.print(f"✅ {status_label} — {iter_count} iterations, {elapsed:.1f}s, {total_tokens:,} tokens", style="bold green" if is_success else "bold yellow")
        
        # Display Code snippet
        self._print_verilog_code(code, f"{module_name}.v", is_success)
        
        # Save state for Slash commands
        self.last_state = {
            "code": code,
            "tb_code": tb_code,
            "spec": spec,
            "history": iteration_history,
            "sim_passed": sim_passed,
            "sim_output": sim_output
        }
        self.session_history.append({
            "query": query,
            "status": status_label,
            "iterations": iter_count,
            "time": elapsed
        })
        
        self.console.print("  /save to save · /explain for explanation · /debug for details", style="dim")
        self.console.print()


    # --- LLM Parsing Helpers ---

    def _analyze_spec(self, query):
        messages = [
            {"role": "system", "content": self.prompt_spec},
            {"role": "user", "content": query}
        ]
        raw_res, toks = self._llm_call(messages, temperature=0.2, max_tokens=1000)
        
        # Quick parse
        cleaned = raw_res.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*\n(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()
        try:
            spec = json.loads(cleaned)
        except:
            spec = {
                "module_name": "design",
                "description": query,
                "inputs": [{"name": "clk", "width": 1}],
                "outputs": [{"name": "out", "width": 1}],
                "functionality": query,
                "complexity_hint": "combinational",
                "constraints": []
            }
        return spec, toks

    def _build_retrieval_query(self, spec):
        parts = [spec.get("description", "")]
        parts.append(spec.get("complexity_hint", ""))
        for inp in spec.get("inputs", []):
             parts.append(inp.get("name", ""))
        for out in spec.get("outputs", []):
             parts.append(out.get("name", ""))
        return " ".join(p for p in parts if p)

    def _format_examples(self, retrieved):
        formatted = []
        for i, m in enumerate(retrieved[:3]):
            name = m.get("module_name", f"example_{i}")
            code = m.get("code", "")
            lines = code.split("\n")
            if len(lines) > 80:
                code = "\n".join(lines[:80]) + "\n// ... (truncated)"
            formatted.append(f"--- Example {i+1}: {name} ---\n{code}")
        return "\n\n".join(formatted) if formatted else "No examples available."

    def _generate_code(self, spec, retrieved):
        spec_str = json.dumps(spec, indent=2)
        examples_str = self._format_examples(retrieved)
        prompt = self.prompt_gen.replace("{spec}", spec_str).replace("{examples}", examples_str)
        
        messages = [{"role": "user", "content": prompt}]
        raw, toks = self._llm_call(messages, temperature=0.3, max_tokens=2000)
        return self._clean_verilog(raw), toks

    def _generate_testbench(self, spec, code):
        spec_str = json.dumps(spec, indent=2)
        module_name = spec.get('module_name', 'design')
        prompt = self.prompt_tb.replace("{spec}", spec_str).replace("{code}", code).replace("{module_name}", module_name)
        
        messages = [{"role": "user", "content": prompt}]
        raw, toks = self._llm_call(messages, temperature=0.2, max_tokens=2000)
        return self._clean_verilog(raw), toks

    def _debug_fix(self, spec, code, errors):
        # Format errors
        lines = []
        for e in errors:
            if isinstance(e, dict):
                lines.append(f"Line {e.get('line', '?')}: [{e.get('error_type', 'unknown')}] {e.get('message', '')}")
            else:
                lines.append(f"Line {e.line}: [{e.error_type}] {e.message}")
        error_str = "\n".join(lines)
        
        spec_str = json.dumps(spec, indent=2)
        prompt = self.prompt_debug.replace("{spec}", spec_str).replace("{code}", code).replace("{errors}", error_str).replace("{examples}", "No examples available.")
        
        messages = [{"role": "user", "content": prompt}]
        raw, toks = self._llm_call(messages, temperature=0.2, max_tokens=2000)
        return self._clean_verilog(raw), toks

    def _clean_verilog(self, raw: str) -> str:
        text = raw.strip()
        if "```" in text:
            match = re.search(r"```(?:verilog|sv|systemverilog)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
            else:
                text = re.sub(r"```(?:verilog|sv|systemverilog)?\s*", "", text, flags=re.IGNORECASE).replace("```", "").strip()
        match = re.search(r"(module\s+\w+[\s\S]*?endmodule)", text, re.IGNORECASE)
        if match:
            text = match.group(1)
        return text.strip()

    def _print_verilog_code(self, code, title, is_success=True):
        lines = code.splitlines()
        truncated = False
        remaining = 0
        if len(lines) > 25:
            code_display = "\n".join(lines[:25])
            remaining = len(lines) - 25
            truncated = True
        else:
            code_display = code
            
        syntax = Syntax(code_display, "verilog", theme="monokai", line_numbers=True)
        border_color = "green" if is_success else "red"
        
        self.console.print(Panel(syntax, title=title, border_style=border_color, padding=(1, 2)))
        
        if truncated:
            self.console.print(f"  ... ({remaining} more lines — use /save to export full code)", style="dim italic")

def main():
    cli = ChipMindCLI()
    cli.start()

if __name__ == "__main__":
    main()
