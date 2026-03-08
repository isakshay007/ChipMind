"""Verilog module chunker for RAG pipeline."""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

# Tag patterns: (pattern, tag) - pattern is regex or substring (case-insensitive)
TAG_PATTERNS = [
    (r"\balu\b", "alu"),
    (r"\badd(er|_)?", "adder"),
    (r"\bcount(er)?", "counter"),
    (r"\bfifo\b", "fifo"),
    (r"\buart\b", "uart"),
    (r"\bmux\b", "mux"),
    (r"\bdecod(e|er)", "decoder"),
    (r"\bencod(e|er)", "encoder"),
    (r"\bram\b|\bmem(ory)?\b|\bsram\b", "memory"),
    (r"\barb(iter)?", "arbiter"),
    (r"\bsha\d*\b|\bhash\b", "crypto"),
    (r"\bmult(iplier)?", "multiplier"),
    (r"\bdiv(ider)?", "divider"),
    (r"\bshift(er)?", "shifter"),
    (r"\breg(ister)?\b|\bff\b|\bflop\b", "register"),
    (r"\bstate\b|\bfsm\b|\bstm\b", "fsm"),
    (r"\bpipe(line)?", "pipeline"),
    (r"\bcache\b", "cache"),
    (r"\baxi\b|\bapb\b|\bahb\b", "bus"),
    (r"\bserial(izer)?", "serializer"),
    (r"\bdeserial(izer)?", "deserializer"),
    (r"\bparity\b|\becc\b", "parity"),
    (r"\blfsr\b", "lfsr"),
    (r"\btimer\b", "timer"),
    (r"\bwatchdog\b", "watchdog"),
]

console = Console()


@dataclass
class VerilogChunk:
    """Chunk representing a Verilog module for embedding."""

    chunk_id: str
    ports: list[str]
    tags: list[str]
    chunk_type: str = "verilog_code"
    module_name: str = ""
    code: str = ""
    description: str = ""
    complexity: str = "complex"
    line_count: int = 0
    source: str = ""
    embedding_text: str = ""
    has_description: bool = False


def _clean_mg_verilog_description(desc: str) -> tuple[str, bool]:
    """
    Clean MG-Verilog descriptions that contain LLM prompt templates.
    Returns (cleaned_description, is_valid).
    Be LESS aggressive — keep anything that looks like a description.
    """
    if not desc or not desc.strip():
        return "", False

    text = desc.strip()

    # Remove XML-like tags first
    text = re.sub(r"<s>|</s>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[INST\]|\[/INST\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<<SYS>>|<<\/SYS>>|<</SYS>>", "", text, flags=re.IGNORECASE)

    # Extract content after instruction phrases (remove everything up to and including)
    for pattern in [
        r"Implement the Verilog module based on the following description\.?\s*",
        r"Complete the Verilog module based on the following description\.?\s*",
        r"Implement the Verilog module\.?\s*",
        r"Complete the Verilog module\.?\s*",
        r"based on the following description\.?\s*",
        r"following description\.?\s*",
        r"Description:\s*",
        r"following block[:\s]*",
    ]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[match.end() :].strip()
            break

    # Remove instruction boilerplate (but keep description content)
    text = re.sub(
        r"Do not include module, input and output definitions\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"End the Verilog module code completion with ['\"]?endmodule['\"]?\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"You only complete chats?\.?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Assume that signals are positive clock/clk edge triggered unless otherwise stated\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Assume that signals are [^.]+\.?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse whitespace, trim
    text = re.sub(r"\s+", " ", text).strip()

    # Valid if we have meaningful content (> 10 chars)
    if len(text) < 10:
        return "", False
    return text, True


def _extract_module_name(code: str) -> str | None:
    """Extract module name from code."""
    match = re.search(r"module\s+(\w+)", code, re.IGNORECASE)
    return match.group(1) if match else None


def _extract_ports(code: str) -> list[str]:
    """Extract input/output port names from code."""
    ports: list[str] = []
    # Match input/output declarations
    for match in re.finditer(
        r"(?:input|output|inout)\s+(?:wire|reg)?\s*(?:\[[^\]]+\])?\s*([a-zA-Z_]\w*)",
        code,
        re.IGNORECASE,
    ):
        ports.append(match.group(1))
    # Also match port lists: input a, b, output c
    for match in re.finditer(
        r"(?:input|output|inout)\s+(?:\[[^\]]+\])?\s*([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)",
        code,
        re.IGNORECASE,
    ):
        for name in re.split(r"\s*,\s*", match.group(1)):
            name = name.strip()
            if name and name not in ports:
                ports.append(name)
    return list(dict.fromkeys(ports))  # preserve order, dedupe


def _classify_complexity(code: str) -> str:
    """Classify module complexity based on code patterns."""
    code_lower = code.lower()
    has_always_posedge = bool(re.search(r"always\s*@\s*\(\s*posedge", code_lower))
    has_case = bool(re.search(r"\bcase[xz]?\s*\(", code_lower))
    has_state = bool(re.search(r"\b(state|next_state|current_state)\b", code_lower))
    has_instantiation = bool(re.search(r"\w+\s+#?\s*\(?\s*\.\w+\s*\(", code_lower))
    always_blocks = len(re.findall(r"\balways\s*@", code_lower))
    has_assign_only = bool(re.search(r"\bassign\s+", code_lower)) and not has_always_posedge
    has_always_star = bool(re.search(r"always\s*@\s*\(\s*\*\s*\)", code_lower))

    if has_case and (has_state or has_always_posedge):
        return "fsm"
    if has_instantiation or always_blocks >= 2:
        return "complex"
    if has_always_posedge and not has_case:
        return "sequential"
    if has_assign_only or (has_always_star and not has_always_posedge):
        return "combinational"
    return "complex"


def _extract_tags_from_code(code: str) -> set[str]:
    """Extract tags by scanning code content (not just module name)."""
    tags: set[str] = set()
    code_lower = code.lower()

    # FSM: case/casez with state-like variables
    if re.search(r"\bcase[xz]?\s*\(", code_lower) and re.search(
        r"\b(state|next_state|current_state|curr_state)\b", code_lower
    ):
        tags.add("fsm")

    # Sequential: always @(posedge or always_ff
    if re.search(r"always\s*@\s*\(\s*posedge|always_ff", code_lower):
        tags.add("sequential")

    # Combinational: only assign statements (no always)
    if re.search(r"\bassign\s+", code_lower) and not re.search(
        r"always\s*@", code_lower
    ):
        tags.add("combinational")

    # Hierarchical: instantiates other modules (pattern: name instance_name ( )
    # Exclude: function/task defs, always, if, case, initial
    inst_pattern = re.compile(
        r"\b(?!function|task|always|if|case|initial|for|while)\w+\s+\w+\s*\(",
        re.IGNORECASE,
    )
    if inst_pattern.search(code_lower):
        tags.add("hierarchical")

    # Parameterized
    if re.search(r"\bparameter\b", code_lower):
        tags.add("parameterized")

    # Content-based tags
    if re.search(r"\bfifo\b|\bqueue\b", code_lower):
        tags.add("fifo")
    if re.search(r"\bram\b|\bmemory\b|\bsram\b", code_lower):
        tags.add("memory")
    if re.search(r"\buart\b|\bserial\b", code_lower):
        tags.add("serial")
    if re.search(r"\baxi\b|\bapb\b|\bwishbone\b", code_lower):
        tags.add("bus")

    return tags


def _extract_tags(module_name: str, complexity: str, code: str = "") -> list[str]:
    """Extract tags from module name, complexity, and code content."""
    tags: set[str] = set()
    name_lower = module_name.lower()

    # Module name patterns
    for pattern, tag in TAG_PATTERNS:
        if re.search(pattern, name_lower):
            tags.add(tag)

    # Code content patterns (ensures nearly every module gets 1-2 tags)
    if code:
        tags.update(_extract_tags_from_code(code))

    # Always add complexity as a tag
    tags.add(complexity)

    return sorted(tags)


class VerilogChunker:
    """Process Verilog modules from all_modules.jsonl into chunks."""

    def __init__(self, modules_path: Path | str):
        self.modules_path = Path(modules_path)

    def process_all(self) -> list[VerilogChunk]:
        """Read JSONL, create VerilogChunk for each module. Returns all chunks."""
        chunks: list[VerilogChunk] = []
        if not self.modules_path.exists():
            console.print(f"[yellow]Modules file not found: {self.modules_path}[/yellow]")
            return chunks
        skip_reasons: Counter[str] = Counter()
        complexity_dist: Counter[str] = Counter()
        with_desc_count = 0

        with open(self.modules_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    skip_reasons["json_error"] += 1
                    continue

                code = data.get("code", "")
                desc_raw = data.get("description", "")
                source = data.get("source", "")
                has_desc = data.get("has_description", False)

                # Skip empty/short code
                if not code or len(code.strip()) < 20:
                    skip_reasons["code_too_short"] += 1
                    continue
                if "endmodule" not in code.lower():
                    skip_reasons["no_endmodule"] += 1
                    continue

                module_name = _extract_module_name(code)
                if not module_name:
                    skip_reasons["no_module_name"] += 1
                    continue

                # Clean description
                description = ""
                if has_desc and desc_raw:
                    desc_clean, valid = _clean_mg_verilog_description(desc_raw)
                    if valid:
                        description = desc_clean
                        has_desc = True
                        with_desc_count += 1
                    else:
                        has_desc = False

                ports = _extract_ports(code)
                complexity = _classify_complexity(code)
                tags = _extract_tags(module_name, complexity, code)
                line_count = len(code.splitlines())

                if has_desc and description:
                    embedding_text = (
                        f"Description: {description}\n"
                        f"Module: {module_name}\n"
                        f"Ports: {ports}\n"
                        f"Tags: {tags}\n"
                        f"Code:\n{code}"
                    )
                else:
                    embedding_text = (
                        f"Module: {module_name}\n"
                        f"Ports: {ports}\n"
                        f"Complexity: {complexity}\n"
                        f"Tags: {tags}\n"
                        f"Code:\n{code}"
                    )

                chunk = VerilogChunk(
                    chunk_id=data.get("module_id", ""),
                    chunk_type="verilog_code",
                    module_name=module_name,
                    code=code,
                    description=description,
                    ports=ports,
                    complexity=complexity,
                    tags=tags,
                    line_count=line_count,
                    source=source,
                    embedding_text=embedding_text,
                    has_description=bool(description),
                )
                chunks.append(chunk)
                complexity_dist[complexity] += 1

        # Tag frequency
        tag_freq: Counter[str] = Counter()
        for c in chunks:
            for t in c.tags:
                tag_freq[t] += 1

        # Print stats
        table = Table(title="Verilog Chunker Stats")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total chunks created", str(len(chunks)))
        table.add_row("Skipped (code too short)", str(skip_reasons.get("code_too_short", 0)))
        table.add_row("Skipped (no endmodule)", str(skip_reasons.get("no_endmodule", 0)))
        table.add_row("Skipped (no module name)", str(skip_reasons.get("no_module_name", 0)))
        other_skip = sum(skip_reasons.values()) - skip_reasons.get("code_too_short", 0) - skip_reasons.get("no_endmodule", 0) - skip_reasons.get("no_module_name", 0)
        table.add_row("Skipped (other)", str(max(0, other_skip)))
        table.add_row("With clean descriptions", str(with_desc_count))
        console.print(table)

        comp_tree = Tree("[bold]Complexity distribution[/bold]")
        for comp, count in complexity_dist.most_common():
            comp_tree.add(f"[cyan]{comp}[/cyan]: {count}")
        console.print(comp_tree)

        tag_tree = Tree("[bold]Top 20 tags[/bold]")
        for tag, count in tag_freq.most_common(20):
            tag_tree.add(f"[cyan]{tag}[/cyan]: {count}")
        console.print(tag_tree)

        return chunks
