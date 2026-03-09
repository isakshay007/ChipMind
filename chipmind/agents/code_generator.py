"""Code generator agent: generates Verilog from spec + RAG context."""

import json
import re
from pathlib import Path

from groq import Groq

from chipmind.agents.state import ChipMindState
from chipmind.config import settings
from chipmind.retrieval.hybrid_retriever import HybridRetriever


class CodeGeneratorAgent:
    """LLM-powered agent that generates Verilog from spec + retrieved examples."""

    def __init__(self, retriever: HybridRetriever):
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env for the code generator."
            )
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL
        self.retriever = retriever
        prompts_dir = Path(__file__).parent / "prompts"
        self.gen_prompt = (prompts_dir / "code_generator.txt").read_text()
        self.debug_prompt = (prompts_dir / "debug_fix.txt").read_text()

    def generate(self, state: ChipMindState) -> dict:
        """Generate Verilog code from spec + RAG context."""
        spec = state.get("spec", {})
        if not spec:
            return {"generated_code": "", "retrieved_modules": []}

        # Step 1: Build retrieval query
        query = self._build_retrieval_query(spec)
        if not query or not query.strip():
            query = spec.get("module_name", "verilog") or "verilog"

        # Step 2: Retrieve similar modules
        try:
            retrieved = self.retriever.search_code(query, k=5)
        except Exception:
            retrieved = []

        # Step 3: Build prompt
        examples = self._format_examples(retrieved, max_examples=3)
        spec_str = json.dumps(spec, indent=2)
        prompt = self.gen_prompt.format(spec=spec_str, examples=examples or "No examples available.")

        # Step 4: Call Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
        except Exception as e:
            import traceback
            print(f"ERROR in code_generator.generate: {e}")
            traceback.print_exc()
            return {
                "generated_code": "",
                "retrieved_modules": retrieved,
                "total_tokens_used": state.get("total_tokens_used", 0),
                "error": str(e),
            }

        # Step 5: Clean response
        code = self._clean_verilog(raw)

        return {
            "generated_code": code,
            "retrieved_modules": retrieved,
            "total_tokens_used": state.get("total_tokens_used", 0) + tokens_used,
        }

    def debug_fix(self, state: ChipMindState) -> dict:
        """Fix errors in generated code using compiler feedback + RAG."""
        errors = state.get("errors", [])
        code = state.get("generated_code", "")
        spec = state.get("spec", {})

        if not errors or not code:
            return {"generated_code": code}

        # Step 1: Format errors
        error_str = self._format_errors(errors)

        # Step 2: Retrieve similar examples
        error_query = " ".join(
            e.get("message", "") if isinstance(e, dict) else getattr(e, "message", "")
            for e in errors[:3]
        )

        try:
            retrieved = self.retriever.search_code(error_query or "verilog error", k=3)
        except Exception:
            retrieved = []
        examples = self._format_examples(retrieved, max_examples=3)

        # Step 3: Build debug prompt
        spec_str = json.dumps(spec, indent=2)
        prompt = self.debug_prompt.format(
            spec=spec_str,
            code=code,
            errors=error_str,
            examples=examples or "No examples available.",
        )

        # Step 4: Call Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
        except Exception as e:
            import traceback
            print(f"ERROR in code_generator.debug_fix: {e}")
            traceback.print_exc()
            return {"generated_code": code, "debug_context": retrieved}

        # Step 5: Clean response
        fixed_code = self._clean_verilog(raw)

        return {
            "generated_code": fixed_code,
            "debug_context": retrieved,
            "total_tokens_used": state.get("total_tokens_used", 0) + tokens_used,
        }

    def _build_retrieval_query(self, spec: dict) -> str:
        """Build retrieval query from spec."""
        parts = [spec.get("description", "")]
        parts.append(spec.get("complexity_hint", ""))
        for inp in spec.get("inputs", []):
            parts.append(inp.get("name", ""))
        for out in spec.get("outputs", []):
            parts.append(out.get("name", ""))
        parts.extend(spec.get("constraints", []))
        return " ".join(p for p in parts if p)

    def _format_examples(self, modules: list[dict], max_examples: int = 3) -> str:
        """Format retrieved modules as prompt context."""
        formatted = []
        for i, m in enumerate(modules[:max_examples]):
            name = m.get("module_name", f"example_{i}")
            code = m.get("code", "")
            lines = code.split("\n")
            if len(lines) > 80:
                code = "\n".join(lines[:80]) + "\n// ... (truncated)"
            formatted.append(f"--- Example {i+1}: {name} ---\n{code}")
        return "\n\n".join(formatted)

    def _format_errors(self, errors: list) -> str:
        """Format errors for prompt. Handles both dict and CompilerError objects."""
        lines = []
        for e in errors:
            if isinstance(e, dict):
                line = e.get("line", "?")
                msg = e.get("message", "")
                error_type = e.get("error_type", "unknown")
            else:
                line = getattr(e, "line", "?")
                msg = getattr(e, "message", "")
                error_type = getattr(e, "error_type", "unknown")
            lines.append(f"Line {line}: [{error_type}] {msg}")
        return "\n".join(lines)

    def _clean_verilog(self, raw: str) -> str:
        """Extract clean Verilog from LLM response."""
        text = raw.strip()
        # Remove markdown code blocks
        if "```" in text:
            match = re.search(
                r"```(?:verilog|sv|systemverilog)?\s*\n(.*?)```",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                text = match.group(1).strip()
            else:
                text = re.sub(r"```(?:verilog|sv|systemverilog)?\s*", "", text, flags=re.IGNORECASE)
                text = text.replace("```", "").strip()

        # Extract module...endmodule block if there's extra text
        match = re.search(r"(module\s+\w+[\s\S]*?endmodule)", text, re.IGNORECASE)
        if match:
            text = match.group(1)

        return text.strip()
