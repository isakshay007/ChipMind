"""Testbench generator agent: generates Verilog testbench from spec + design code."""

import json
import re
from pathlib import Path

from groq import Groq

from chipmind.agents.state import ChipMindState
from chipmind.config import settings


class TestbenchGeneratorAgent:
    """LLM-powered agent that generates Verilog testbench from spec and design."""

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env for the testbench generator."
            )
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL
        prompts_dir = Path(__file__).parent / "prompts"
        self.prompt = (prompts_dir / "testbench_generator.txt").read_text()

    def generate(self, state: ChipMindState) -> dict:
        """Generate Verilog testbench from spec and generated_code."""
        spec = state.get("spec", {})
        code = state.get("generated_code", "")

        if not spec or not code:
            return {"generated_testbench": "", "total_tokens_used": 0}

        module_name = spec.get("module_name", "design")
        spec_str = json.dumps(spec, indent=2)
        prompt = self.prompt.format(
            spec=spec_str,
            code=code,
            module_name=module_name,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
        except Exception:
            return {"generated_testbench": "", "total_tokens_used": state.get("total_tokens_used", 0)}

        tb = self._clean_verilog(raw)

        return {
            "generated_testbench": tb,
            "total_tokens_used": state.get("total_tokens_used", 0) + tokens_used,
        }

    def _clean_verilog(self, raw: str) -> str:
        """Extract clean Verilog from LLM response."""
        text = raw.strip()
        if "```" in text:
            match = re.search(
                r"```(?:verilog|sv|systemverilog)?\s*\n(.*?)```",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                text = match.group(1).strip()
            else:
                text = re.sub(
                    r"```(?:verilog|sv|systemverilog)?\s*", "", text, flags=re.IGNORECASE
                )
                text = text.replace("```", "").strip()

        match = re.search(r"(module\s+\w+[\s\S]*?endmodule)", text, re.IGNORECASE)
        if match:
            text = match.group(1)

        return text.strip()
