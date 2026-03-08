"""Spec analyzer agent: converts natural language to structured hardware spec."""

import json
from pathlib import Path
from typing import Any

from groq import Groq

from chipmind.agents.state import ChipMindState
from chipmind.config import settings


class SpecAnalyzerAgent:
    """LLM-powered agent that extracts structured spec from natural language."""

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env for the spec analyzer."
            )
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).parent / "prompts" / "spec_analyzer.txt"
        return prompt_path.read_text()

    def analyze(self, state: ChipMindState) -> dict:
        """Analyze user query and return updated state fields.

        LangGraph nodes return a dict of ONLY the fields they update,
        not the entire state.
        """
        user_query = state.get("user_query") or ""
        if not user_query.strip():
            return {"spec": self._minimal_spec(""), "total_tokens_used": 0}

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            raw = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
        except Exception as e:
            return {
                "spec": self._minimal_spec(user_query),
                "total_tokens_used": state.get("total_tokens_used", 0),
                "error": str(e),
            }

        try:
            spec = self._parse_json(raw)
        except json.JSONDecodeError:
            spec = self._retry_parse(state["user_query"])

        required = ["module_name", "inputs", "outputs", "functionality", "complexity_hint"]
        for field in required:
            if field not in spec:
                spec[field] = self._default_for_field(field, state["user_query"])

        return {
            "spec": spec,
            "total_tokens_used": state.get("total_tokens_used", 0) + tokens_used,
        }

    def _parse_json(self, raw: str) -> dict:
        """Parse JSON from LLM response, stripping markdown if present."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n", 1)
            cleaned = lines[1] if len(lines) > 1 else cleaned
            if "```" in cleaned:
                cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned.strip())

    def _retry_parse(self, query: str) -> dict:
        """Retry with explicit JSON instruction."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON generator. Return ONLY valid JSON. No text before or after.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate a Verilog module specification for: {query}\n\n"
                        "Required fields: module_name (str), description (str), "
                        "inputs (list of {name, width, description}), "
                        "outputs (list of {name, width, description}), "
                        "functionality (str), complexity_hint (str: combinational/sequential/fsm), "
                        "constraints (list of str)"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        try:
            raw = response.choices[0].message.content
            return self._parse_json(raw)
        except json.JSONDecodeError:
            return self._minimal_spec(query)

    def _minimal_spec(self, query: str) -> dict:
        """Last resort: minimal valid spec."""
        return {
            "module_name": "design",
            "description": query,
            "inputs": [{"name": "clk", "width": 1, "description": "Clock"}],
            "outputs": [{"name": "out", "width": 1, "description": "Output"}],
            "functionality": query,
            "complexity_hint": "combinational",
            "constraints": [],
        }

    def _default_for_field(self, field: str, query: str) -> Any:
        """Default value for missing required field."""
        defaults: dict[str, Any] = {
            "module_name": "design",
            "description": query,
            "inputs": [{"name": "in", "width": 1, "description": "Input"}],
            "outputs": [{"name": "out", "width": 1, "description": "Output"}],
            "functionality": query,
            "complexity_hint": "combinational",
            "constraints": [],
        }
        return defaults.get(field, "")
