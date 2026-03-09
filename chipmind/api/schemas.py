"""Pydantic schemas for ChipMind API."""

from pydantic import BaseModel, Field


# --- Generate ---
class GenerateRequest(BaseModel):
    query: str = Field(..., description="Natural language description of the design")
    max_iterations: int = Field(default=5, ge=1, le=20, description="Max debug loop iterations")


class RetrievedModuleSummary(BaseModel):
    module_name: str
    score: float | None = None
    code_preview: str = Field(..., max_length=200)


class GenerateResponse(BaseModel):
    final_code: str
    status: str
    iterations: int
    iteration_history: list
    spec: dict
    retrieved_modules: list[RetrievedModuleSummary]
    metrics: dict  # tokens, time_seconds


# --- Retrieve ---
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(default=5, ge=1, le=50, description="Number of results")
    type: str = Field(default="code", description="'code' or 'docs'")


class RetrievedChunk(BaseModel):
    module_name: str | None = None
    section_title: str | None = None
    chunk_type: str
    score: float | None = None
    code_preview: str = Field(default="", max_length=200)


# --- Compile ---
class CompileRequest(BaseModel):
    code: str = Field(..., description="Verilog design code")
    testbench: str | None = Field(default=None, description="Optional testbench for simulation")


class CompileError(BaseModel):
    file: str = ""
    line: int = 0
    message: str
    error_type: str = "other"


class CompileResponse(BaseModel):
    compiled: bool
    errors: list[CompileError]
    sim_output: str = ""
    passed: bool = False


# --- Health ---
class HealthResponse(BaseModel):
    status: str
    model: str
    index_size: int = 0
