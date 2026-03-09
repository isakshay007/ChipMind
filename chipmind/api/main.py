"""FastAPI backend for ChipMind."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from chipmind.api.schemas import (
    CompileRequest,
    CompileResponse,
    CompileError,
    GenerateRequest,
    GenerateResponse,
    RetrievedModuleSummary,
    RetrieveRequest,
    RetrievedChunk,
    HealthResponse,
)
from chipmind.config import settings

# Module-level state (loaded on startup)
_graph = None
_index_size = 0


def _load_graph():
    """Load ChipMindGraph. Returns None if indexes missing."""
    global _graph, _index_size
    try:
        from chipmind.agents.graph import ChipMindGraph

        index_dir = "data/processed/indexes"
        if not Path(index_dir).exists():
            return None
        _graph = ChipMindGraph(index_dir=index_dir)
        try:
            with open("data/processed/all_chunks.jsonl") as f:
                _index_size = sum(1 for line in f if line.strip())
        except Exception:
            _index_size = 0
        return _graph
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load graph on startup."""
    global _graph
    _graph = _load_graph()
    yield
    _graph = None


app = FastAPI(
    title="ChipMind API",
    description="AI-powered Verilog design assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check with model and index info."""
    index_size = 0
    try:
        chunks_path = Path("data/processed/all_chunks.jsonl")
        if chunks_path.exists():
            with open(chunks_path) as f:
                index_size = sum(1 for line in f if line.strip())
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        model=settings.LLM_MODEL,
        index_size=index_size,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Run full ChipMind pipeline."""
    global _graph
    if _graph is None:
        raise HTTPException(
            status_code=503,
            detail="ChipMind graph not loaded. Run 'make build-index' first.",
        )
    try:
        result = _graph.run(query=req.query, max_iterations=req.max_iterations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build retrieved_modules summary (names + scores, no full code)
    retrieved = []
    for m in result.get("retrieved_modules", [])[:5]:
        name = m.get("module_name", "unknown")
        code = m.get("code", "")
        score = m.get("rrf_score")
        retrieved.append(
            RetrievedModuleSummary(
                module_name=name,
                score=float(score) if score is not None else None,
                code_preview=code[:200] if code else "",
            )
        )

    iterations = result.get("iteration", 0) or len(result.get("iteration_history", []))
    return GenerateResponse(
        final_code=result.get("final_code", result.get("generated_code", "")),
        status=result.get("final_status", "unknown"),
        iterations=iterations,
        iteration_history=result.get("iteration_history", []),
        spec=result.get("spec", {}),
        retrieved_modules=retrieved,
        metrics={
            "tokens": result.get("total_tokens_used", 0),
            "time_seconds": result.get("total_time_seconds", 0),
        },
    )


@app.post("/retrieve", response_model=list[RetrievedChunk])
def retrieve(req: RetrieveRequest):
    """Search knowledge base (code or docs)."""
    global _graph
    if _graph is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run 'make build-index' first.",
        )
    try:
        if req.type == "docs":
            results = _graph.retriever.search_docs(req.query, k=req.k)
        else:
            results = _graph.retriever.search_code(req.query, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    chunks = []
    for r in results:
        code = r.get("code", r.get("text", ""))
        chunks.append(
            RetrievedChunk(
                module_name=r.get("module_name"),
                section_title=r.get("section_title"),
                chunk_type=r.get("chunk_type", "unknown"),
                score=float(r.get("rrf_score", 0)) if r.get("rrf_score") is not None else None,
                code_preview=(code or "")[:200],
            )
        )
    return chunks


@app.post("/compile", response_model=CompileResponse)
def compile_design(req: CompileRequest):
    """Compile and optionally simulate Verilog code."""
    try:
        from chipmind.agents.compiler_gate import CompilerGate

        compiler = CompilerGate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compiler init failed: {e}")

    if not req.code.strip():
        return CompileResponse(
            compiled=False,
            errors=[CompileError(message="Empty code")],
            sim_output="",
            passed=False,
        )

    try:
        if req.testbench:
            sim_result = compiler.compile_and_simulate(req.code, req.testbench)
            errors = [
                CompileError(
                    file=e.file,
                    line=e.line,
                    message=e.message,
                    error_type=e.error_type,
                )
                for e in sim_result.compile_errors
            ]
            return CompileResponse(
                compiled=sim_result.compiled,
                errors=errors,
                sim_output=sim_result.sim_output or "",
                passed=sim_result.passed,
            )
        else:
            compile_result = compiler.compile(req.code)
            errors = [
                CompileError(
                    file=e.file,
                    line=e.line,
                    message=e.message,
                    error_type=e.error_type,
                )
                for e in compile_result.errors
            ]
            return CompileResponse(
                compiled=compile_result.success,
                errors=errors,
                sim_output="",
                passed=False,
            )
    except Exception as e:
        return CompileResponse(
            compiled=False,
            errors=[CompileError(message=str(e))],
            sim_output="",
            passed=False,
        )
