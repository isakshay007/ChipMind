# ChipMind Data Requirements

This document describes the data sources and pipeline used to build ChipMind's knowledge base and run evaluations.

## Overview

ChipMind uses two main data types:

1. **Verilog modules** — For RAG retrieval over code examples
2. **EDA documentation** — For reference documentation retrieval
3. **VerilogEval benchmark** — For evaluation (spec-to-RTL problems)

---

## Data Pipeline

The full pipeline is:

```
make download-data  →  make scrape-docs  →  make extract-modules  →  make chunk  →  make build-index
```

| Step | Command | Output |
|------|---------|--------|
| Download | `make download-data` | `data/raw/verilogdb/`, `data/raw/verilog-eval/` |
| Scrape docs | `make scrape-docs` | `data/raw/eda_docs/` |
| Extract modules | `make extract-modules` | `data/processed/all_modules.jsonl` |
| Chunk | `make chunk` | `data/processed/all_chunks.jsonl` |
| Build index | `make build-index` | `data/processed/indexes/` (FAISS + BM25) |

Or run everything: `make pipeline`

---

## HuggingFace Datasets

The following HuggingFace repositories are tried (in order):

| Repo ID | Description |
|---------|-------------|
| `GaTech-EIC/verilogdb` | VerilogDB (20K+ verified modules) |
| `FICS-LLM/VerilogDB` | VerilogDB alternative |
| `GaTech-EIC/MG-Verilog` | MG-Verilog (arrow format) |
| `shailja/Verilog_GitHub` | Verilog from GitHub |
| `davide221/verilog-raw-100k` | ~100K raw Verilog modules (parquet) |
| `dakies/nvlabs-verilogeval` | VerilogEval on HuggingFace |

**Note:** Some datasets may require a HuggingFace account or `huggingface-cli login` for gated access.

---

## GitHub Fallback

If HuggingFace download fails, the pipeline falls back to:

| Repository | Purpose |
|------------|---------|
| [NVlabs/verilog-eval](https://github.com/NVlabs/verilog-eval) | VerilogEval benchmark (spec-to-RTL) |
| [shailja-thakur/VGen](https://github.com/shailja-thakur/VGen) | Additional Verilog examples |

---

## VerilogEval Benchmark

- **Source:** [NVlabs/verilog-eval](https://github.com/NVlabs/verilog-eval) — `dataset_spec-to-rtl`
- **Size:** 156 problems
- **Format:** Per problem: `*_prompt.txt` (spec), `*_ref.sv` (reference), `*_test.sv` (testbench)
- **Pass criterion:** Simulation output contains `Mismatches: 0 in N samples`
- **Used for:** Benchmarking Pass@1, syntax rate, and agentic debug loop impact

---

## EDA Documentation

- **Source:** Scraped via `data/scripts/scrape_eda_docs.py`
- **Output:** `data/raw/eda_docs/`
- **Use:** Documentation chunks for hybrid RAG retrieval

---

## Minimal vs Full Setup

| Use Case | Data Needed |
|----------|-------------|
| **CLI / API without RAG** | None. Generation works but without retrieval. |
| **Full RAG** | `make pipeline` (requires disk space for raw + processed data) |
| **Evaluation only** | `make download-data` (gets VerilogEval). Index not required for eval. |
| **Quick test** | `make eval-quick` — uses existing VerilogEval if present |

---

## Disk Space (Approximate)

- Raw data: ~500 MB – 2 GB (depending on datasets)
- Processed chunks: ~100–500 MB
- FAISS index: ~50–200 MB (depends on chunk count)
- Total: ~1–3 GB for full pipeline

---

## Troubleshooting

- **"No datasets downloaded"** — Check network, HuggingFace token (`huggingface-cli login`), and repo availability.
- **"Index directory not found"** — Run `make pipeline` or at least `make chunk` then `make build-index`.
- **BM25 segfault** — Large corpora are capped at 50K docs. Reduce chunk count or split indexes.
