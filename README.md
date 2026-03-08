# ChipMind

AI-powered chip design assistant.

## Pipeline (before chunks, indexes, tests)

Run in order:

```bash
make pipeline    # Full: download-data → scrape-docs → extract-modules → chunk → build-index
make test        # Run tests
```

Or step-by-step:

1. **make download-data** — Downloads Verilog datasets (HuggingFace, GitHub)
2. **make scrape-docs** — Scrapes EDA documentation
3. **make extract-modules** — Produces `all_modules.jsonl` from raw data
4. **make chunk** — Produces `all_chunks.jsonl` (requires extract-modules)
5. **make build-index** — Builds FAISS + BM25 indexes (requires chunk)
6. **make test** — Runs pytest (e2e tests need indexes, GROQ_API_KEY, iverilog)
