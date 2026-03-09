# Contributing to ChipMind

Thank you for your interest in contributing to ChipMind! This document provides guidelines for contributing to the project.

## Owner & Access

**Akshay Keerthi AS** ([@isakshay007](https://github.com/isakshay007)) — Project author, code owner, and sole maintainer.

**⚠️ Main branch access:** Only **@isakshay007** can push to or merge into `main`. Contributors work on their own branches and submit Pull Requests. See [.github/GITHUB_SETUP.md](.github/GITHUB_SETUP.md) for setup details.

---

## Code of Conduct

Be respectful, constructive, and professional in all interactions.

---

## How to Contribute

### Reporting Bugs

- Use the [GitHub Issues](../../issues) bug report template.
- Include: Python version, OS, steps to reproduce, expected vs actual behavior.
- If relevant, attach error logs or screenshots.

### Suggesting Features

- Use the [feature request](../../issues/new?template=feature_request.md) template.
- Describe the use case and proposed solution clearly.

### Pull Requests (Required for all changes)

**You work on your branch. Never push to `main`.** All contributions go through Pull Requests. Only the owner (@isakshay007) reviews and merges PRs into `main`.

1. **Fork** the repository, then create a branch in your fork:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Set up the environment**:
   ```bash
   make setup
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   make install
   ```

3. **Run tests** before submitting:
   ```bash
   make test
   ```

4. **Follow style**:
   - Use 4 spaces for indentation.
   - Follow existing code style.
   - Add docstrings for new public functions/classes.

5. **Commit** with clear messages (e.g., `Add X`, `Fix Y`).

6. **Push to your fork** and open a Pull Request **from your branch to `main`**. Fill out the PR template. Only @isakshay007 can merge.

---

## Development Setup

### Prerequisites

- Python 3.10+
- [Icarus Verilog](http://iverilog.icarus.com/) (`brew install icarus-verilog` or `apt-get install iverilog`)
- Git

### Local Setup

**To use ChipMind (read-only):**
```bash
git clone https://github.com/isakshay007/ChipMind.git
cd ChipMind
make setup && source .venv/bin/activate && make install
cp .env.example .env  # Add GROQ_API_KEY or NVIDIA_API_KEY
```

**To contribute (fork first):**
```bash
# 1. Fork on GitHub, then clone YOUR fork
git clone https://github.com/YOUR_USERNAME/ChipMind.git
cd ChipMind
git remote add upstream https://github.com/isakshay007/ChipMind.git

# 2. Setup
make setup && source .venv/bin/activate && make install
cp .env.example .env
```

### Building the Knowledge Base (for RAG)

```bash
make pipeline
```

This runs: `download-data` → `scrape-docs` → `extract-modules` → `chunk` → `build-index`.

### Running Tests

```bash
make test
```

For quick smoke tests without indexes:

```bash
pytest tests/ -v -m "not slow"
```

---

## Project Structure

| Directory    | Purpose                                      |
|-------------|-----------------------------------------------|
| `chipmind/agents/`    | LangGraph nodes (spec, code gen, testbench, compiler, error classifier, debug) |
| `chipmind/retrieval/`| Hybrid RAG (FAISS + BM25)                     |
| `chipmind/ingestion/`| Data pipeline (chunking, indexing)           |
| `chipmind/evaluation/`| VerilogEval benchmark runner                 |
| `chipmind/api/`      | FastAPI backend                              |
| `frontend/`          | Streamlit dashboard                          |
| `data/`              | Raw and processed data                       |
| `tests/`             | Pytest test suite                            |

---

## Data Requirements

See [DATA.md](DATA.md) for details on datasets, HuggingFace repos, and data pipeline steps.

---

## Contact

For questions or discussions, open an [Issue](../../issues) or reach out to **Akshay Keerthi AS**.
