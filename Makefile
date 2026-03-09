.PHONY: setup install test clean download-data extract-modules scrape-docs chunk build-index pipeline eval-quick eval-full eval-resume analyze-rag api dashboard docker-build docker-run docker-stop cli

# Full pipeline: run in order before chunk + build-index + test
# download-data and scrape-docs can run in parallel (independent)
pipeline: download-data scrape-docs extract-modules chunk build-index
	@echo "Pipeline complete. Run 'make test' to run tests."

setup:
	python -m venv .venv
	@echo "Virtual environment created. Run 'source .venv/bin/activate' (or .venv\\Scripts\\activate on Windows) then 'make install'"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

download-data:
	python data/scripts/download_verilogdb.py
	python data/scripts/download_verilog_eval.py

extract-modules:
	python data/scripts/extract_verilog_modules.py

scrape-docs:
	python data/scripts/scrape_eda_docs.py

chunk:
	python -m chipmind.ingestion.ingest_all

build-index:
	python -m chipmind.ingestion.build_index

# VerilogEval benchmark
eval-quick:
	python -m chipmind.evaluation.run_eval --max-problems 10

eval-full:
	python -m chipmind.evaluation.run_eval

eval-resume:
	python -m chipmind.evaluation.run_eval --resume

analyze-rag:
	python -m chipmind.evaluation.analyze_rag

# API and Dashboard
api:
	uvicorn chipmind.api.main:app --reload --port 8000

dashboard:
	streamlit run frontend/app.py --server.port 8501

# Docker
docker-build:
	docker-compose build

docker-run:
	docker-compose up

docker-stop:
	docker-compose down

# CLI
cli:
	python -m chipmind.cli
