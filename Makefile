.PHONY: setup install test clean download-data extract-modules scrape-docs chunk build-index

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
