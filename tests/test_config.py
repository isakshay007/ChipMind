"""Tests for chipmind config."""

import pytest

from chipmind.config import Settings


def test_settings_defaults():
    """Test that Settings loads with default values."""
    settings = Settings()
    assert settings.LLM_PROVIDER == "groq"
    assert settings.LLM_MODEL == "llama-3.3-70b-versatile"
    assert settings.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert settings.FAISS_INDEX_PATH == "data/processed/faiss_index"
    assert settings.BM25_INDEX_PATH == "data/processed/bm25_index.pkl"
    assert settings.LANGSMITH_PROJECT == "chipmind"
    assert settings.LANGSMITH_TRACING == "true"
