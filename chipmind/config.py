"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys (optional)
    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    TOGETHER_API_KEY: str | None = None
    NVIDIA_API_KEY: str | None = None

    # LLM Configuration
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Index Paths
    FAISS_INDEX_PATH: str = "data/processed/faiss_index"
    BM25_INDEX_PATH: str = "data/processed/bm25_index.pkl"

    # LangSmith
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "chipmind"
    LANGSMITH_TRACING: str = "true"

    def model_post_init(self, __context) -> None:
        """Sync LangSmith settings to os.environ so LangChain/LangGraph pick them up."""
        import os

        if self.LANGSMITH_API_KEY:
            os.environ.setdefault("LANGSMITH_API_KEY", self.LANGSMITH_API_KEY)
            os.environ.setdefault("LANGSMITH_PROJECT", self.LANGSMITH_PROJECT)
            os.environ.setdefault("LANGSMITH_TRACING", self.LANGSMITH_TRACING)


# Singleton - import this at app startup so LangSmith env vars are synced
settings = Settings()
