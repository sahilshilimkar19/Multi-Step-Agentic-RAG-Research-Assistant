"""Runtime configuration loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    llm_provider: str = Field(default="anthropic", alias="LLM_PROVIDER")
    planner_model: str = Field(default="claude-sonnet-4-5", alias="PLANNER_MODEL")
    grader_model: str = Field(default="gpt-4o-mini", alias="GRADER_MODEL")
    synthesizer_model: str = Field(default="claude-sonnet-4-5", alias="SYNTHESIZER_MODEL")

    max_iterations: int = Field(default=4, alias="MAX_ITERATIONS")
    min_relevant_docs: int = Field(default=5, alias="MIN_RELEVANT_DOCS")
    relevance_threshold: float = Field(default=0.6, alias="RELEVANCE_THRESHOLD")

    checkpoint_db: str = Field(default="./checkpoints.sqlite", alias="CHECKPOINT_DB")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=False, alias="LOG_JSON")


@lru_cache
def get_settings() -> Settings:
    """Return the singleton Settings instance, loading from .env on first call."""
    return Settings()
