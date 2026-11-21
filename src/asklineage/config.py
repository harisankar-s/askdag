from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from asklineage.llm.openai_compatible import ResponseFormat


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://asklineage:asklineage@localhost:5432/asklineage"

    llm_provider: Literal["none", "openai_compatible"] = "none"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    # Ollama ignores the key, but OpenAI-compatible clients refuse to start without one.
    llm_api_key: SecretStr = SecretStr("ollama")
    llm_timeout_seconds: float = 60.0
    llm_response_format: ResponseFormat = "json_schema"


@lru_cache
def get_settings() -> Settings:
    return Settings()
