from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from asklineage.llm.openai_compatible import ResponseFormat
from asklineage.search.scan import Scan


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
    # How many node names the summary prompt may list; counts always cover every node.
    llm_max_named_nodes: int = Field(default=20, ge=1, le=500)

    embed_provider: Literal["none", "openai_compatible"] = "none"
    embed_base_url: str = "http://localhost:11434/v1"
    embed_model: str = "nomic-embed-text"
    embed_api_key: SecretStr = SecretStr("ollama")
    embed_timeout_seconds: float = 60.0
    # Must match the vector column; pgvector cannot index more than 2,000 dimensions.
    embed_dimensions: int = Field(default=768, ge=1, le=2000)
    embed_batch_size: int = Field(default=32, ge=1, le=2048)
    # nomic-embed-text's task prefixes; set both to "" for models without them.
    embed_document_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    # Path patterns of the models users should query; a matching staging model or source lifts
    # them in semantic search. [] turns lineage-aware ranking off.
    search_preferred_paths: list[str] = ["models/mart/*", "models/marts/*"]
    # A semantic top score below this is flagged low confidence (ADR 0012: 0.667 caught 3 of 5
    # unanswerable Cal-ITP questions and blocked 1 of 30). A flag, not an abstain.
    search_low_confidence_below: float = Field(default=0.667, ge=0, le=1)
    # How vector queries find the nearest chunks (search/scan.py, ADR 0018): topn matches exact
    # on Cal-ITP at half the latency; switch to hnsw near 10k chunks, after `eval vector-scale`.
    search_vector_scan: Scan = "topn"
    # topn and hnsw fetch k x this many chunks before keeping each node's or document's best.
    search_overfetch: int = Field(default=4, ge=1, le=50)
    # HNSW's candidate list (ADR 0018: 100 kept recall at 0.98 on 100k padded chunks).
    search_hnsw_ef_search: int = Field(default=100, ge=1, le=1000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
