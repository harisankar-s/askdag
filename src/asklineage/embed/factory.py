import openai

from asklineage.config import Settings
from asklineage.embed.base import EmbeddingProvider
from asklineage.embed.openai_compatible import OpenAICompatibleEmbedder


def get_embedder(settings: Settings) -> EmbeddingProvider | None:
    """None means no embedding model is configured; search falls back to keywords."""
    match settings.embed_provider:
        case "none":
            return None
        case "openai_compatible":
            client = openai.OpenAI(
                base_url=settings.embed_base_url,
                api_key=settings.embed_api_key.get_secret_value(),
                timeout=settings.embed_timeout_seconds,
                max_retries=1,
            )
            return OpenAICompatibleEmbedder(
                client,
                settings.embed_model,
                settings.embed_dimensions,
                document_prefix=settings.embed_document_prefix,
                query_prefix=settings.embed_query_prefix,
                batch_size=settings.embed_batch_size,
            )
