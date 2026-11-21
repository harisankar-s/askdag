import openai

from asklineage.config import Settings
from asklineage.llm.base import LLMProvider
from asklineage.llm.openai_compatible import OpenAICompatibleProvider


def get_provider(settings: Settings) -> LLMProvider | None:
    """None means no LLM is configured; callers use the template summary instead."""
    match settings.llm_provider:
        case "none":
            return None
        case "openai_compatible":
            client = openai.OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key.get_secret_value(),
                timeout=settings.llm_timeout_seconds,
                # A slow local model can use most of the timeout; two retries would triple it.
                max_retries=1,
            )
            return OpenAICompatibleProvider(
                client, settings.llm_model, settings.llm_response_format
            )
