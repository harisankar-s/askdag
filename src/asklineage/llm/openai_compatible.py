"""Adapter for any server that speaks the OpenAI chat-completions API: Ollama, vLLM, LiteLLM,
and Anthropic's OpenAI-compatible endpoint (with LLM_RESPONSE_FORMAT=off and
LLM_TEMPERATURE=null: it accepts only strict schemas, and newer models reject temperature)."""

import json
import time
from typing import Any, Literal

import openai
from pydantic import BaseModel

from asklineage.llm.base import Completion, LLMError, LLMOutputError, parse_output

# json_schema: the server constrains output to the schema (Ollama, OpenAI, vLLM).
# json_object: the server only guarantees some JSON. off: the server is sent no format hint.
ResponseFormat = Literal["json_schema", "json_object", "off"]

SCHEMA_INSTRUCTION = "Reply with one JSON object that matches this JSON Schema, and nothing else:"


class OpenAICompatibleProvider:
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        response_format: ResponseFormat = "json_schema",
        temperature: float | None = 0.0,
    ) -> None:
        """temperature None leaves it out, for models that reject it."""
        self.model = model
        self._client = client
        self._response_format = response_format
        self._temperature = temperature

    def complete_json[T: BaseModel](
        self, *, system: str, prompt: str, output: type[T]
    ) -> Completion[T]:
        schema = output.model_json_schema()
        # The schema also goes in the prompt, so servers that ignore response_format still see it.
        instructions = f"{system}\n\n{SCHEMA_INSTRUCTION}\n{json.dumps(schema)}"
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._response_format == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": output.__name__, "schema": schema},
            }
        elif self._response_format == "json_object":
            request["response_format"] = {"type": "json_object"}

        server = str(self._client.base_url)
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except openai.NotFoundError as exc:
            raise LLMError(
                f"{server} has no model {self.model!r}. With Ollama: `ollama pull {self.model}`"
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"Cannot reach {server} ({exc}). Is the server running?") from exc
        except openai.APIError as exc:
            raise LLMError(f"{server} returned an error: {exc}") from exc
        latency_ms = round((time.perf_counter() - started) * 1000)

        if not response.choices:
            raise LLMError(f"{server} returned no choices")
        choice = response.choices[0]
        raw = choice.message.content or ""
        if choice.finish_reason == "length":
            raise LLMOutputError(
                "Reply was cut off at the token limit", raw, "finish_reason=length"
            )
        usage = response.usage
        return Completion(
            data=parse_output(raw, output),
            model=response.model or self.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_ms=latency_ms,
        )
