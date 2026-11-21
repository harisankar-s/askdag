"""The one call the answer layer makes to a language model: prompt in, validated object out."""

import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ValidationError


class LLMError(Exception):
    """The provider gave no usable reply: network, auth, rate limit, timeout or no replies left."""


class LLMOutputError(LLMError):
    """The provider replied, but the reply is not valid JSON for the requested schema."""

    def __init__(self, message: str, raw: str, detail: str) -> None:
        super().__init__(message)
        self.raw = raw
        self.detail = detail


@dataclass(frozen=True)
class Completion[T: BaseModel]:
    data: T
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int


class LLMProvider(Protocol):
    model: str

    def complete_json[T: BaseModel](
        self, *, system: str, prompt: str, output: type[T]
    ) -> Completion[T]: ...


# Small local models often wrap JSON in a Markdown fence even when told not to.
FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_output[T: BaseModel](raw: str, output: type[T]) -> T:
    match = FENCE.match(raw)
    text = match.group(1) if match else raw
    try:
        return output.model_validate_json(text)
    except ValidationError as exc:
        raise LLMOutputError(
            f"Reply does not match {output.__name__} ({exc.error_count()} error(s))",
            raw=raw,
            detail=str(exc),
        ) from exc
