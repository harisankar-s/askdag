"""The one call the search layer makes to an embedding model: texts in, vectors out."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

# Many retrieval models embed questions and indexed text differently (task prefixes).
type Kind = Literal["document", "query"]


class EmbeddingError(Exception):
    """No usable vectors: network, auth, missing model, or a reply of the wrong shape."""


@dataclass(frozen=True)
class Embeddings:
    vectors: list[list[float]]
    model: str
    input_tokens: int | None
    latency_ms: int


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int
    document_prefix: str

    def embed(self, texts: Sequence[str], *, kind: Kind) -> Embeddings: ...


def index_name(embedder: EmbeddingProvider) -> str:
    """The tag stored with each vector: the model, plus the document prefix if any.

    A different prefix gives different vectors from the same model, so it must force a re-embed.
    """
    prefix = embedder.document_prefix.strip()
    return f"{embedder.model} [{prefix}]" if prefix else embedder.model


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norms == 0:
        raise ValueError("cosine is undefined for a zero vector")
    return sum(x * y for x, y in zip(a, b, strict=True)) / norms
