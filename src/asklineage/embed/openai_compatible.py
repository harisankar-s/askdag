"""Adapter for any server that speaks the OpenAI embeddings API: Ollama, vLLM, LiteLLM."""

import time
from collections.abc import Sequence

import openai

from asklineage.embed.base import EmbeddingError, Embeddings, Kind
from asklineage.model_span import model_span


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        dimensions: int,
        *,
        document_prefix: str = "",
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.document_prefix = document_prefix
        self._client = client
        self._prefixes: dict[str, str] = {"document": document_prefix, "query": query_prefix}
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str], *, kind: Kind) -> Embeddings:
        if any(not text.strip() for text in texts):
            raise ValueError("cannot embed an empty text")
        if not texts:
            return Embeddings([], self.model, None, 0)
        with model_span("embeddings", self.model, self._client) as span:
            span.set_attribute("asklineage.embed_kind", kind)
            span.set_attribute("asklineage.texts", len(texts))
            result = self._embed(texts, kind)
            if result.input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
            return result

    def _embed(self, texts: Sequence[str], kind: Kind) -> Embeddings:
        prefix = self._prefixes[kind]
        vectors: list[list[float]] = []
        tokens: int | None = 0
        latency_ms = 0
        model = self.model
        for start in range(0, len(texts), self._batch_size):
            batch = [prefix + text for text in texts[start : start + self._batch_size]]
            batch_vectors, batch_tokens, batch_ms, model = self._embed_batch(batch)
            vectors += batch_vectors
            latency_ms += batch_ms
            tokens = None if tokens is None or batch_tokens is None else tokens + batch_tokens
        return Embeddings(vectors, model, tokens, latency_ms)

    def _embed_batch(self, batch: list[str]) -> tuple[list[list[float]], int | None, int, str]:
        server = str(self._client.base_url)
        started = time.perf_counter()
        try:
            # float, not the client's base64 default: some compatible servers only send floats.
            response = self._client.embeddings.create(
                model=self.model, input=batch, encoding_format="float"
            )
        except openai.NotFoundError as exc:
            raise EmbeddingError(
                f"{server} has no model {self.model!r}. With Ollama: `ollama pull {self.model}`"
            ) from exc
        except openai.APIConnectionError as exc:
            raise EmbeddingError(f"Cannot reach {server} ({exc}). Is the server running?") from exc
        except openai.APIError as exc:
            raise EmbeddingError(f"{server} returned an error: {exc}") from exc
        latency_ms = round((time.perf_counter() - started) * 1000)

        if len(response.data) != len(batch):
            raise EmbeddingError(
                f"{server} returned {len(response.data)} vectors for {len(batch)} texts"
            )
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingError(
                    f"{self.model!r} returned {len(vector)} dimensions, but EMBED_DIMENSIONS is "
                    f"{self.dimensions}. Set EMBED_DIMENSIONS to match the model; the vector "
                    "table must then be rebuilt and every node embedded again."
                )
        usage = response.usage
        return vectors, usage.prompt_tokens if usage else None, latency_ms, response.model
