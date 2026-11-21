import json
from collections.abc import Callable

import httpx2
import openai
import pytest
from pydantic import SecretStr

from asklineage.config import Settings
from asklineage.embed.base import EmbeddingError, cosine
from asklineage.embed.factory import get_embedder
from asklineage.embed.hashing import HashingEmbedder
from asklineage.embed.openai_compatible import OpenAICompatibleEmbedder

BASE_URL = "http://embed.test/v1"
DIMS = 4


def vector_for(text: str, dims: int = DIMS) -> list[float]:
    return [float(len(text)), 1.0] + [0.0] * (dims - 2)


def embeddings_body(texts: list[str], dims: int = DIMS, reverse: bool = False) -> dict:
    data = [
        {"object": "embedding", "index": i, "embedding": vector_for(text, dims)}
        for i, text in enumerate(texts)
    ]
    return {
        "object": "list",
        "data": data[::-1] if reverse else data,
        "model": "test-embed",
        "usage": {"prompt_tokens": 10 * len(texts), "total_tokens": 10 * len(texts)},
    }


def echo(request: httpx2.Request, **kwargs) -> httpx2.Response:
    return httpx2.Response(
        200, json=embeddings_body(json.loads(request.content)["input"], **kwargs)
    )


def make_embedder(
    handler: Callable[[httpx2.Request], httpx2.Response] = echo, batch_size: int = 32
) -> tuple[OpenAICompatibleEmbedder, list[dict]]:
    sent: list[dict] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return handler(request)

    client = openai.OpenAI(
        base_url=BASE_URL,
        api_key="test",
        max_retries=0,
        http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(record)),
    )
    embedder = OpenAICompatibleEmbedder(
        client,
        "test-embed",
        DIMS,
        document_prefix="doc: ",
        query_prefix="q: ",
        batch_size=batch_size,
    )
    return embedder, sent


@pytest.mark.parametrize(("kind", "prefix"), [("document", "doc: "), ("query", "q: ")])
def test_prefix_follows_kind(kind, prefix):
    embedder, sent = make_embedder()

    embedder.embed(["stops"], kind=kind)

    assert sent[0]["input"] == [f"{prefix}stops"]
    assert sent[0]["model"] == "test-embed"
    assert sent[0]["encoding_format"] == "float"


def test_vectors_come_back_in_input_order_with_usage():
    embedder, _ = make_embedder(lambda request: echo(request, reverse=True))

    result = embedder.embed(["a", "bbb"], kind="document")

    assert result.vectors == [vector_for("doc: a"), vector_for("doc: bbb")]
    assert (result.model, result.input_tokens) == ("test-embed", 20)


def test_texts_are_sent_in_batches():
    embedder, sent = make_embedder(batch_size=2)

    result = embedder.embed(["a", "b", "c", "d", "e"], kind="document")

    assert [len(body["input"]) for body in sent] == [2, 2, 1]
    assert len(result.vectors) == 5
    assert result.input_tokens == 50


def test_no_texts_makes_no_request():
    embedder, sent = make_embedder()

    result = embedder.embed([], kind="query")

    assert (result.vectors, sent) == ([], [])


def test_empty_text_is_refused():
    embedder, sent = make_embedder()

    with pytest.raises(ValueError, match="empty"):
        embedder.embed(["ok", "  "], kind="document")
    assert sent == []


def test_wrong_dimension_names_the_setting():
    embedder, _ = make_embedder(lambda request: echo(request, dims=DIMS + 2))

    with pytest.raises(EmbeddingError, match="EMBED_DIMENSIONS"):
        embedder.embed(["a"], kind="document")


def test_wrong_vector_count_is_an_error():
    embedder, _ = make_embedder(lambda _: httpx2.Response(200, json=embeddings_body(["a"])))

    with pytest.raises(EmbeddingError, match="1 vectors for 2 texts"):
        embedder.embed(["a", "b"], kind="document")


def test_missing_model_says_how_to_pull_it():
    embedder, _ = make_embedder(
        lambda _: httpx2.Response(404, json={"error": {"message": "model not found"}})
    )

    with pytest.raises(EmbeddingError, match="ollama pull test-embed"):
        embedder.embed(["a"], kind="document")


def test_unreachable_server_is_an_embedding_error():
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    embedder, _ = make_embedder(refuse)

    with pytest.raises(EmbeddingError, match="Cannot reach"):
        embedder.embed(["a"], kind="query")


def test_server_error_is_an_embedding_error():
    embedder, _ = make_embedder(lambda _: httpx2.Response(500, json={"error": {"message": "x"}}))

    with pytest.raises(EmbeddingError, match="returned an error"):
        embedder.embed(["a"], kind="query")


def test_hashing_embedder_is_deterministic_and_unit_length():
    embedder = HashingEmbedder(64)

    first = embedder.embed(["dim_schedule_feeds"], kind="document").vectors[0]
    again = embedder.embed(["dim_schedule_feeds"], kind="query").vectors[0]

    assert first == again
    assert len(first) == 64
    assert sum(x * x for x in first) == pytest.approx(1.0)


def test_hashing_embedder_ranks_shared_words_higher():
    embedder = HashingEmbedder()
    query, stops, fares = embedder.embed(
        ["bus stop locations", "dim_stops: bus stop locations by feed", "fare payments by card"],
        kind="document",
    ).vectors

    assert cosine(query, stops) > cosine(query, fares)


def test_hashing_embedder_refuses_text_without_words():
    with pytest.raises(ValueError, match="no words"):
        HashingEmbedder().embed(["--"], kind="document")


def test_cosine_of_zero_vector_is_refused():
    with pytest.raises(ValueError):
        cosine([0.0, 0.0], [1.0, 0.0])


def test_factory_returns_none_when_embeddings_are_off():
    assert get_embedder(Settings(_env_file=None, embed_provider="none")) is None


def test_factory_builds_openai_compatible_embedder_with_nomic_defaults():
    settings = Settings(
        _env_file=None,
        embed_provider="openai_compatible",
        embed_base_url=BASE_URL,
        embed_api_key=SecretStr("k"),
    )

    embedder = get_embedder(settings)

    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert (embedder.model, embedder.dimensions) == ("nomic-embed-text", 768)
    assert embedder._prefixes == {"document": "search_document: ", "query": "search_query: "}
