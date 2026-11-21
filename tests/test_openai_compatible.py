import json
from collections.abc import Callable

import httpx2
import openai
import pytest
from pydantic import BaseModel, SecretStr

from asklineage.config import Settings
from asklineage.llm.base import LLMError, LLMOutputError
from asklineage.llm.factory import get_provider
from asklineage.llm.openai_compatible import OpenAICompatibleProvider

BASE_URL = "http://llm.test/v1"


class Reply(BaseModel):
    summary: str
    count: int


def completion_body(content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }


def make_provider(
    handler: Callable[[httpx2.Request], httpx2.Response], response_format="json_schema"
) -> tuple[OpenAICompatibleProvider, list[dict]]:
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
    return OpenAICompatibleProvider(client, "test-model", response_format), sent


def test_valid_reply_is_parsed_with_usage():
    provider, _ = make_provider(
        lambda _: httpx2.Response(200, json=completion_body('{"summary": "ok", "count": 4}'))
    )

    completion = provider.complete_json(system="sys", prompt="facts", output=Reply)

    assert completion.data == Reply(summary="ok", count=4)
    assert (completion.model, completion.input_tokens, completion.output_tokens) == (
        "test-model",
        120,
        30,
    )


def test_request_carries_schema_in_prompt_and_response_format():
    provider, sent = make_provider(
        lambda _: httpx2.Response(200, json=completion_body('{"summary": "ok", "count": 1}'))
    )

    provider.complete_json(system="Summarize.", prompt="facts here", output=Reply)

    body = sent[0]
    assert body["model"] == "test-model"
    assert body["temperature"] == 0
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"].startswith("Summarize.")
    assert '"count"' in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "facts here"}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "Reply"


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [("json_object", {"type": "json_object"}), ("off", None)],
)
def test_weaker_response_formats(response_format, expected):
    provider, sent = make_provider(
        lambda _: httpx2.Response(200, json=completion_body('{"summary": "ok", "count": 1}')),
        response_format,
    )

    provider.complete_json(system="", prompt="", output=Reply)

    assert sent[0].get("response_format") == expected


def test_reply_off_schema_raises_output_error():
    provider, _ = make_provider(
        lambda _: httpx2.Response(200, json=completion_body('{"summary": "ok"}'))
    )

    with pytest.raises(LLMOutputError):
        provider.complete_json(system="", prompt="", output=Reply)


def test_truncated_reply_raises_output_error():
    provider, _ = make_provider(
        lambda _: httpx2.Response(200, json=completion_body('{"summ', finish_reason="length"))
    )

    with pytest.raises(LLMOutputError, match="cut off"):
        provider.complete_json(system="", prompt="", output=Reply)


def test_missing_model_says_how_to_pull_it():
    provider, _ = make_provider(
        lambda _: httpx2.Response(404, json={"error": {"message": "model not found"}})
    )

    with pytest.raises(LLMError, match="ollama pull test-model"):
        provider.complete_json(system="", prompt="", output=Reply)


def test_unreachable_server_is_an_llm_error():
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    provider, _ = make_provider(refuse)

    with pytest.raises(LLMError, match="Cannot reach"):
        provider.complete_json(system="", prompt="", output=Reply)


def test_server_error_is_an_llm_error():
    provider, _ = make_provider(lambda _: httpx2.Response(500, json={"error": {"message": "boom"}}))

    with pytest.raises(LLMError, match="returned an error"):
        provider.complete_json(system="", prompt="", output=Reply)


def test_factory_returns_none_when_llm_is_off():
    assert get_provider(Settings(_env_file=None, llm_provider="none")) is None


def test_factory_builds_openai_compatible_provider():
    settings = Settings(
        _env_file=None,
        llm_provider="openai_compatible",
        llm_base_url=BASE_URL,
        llm_model="qwen2.5:7b",
        llm_api_key=SecretStr("k"),
    )

    provider = get_provider(settings)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "qwen2.5:7b"
