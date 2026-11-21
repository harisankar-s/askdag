import pytest
from pydantic import BaseModel

from asklineage.config import Settings
from asklineage.llm.base import LLMError, LLMOutputError, parse_output
from asklineage.llm.scripted import ScriptedProvider


class Reply(BaseModel):
    summary: str
    count: int


def test_scripted_provider_returns_validated_object_and_records_call():
    provider = ScriptedProvider([{"summary": "ok", "count": 3}])

    completion = provider.complete_json(system="sys", prompt="facts", output=Reply)

    assert completion.data == Reply(summary="ok", count=3)
    assert completion.model == "scripted"
    assert [(c.system, c.prompt, c.output) for c in provider.calls] == [("sys", "facts", Reply)]


def test_replies_are_consumed_in_order():
    provider = ScriptedProvider([{"summary": "a", "count": 1}, {"summary": "b", "count": 2}])

    first = provider.complete_json(system="", prompt="", output=Reply)
    second = provider.complete_json(system="", prompt="", output=Reply)

    assert (first.data.summary, second.data.summary) == ("a", "b")


def test_markdown_fenced_json_is_accepted():
    assert parse_output('```json\n{"summary": "ok", "count": 1}\n```', Reply).count == 1


@pytest.mark.parametrize(
    "raw",
    ["not json at all", '{"summary": "ok"}', '{"summary": "ok", "count": "many"}'],
    ids=["invalid-json", "missing-field", "wrong-type"],
)
def test_reply_that_does_not_match_schema_raises_output_error(raw):
    provider = ScriptedProvider([raw])

    with pytest.raises(LLMOutputError) as exc_info:
        provider.complete_json(system="", prompt="", output=Reply)

    assert exc_info.value.raw == raw
    assert exc_info.value.detail


def test_scripted_error_is_raised():
    provider = ScriptedProvider([LLMError("timeout")])

    with pytest.raises(LLMError, match="timeout"):
        provider.complete_json(system="", prompt="", output=Reply)


def test_running_out_of_replies_is_an_llm_error():
    provider = ScriptedProvider([])

    with pytest.raises(LLMError, match="no replies left"):
        provider.complete_json(system="", prompt="", output=Reply)


def test_llm_is_off_by_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    assert Settings(_env_file=None).llm_provider == "none"
