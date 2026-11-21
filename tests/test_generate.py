from pydantic import BaseModel

from asklineage.answer.generate import Violation, generate_checked
from asklineage.llm.base import LLMError
from asklineage.llm.scripted import ScriptedProvider


class Reply(BaseModel):
    value: int


def must_be_five(reply: Reply) -> list[Violation]:
    return (
        [] if reply.value == 5 else [Violation(1, "wrong_value", f"value is 5, not {reply.value}")]
    )


def run(replies):
    provider = ScriptedProvider(replies, model="m")
    generated = generate_checked(
        provider, system="sys", prompt="facts", output=Reply, check=must_be_five
    )
    return provider, generated


def test_a_passing_reply_is_returned_after_one_call():
    provider, generated = run([{"value": 5}])

    assert generated.output == Reply(value=5)
    assert len(generated.attempts) == 1
    assert generated.attempts[0].passed
    assert [call.system for call in provider.calls] == ["sys"]


def test_a_failing_reply_is_retried_with_the_reply_and_the_problems():
    provider, generated = run([{"value": 6}, {"value": 5}])

    assert generated.output == Reply(value=5)
    retry = provider.calls[1].prompt
    assert retry.startswith("facts")
    assert 'Your previous reply was:\n{"value":6}' in retry
    assert "claim 1: value is 5, not 6" in retry
    assert "keep the claims that had no problems" in retry


def test_two_failures_give_no_output_and_both_attempts():
    _, generated = run([{"value": 6}, {"value": 7}])

    assert generated.output is None
    assert [a.violations[0].detail for a in generated.attempts] == [
        "value is 5, not 6",
        "value is 5, not 7",
    ]


def test_invalid_json_counts_as_a_failed_attempt_and_is_shown_on_retry():
    provider, generated = run(["not json", {"value": 5}])

    assert generated.output == Reply(value=5)
    assert generated.attempts[0].violations[0].rule == "invalid_output"
    assert "Your previous reply was:\nnot json" in provider.calls[1].prompt


def test_provider_errors_are_not_retried():
    provider, generated = run([LLMError("Cannot reach the server"), {"value": 5}])

    assert generated.output is None
    assert generated.attempts[0].error == "Cannot reach the server"
    assert len(provider.calls) == 1
