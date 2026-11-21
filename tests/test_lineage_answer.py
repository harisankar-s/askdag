from dataclasses import replace

from lineage_examples import (
    EXAMPLE_LINEAGE,
    EXAMPLE_MAX_NAMED,
    EXAMPLE_ROOT,
    UPSTREAM_EXAMPLE_LINEAGE,
    UPSTREAM_EXAMPLE_ROOT,
)

from asklineage.answer.lineage_answer import HighlightStatus, answer_lineage, tag_owners
from asklineage.answer.prompt import EXAMPLE_HIGHLIGHTS, UPSTREAM_EXAMPLE_HIGHLIGHTS, system_prompt
from asklineage.answer.template import template_summary
from asklineage.lineage.graph import Direction
from asklineage.llm.base import LLMError
from asklineage.llm.scripted import ScriptedProvider

GOOD = EXAMPLE_HIGHLIGHTS.model_dump()
BAD = {
    "claims": [
        {
            "upstream": "int_order_items",
            "downstream": "mart_revenue",
            "hops": "direct",
            "text": "`mart_revenue` reads directly from `int_order_items`.",
        }
    ]
}


def ask(provider, lineage=EXAMPLE_LINEAGE, root=EXAMPLE_ROOT):
    return answer_lineage(lineage, root, provider, max_named=EXAMPLE_MAX_NAMED)


def test_summary_always_comes_from_the_template():
    for provider in (None, ScriptedProvider([GOOD]), ScriptedProvider([BAD, BAD])):
        answer = ask(provider)

        assert answer.summary == template_summary(answer.facts)
        assert answer.lineage is EXAMPLE_LINEAGE


def test_checked_highlights_are_added():
    provider = ScriptedProvider([GOOD])

    answer = ask(provider)

    assert answer.status is HighlightStatus.PASSED
    assert answer.highlights == tuple(EXAMPLE_HIGHLIGHTS.claims)
    assert provider.calls[0].system == system_prompt(Direction.DOWNSTREAM)


def test_highlights_that_pass_on_retry_are_labelled():
    answer = ask(ScriptedProvider([BAD, GOOD]))

    assert answer.status is HighlightStatus.PASSED_AFTER_RETRY
    assert answer.attempts[0].violations[0].rule == "no_such_edge"


def test_rejected_highlights_are_dropped_and_the_facts_remain():
    answer = ask(ScriptedProvider([BAD, BAD]))

    assert answer.status is HighlightStatus.REJECTED
    assert answer.highlights == ()
    assert len(answer.attempts) == 2


def test_an_unreachable_llm_drops_the_highlights():
    answer = ask(ScriptedProvider([LLMError("Cannot reach the server")]))

    assert answer.status is HighlightStatus.REJECTED
    assert answer.attempts[0].error == "Cannot reach the server"


def test_no_provider_means_highlights_are_off():
    answer = ask(None)

    assert answer.status is HighlightStatus.OFF
    assert answer.attempts == ()


def test_empty_result_never_calls_the_llm():
    provider = ScriptedProvider([GOOD])
    empty = replace(EXAMPLE_LINEAGE, nodes=(), edges=(), truncated=False)

    answer = ask(provider, lineage=empty)

    assert answer.status is HighlightStatus.SKIPPED
    assert answer.summary == "Nothing reads from `stg_orders`."
    assert provider.calls == []


def test_upstream_question_gets_the_upstream_example():
    provider = ScriptedProvider([UPSTREAM_EXAMPLE_HIGHLIGHTS.model_dump()])

    answer = ask(provider, UPSTREAM_EXAMPLE_LINEAGE, UPSTREAM_EXAMPLE_ROOT)

    assert answer.status is HighlightStatus.PASSED
    assert provider.calls[0].system == system_prompt(Direction.UPSTREAM)


def test_highlights_get_owner_tags_from_the_result():
    answer = ask(ScriptedProvider([GOOD]))

    assert answer.highlight_texts == (
        "`mart_weekly` (no owner recorded) depends on `stg_orders` through other models.",
        "`fct_orders` (finance) reads directly from `stg_orders`.",
    )


def test_owner_tags_go_on_the_first_mention_only_and_skip_unknown_names():
    owners = {"a": "team-a", "b": None}

    assert tag_owners("`a` feeds `b`, and `a` feeds `root`.", owners) == (
        "`a` (team-a) feeds `b` (no owner recorded), and `a` feeds `root`."
    )
