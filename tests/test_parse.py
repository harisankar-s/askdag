import pytest

from asklineage.answer.parse import (
    EXAMPLES,
    ParsedQuestion,
    PreviousQuestion,
    build_prompt,
    check_parse,
    clean_entity,
    entity_in_question,
    parse_question,
)
from asklineage.lineage.graph import Direction
from asklineage.llm.scripted import ScriptedProvider

QUESTION = "What breaks downstream if I change stg_orders?"
PREVIOUS = PreviousQuestion("stg_orders", Direction.DOWNSTREAM, 3)


def parsed(
    entity="stg_orders",
    direction=Direction.DOWNSTREAM,
    depth=None,
    is_lineage=True,
    same_node=False,
):
    return ParsedQuestion(
        is_lineage=is_lineage,
        same_node=same_node,
        entity=entity,
        direction=direction,
        depth=depth,
    )


def rules(violations) -> list[str]:
    return [violation.rule for violation in violations]


def test_a_name_copied_from_the_question_passes():
    assert check_parse(parsed(), QUESTION) == []


@pytest.mark.parametrize("question, previous, example", EXAMPLES)
def test_every_prompt_example_passes_its_own_check(question, previous, example):
    assert check_parse(example, question, previous) == []


def test_a_follow_up_about_the_previous_node_passes():
    follow_up = parsed(entity=None, direction=Direction.UPSTREAM, same_node=True)

    assert check_parse(follow_up, "And upstream?", PREVIOUS) == []


def test_a_follow_up_needs_a_previous_question():
    follow_up = parsed(entity=None, direction=Direction.UPSTREAM, same_node=True)

    assert rules(check_parse(follow_up, "And upstream?")) == ["no_previous_node"]


def test_a_follow_up_may_not_switch_to_another_node():
    follow_up = parsed(entity="fct_orders", same_node=True)

    assert rules(check_parse(follow_up, "And upstream?", PREVIOUS)) == ["same_node_entity"]


def test_a_new_node_in_a_follow_up_must_still_be_in_the_question():
    reply = parsed(entity="fct_orders", direction=None)

    assert rules(check_parse(reply, "What about mart_weekly?", PREVIOUS)) == [
        "entity_not_in_question"
    ]


def test_direction_may_be_left_to_the_previous_question():
    reply = parsed(entity="fct_orders", direction=None)

    assert check_parse(reply, "What about fct_orders?", PREVIOUS) == []
    assert rules(check_parse(reply, "What about fct_orders?")) == ["missing_direction"]


def test_the_prompt_says_whether_there_is_a_previous_question():
    assert "There is no previous question" in build_prompt(QUESTION)
    assert "`stg_orders` (downstream, 3 hops)" in build_prompt("And upstream?", PREVIOUS)


def test_a_name_not_in_the_question_is_rejected():
    assert rules(check_parse(parsed("fct_orders"), QUESTION)) == ["entity_not_in_question"]


def test_a_corrected_spelling_is_rejected():
    violations = check_parse(parsed("stg_orders"), "what depends on stg_ordrs?")

    assert rules(violations) == ["entity_not_in_question"]


@pytest.mark.parametrize(
    "entity, question",
    [
        ("dim_feeds", "What uses dim_feeds_v2?"),
        ("feeds", "What uses dim_feeds?"),
        ("payments", "What reads raw.payments?"),
        ("raw", "What reads raw.payments?"),
    ],
)
def test_part_of_a_name_is_not_the_name(entity, question):
    assert not entity_in_question(entity, question)


@pytest.mark.parametrize(
    "entity, question",
    [
        ("stg_orders", "What depends on stg_orders."),
        ("STG_ORDERS", "what depends on stg_orders"),
        ("raw.payments", "Who reads `raw.payments`?"),
        ("dim schedule feeds", "What does dim  schedule feeds depend on?"),
    ],
)
def test_whole_names_match_ignoring_case_spacing_and_punctuation(entity, question):
    assert entity_in_question(entity, question)


def test_quotes_and_backticks_are_stripped_from_the_entity():
    assert clean_entity(" `stg_orders` ") == "stg_orders"
    assert check_parse(parsed("`stg_orders`"), QUESTION) == []


def test_a_lineage_question_needs_an_entity_and_a_direction():
    violations = check_parse(parsed(entity=None, direction=None), QUESTION)

    assert rules(violations) == ["missing_entity", "missing_direction"]


def test_a_question_that_is_not_lineage_needs_nothing_else():
    reply = parsed(entity=None, direction=None, is_lineage=False)

    assert check_parse(reply, "What's the weather?") == []


def test_parse_retries_once_with_the_problem():
    bad = parsed("fct_orders").model_dump()
    good = parsed().model_dump()
    provider = ScriptedProvider([bad, good])

    generated = parse_question(provider, QUESTION)

    assert generated.output == parsed()
    assert len(generated.attempts) == 2
    assert QUESTION in provider.calls[0].prompt
    assert "'fct_orders' is not in the question" in provider.calls[1].prompt


def test_depth_out_of_range_fails_schema_validation_and_is_retried():
    too_deep = {**parsed().model_dump(), "depth": 99}
    provider = ScriptedProvider([too_deep, too_deep])

    generated = parse_question(provider, QUESTION)

    assert generated.output is None
    assert rules(generated.attempts[0].violations) == ["invalid_output"]
