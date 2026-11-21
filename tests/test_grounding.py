from collections.abc import Callable

import pytest
from lineage_examples import (
    EXAMPLE_LINEAGE,
    EXAMPLE_MAX_NAMED,
    EXAMPLE_ROOT,
    UPSTREAM_EXAMPLE_LINEAGE,
    UPSTREAM_EXAMPLE_ROOT,
)

from asklineage.answer.claims import Highlights, RelationClaim
from asklineage.answer.facts import build_facts
from asklineage.answer.generate import Violation
from asklineage.answer.grounding import check
from asklineage.answer.prompt import (
    EXAMPLE_FACTS,
    EXAMPLE_HIGHLIGHTS,
    UPSTREAM_EXAMPLE_FACTS,
    UPSTREAM_EXAMPLE_HIGHLIGHTS,
)

FACTS = build_facts(EXAMPLE_LINEAGE, EXAMPLE_ROOT, EXAMPLE_MAX_NAMED)
UPSTREAM_FACTS = build_facts(UPSTREAM_EXAMPLE_LINEAGE, UPSTREAM_EXAMPLE_ROOT, EXAMPLE_MAX_NAMED)

# Positions in EXAMPLE_HIGHLIGHTS.claims
INDIRECT, DIRECT = 0, 1


def edit(index: int, **fields) -> Callable[[list[RelationClaim]], None]:
    def apply(claims: list[RelationClaim]) -> None:
        claims[index] = claims[index].model_copy(update=fields)

    return apply


def rules(change: Callable[[list[RelationClaim]], None], **kwargs) -> set[str]:
    claims = list(EXAMPLE_HIGHLIGHTS.claims)
    change(claims)
    return {v.rule for v in check(Highlights(claims=claims), FACTS, EXAMPLE_LINEAGE, **kwargs)}


def text(index: int, sentence: str) -> Callable[[list[RelationClaim]], None]:
    return edit(index, text=sentence)


def test_example_lineages_produce_the_prompt_example_facts():
    assert FACTS == EXAMPLE_FACTS
    assert UPSTREAM_FACTS == UPSTREAM_EXAMPLE_FACTS


def test_the_prompt_examples_pass_their_own_check():
    assert check(EXAMPLE_HIGHLIGHTS, FACTS, EXAMPLE_LINEAGE) == ()
    assert check(UPSTREAM_EXAMPLE_HIGHLIGHTS, UPSTREAM_FACTS, UPSTREAM_EXAMPLE_LINEAGE) == ()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            edit(
                DIRECT,
                upstream="fct_orders",
                downstream="stg_orders",
                text="`stg_orders` reads directly from `fct_orders`.",
            ),
            "wrong_direction",
        ),
        (
            edit(
                DIRECT,
                upstream="int_order_items",
                downstream="mart_revenue",
                text="`mart_revenue` reads directly from `int_order_items`.",
            ),
            "no_such_edge",
        ),
        (
            edit(
                INDIRECT,
                upstream="fct_orders",
                downstream="mart_unit_costs",
                text="`mart_unit_costs` depends on `fct_orders` through other models.",
            ),
            "no_such_path",
        ),
        (
            edit(
                DIRECT,
                downstream="fct_refunds",
                text="`fct_refunds` reads directly from `stg_orders`.",
            ),
            "unknown_node",
        ),
        (
            text(DIRECT, "`fct_orders` reads directly from `stg_orders`, as `mart_weekly` does."),
            "name_not_in_claim",
        ),
        (text(DIRECT, "`fct_orders` feeds `stg_orders` directly."), "text_contradicts_direction"),
        (text(DIRECT, "fct_orders reads directly from `stg_orders`."), "unmarked_name"),
        (
            text(DIRECT, "`fct_orders` reads directly from `stg_orders` 3 times a day."),
            "unsupported_number",
        ),
    ],
)
def test_each_rule_catches_its_mistake(change, expected):
    assert expected in rules(change)


@pytest.mark.parametrize(
    "sentence",
    [
        # What qwen2.5:7b wrote on Cal-ITP (run 4); both owner statements were false.
        "`fct_orders`, with no owner recorded, reads directly from `stg_orders`.",
        "`fct_orders` reads directly from `stg_orders`, a source node with no owner recorded.",
        "`fct_orders`, owned by finance, reads directly from `stg_orders`.",
        "`fct_orders` reads directly from `stg_orders`, which the finance team maintains.",
        "`fct_orders` (`finance`) reads directly from `stg_orders`.",
    ],
)
def test_any_owner_wording_is_rejected(sentence):
    assert "owner_mentioned" in rules(text(DIRECT, sentence))


def test_own_as_an_ordinary_word_is_allowed():
    sentence = "`fct_orders` reads directly from its own staging model `stg_orders`."

    assert rules(text(DIRECT, sentence)) == set()


def test_project_names_without_backticks_are_caught_only_when_known():
    sentence = text(DIRECT, "`fct_orders` reads directly from `stg_orders`, like dim_calendar.")

    assert "unmarked_name" in rules(sentence, known_names={"dim_calendar"})
    assert "unmarked_name" not in rules(sentence)


def test_english_words_are_not_mistaken_for_node_names():
    sentence = text(DIRECT, "`fct_orders` reads the orders directly from `stg_orders`.")

    assert rules(sentence, known_names={"orders"}) == set()


def test_violations_say_which_claim_they_belong_to():
    assert str(Violation(3, "no_such_edge", "`a` does not feed `b`")) == (
        "claim 3: `a` does not feed `b`"
    )
    assert str(Violation(None, "invalid_output", "not JSON")) == "not JSON"
