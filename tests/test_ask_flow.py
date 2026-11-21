import pytest
from helpers import TEST_PROJECT, make_node

from asklineage.answer.lineage_answer import HighlightStatus
from asklineage.api.flow import run_ask
from asklineage.api.models import (
    AbstainResponse,
    AskRequest,
    ClarificationResponse,
    LineageResponse,
)
from asklineage.lineage.graph import Direction
from asklineage.llm.base import LLMError
from asklineage.llm.scripted import ScriptedProvider

RAW_ORDERS = make_node("orders", "source", source_name="raw")
ORDERS = make_node("orders")
STG_ORDERS = make_node("stg_orders", owner="finance")
FCT_ORDERS = make_node("fct_orders", owner="finance")
MART_WEEKLY = make_node("mart_weekly")

QUESTION = "What breaks if I change stg_orders?"
PARSE = {"is_lineage": True, "entity": "stg_orders", "direction": "downstream", "depth": None}
HIGHLIGHTS = {
    "claims": [
        {
            "upstream": "stg_orders",
            "downstream": "fct_orders",
            "hops": "direct",
            "text": "`fct_orders` reads directly from `stg_orders`.",
        }
    ]
}


@pytest.fixture(autouse=True)
def graph(load_graph) -> None:
    load_graph(
        [RAW_ORDERS, ORDERS, STG_ORDERS, FCT_ORDERS, MART_WEEKLY],
        [
            (RAW_ORDERS.unique_id, STG_ORDERS.unique_id),
            (STG_ORDERS.unique_id, FCT_ORDERS.unique_id),
            (FCT_ORDERS.unique_id, MART_WEEKLY.unique_id),
        ],
    )


def ask(conn, provider=None, **fields):
    return run_ask(conn, provider, AskRequest(project=TEST_PROJECT, **fields), max_named=20)


def stages(response) -> list[str]:
    return [attempt.stage for attempt in response.attempts]


def test_structured_fields_answer_without_an_llm(conn):
    response = ask(conn, node="stg_orders")

    assert isinstance(response, LineageResponse)
    assert response.node.unique_id == STG_ORDERS.unique_id
    assert (response.direction, response.depth) == (Direction.DOWNSTREAM, 5)
    assert [row.display_name for row in response.table] == ["fct_orders", "mart_weekly"]
    assert response.summary.startswith("2 nodes depend on `stg_orders`")
    assert response.facts.total == 2
    assert response.highlight_status is HighlightStatus.OFF
    assert response.attempts == []


def test_structured_depth_and_direction_are_used_and_truncation_reported(conn):
    response = ask(conn, node="fct_orders", direction="upstream", depth=1)

    assert [row.display_name for row in response.table] == ["stg_orders"]
    assert response.facts.truncated


def test_a_free_text_question_is_parsed_resolved_and_answered(conn):
    provider = ScriptedProvider([PARSE, HIGHLIGHTS])

    response = ask(conn, provider, question=QUESTION)

    assert isinstance(response, LineageResponse)
    assert response.node.unique_id == STG_ORDERS.unique_id
    assert response.highlight_status is HighlightStatus.PASSED
    assert response.highlights[0].text == "`fct_orders` (finance) reads directly from `stg_orders`."
    assert stages(response) == ["parse", "highlights"]


def test_parsed_direction_and_depth_are_used(conn):
    parse = {**PARSE, "entity": "fct_orders", "direction": "upstream", "depth": 1}
    provider = ScriptedProvider([parse, LLMError("down")])

    response = ask(conn, provider, question="What feeds fct_orders, one level up?")

    assert (response.direction, response.depth) == (Direction.UPSTREAM, 1)
    assert [row.display_name for row in response.table] == ["stg_orders"]
    assert response.highlight_status is HighlightStatus.REJECTED


def test_a_question_that_is_not_lineage_abstains(conn):
    provider = ScriptedProvider(
        [{"is_lineage": False, "entity": None, "direction": None, "depth": None}]
    )

    response = ask(conn, provider, question="Who owns fct_orders?")

    assert isinstance(response, AbstainResponse)
    assert response.reason == "not_lineage"
    assert len(provider.calls) == 1


def test_free_text_without_an_llm_abstains(conn):
    response = ask(conn, question=QUESTION)

    assert isinstance(response, AbstainResponse)
    assert response.reason == "llm_off"


def test_a_question_that_fails_the_check_twice_abstains(conn):
    guessed = {**PARSE, "entity": "fct_orders"}
    response = ask(conn, ScriptedProvider([guessed, guessed]), question=QUESTION)

    assert isinstance(response, AbstainResponse)
    assert response.reason == "not_understood"
    assert stages(response) == ["parse", "parse"]


def test_an_unreachable_llm_abstains(conn):
    response = ask(conn, ScriptedProvider([LLMError("down")]), question=QUESTION)

    assert isinstance(response, AbstainResponse)
    assert response.reason == "llm_unavailable"


def test_a_shared_name_asks_which_node(conn):
    response = ask(conn, node="orders")

    assert isinstance(response, ClarificationResponse)
    assert response.reason == "ambiguous"
    assert {c.unique_id for c in response.candidates} == {ORDERS.unique_id, RAW_ORDERS.unique_id}


def test_a_typo_suggests_but_never_answers(conn):
    provider = ScriptedProvider([{**PARSE, "entity": "stg_ordrs"}])

    response = ask(conn, provider, question="What depends on stg_ordrs?")

    assert isinstance(response, ClarificationResponse)
    assert response.reason == "not_found"
    assert response.query == "stg_ordrs"
    assert response.candidates[0].unique_id == STG_ORDERS.unique_id
    assert len(provider.calls) == 1
