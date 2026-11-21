import pytest
from ask_examples import (
    EDGES,
    HIGHLIGHTS,
    NODES,
    NOT_LINEAGE,
    ORDERS,
    PARSE,
    QUESTION,
    RAW_ORDERS,
    STG_ORDERS,
    owner_reply,
)
from helpers import TEST_PROJECT

from asklineage.answer.lineage_answer import HighlightStatus
from asklineage.api.flow import SearchOptions, run_ask
from asklineage.api.models import (
    AbstainResponse,
    AskRequest,
    ClarificationResponse,
    LineageResponse,
    OwnerResponse,
    SearchResponse,
)
from asklineage.embed.hashing import HashingEmbedder
from asklineage.embed.store import sync_chunks
from asklineage.lineage.graph import Direction
from asklineage.llm.base import LLMError
from asklineage.llm.scripted import ScriptedProvider


@pytest.fixture(autouse=True)
def graph(load_graph) -> None:
    load_graph(NODES, EDGES)


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


def test_an_out_of_scope_question_abstains(conn):
    provider = ScriptedProvider([NOT_LINEAGE])

    response = ask(conn, provider, question="How do I write a window function?")

    assert isinstance(response, AbstainResponse)
    assert response.reason == "out_of_scope"
    assert len(provider.calls) == 1


def test_an_owner_question_is_answered_from_the_nodes_table(conn):
    provider = ScriptedProvider([owner_reply()])

    response = ask(conn, provider, question="Who owns stg_orders?")

    assert isinstance(response, OwnerResponse)
    assert response.node.unique_id == STG_ORDERS.unique_id
    assert (response.owner, response.owner_source) == ("finance", "owners_file")
    assert response.owner_source_meaning.startswith("assigned by path rules")
    assert response.message.startswith("stg_orders is owned by finance")
    assert stages(response) == ["parse"]


def test_an_owner_question_says_when_no_owner_is_recorded(conn):
    provider = ScriptedProvider([owner_reply("mart_weekly")])

    response = ask(conn, provider, question="Who owns mart_weekly?")

    assert isinstance(response, OwnerResponse)
    assert (response.owner, response.owner_source, response.owner_source_meaning) == (
        None,
        None,
        None,
    )
    assert "No owner is recorded for mart_weekly" in response.message


def test_an_owner_question_about_an_ambiguous_name_asks_which(conn):
    provider = ScriptedProvider([owner_reply("orders")])

    response = ask(conn, provider, question="Who owns orders?")

    assert isinstance(response, ClarificationResponse)
    assert response.reason == "ambiguous"


def test_free_text_without_an_llm_abstains(conn):
    response = ask(conn, question=QUESTION)

    assert isinstance(response, AbstainResponse)
    assert response.reason == "llm_off"


def test_a_question_that_fails_the_check_twice_abstains(conn):
    guessed = {**PARSE, "entity": "mart_weekly"}
    question = "Does stg_orders feed fct_orders?"  # two names: nothing to recover
    response = ask(conn, ScriptedProvider([guessed, guessed]), question=question)

    assert isinstance(response, AbstainResponse)
    assert response.reason == "not_understood"
    assert stages(response) == ["parse", "parse"]


def test_a_name_the_model_changed_is_taken_from_the_question(conn):
    guessed = {**PARSE, "entity": "fct_orders"}
    provider = ScriptedProvider([guessed, guessed, LLMError("down")])

    response = ask(conn, provider, question=QUESTION)

    assert isinstance(response, LineageResponse)
    assert response.node.unique_id == STG_ORDERS.unique_id


def test_a_typo_the_model_corrected_still_gets_suggestions(conn):
    corrected = {**PARSE, "entity": "stg_orders"}
    provider = ScriptedProvider([corrected, corrected])

    response = ask(conn, provider, question="What depends on stg_ordrs?")

    assert isinstance(response, ClarificationResponse)
    assert (response.reason, response.query) == ("not_found", "stg_ordrs")
    assert response.candidates[0].unique_id == STG_ORDERS.unique_id
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


def search_reply():
    return {**NOT_LINEAGE, "intent": "search"}


def test_a_search_question_without_embeddings_falls_back_to_keywords(conn):
    response = ask(conn, ScriptedProvider([search_reply()]), question="Where are the weekly marts?")

    assert isinstance(response, SearchResponse)
    assert response.mode == "keyword"
    assert response.low_confidence
    assert [hit.display_name for hit in response.hits] == ["mart_weekly"]
    assert stages(response) == ["parse"]


def test_a_search_question_ranks_by_meaning_and_flags_weak_matches(conn):
    embedder = HashingEmbedder()
    sync_chunks(conn, TEST_PROJECT, embedder)
    options = SearchOptions(embedder=embedder, low_confidence_below=0.99)

    response = run_ask(
        conn,
        ScriptedProvider([search_reply()]),
        AskRequest(project=TEST_PROJECT, question="weekly mart"),
        max_named=20,
        search_options=options,
    )

    assert response.mode == "semantic"
    assert response.hits[0].display_name == "mart_weekly"
    assert response.low_confidence
    assert "under 0.99" in response.message
