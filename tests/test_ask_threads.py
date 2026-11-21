from datetime import timedelta
from uuid import uuid4

import psycopg
import pytest
from ask_examples import (
    EDGES,
    FCT_ORDERS,
    HIGHLIGHTS,
    NODES,
    NOT_LINEAGE,
    ORDERS,
    QUESTION,
    RAW_ORDERS,
    STG_ORDERS,
    parse_reply,
)
from helpers import TEST_PROJECT
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row

from asklineage.answer.lineage_answer import HighlightStatus
from asklineage.api.flow import (
    Remembered,
    build_graph,
    checkpoint_serde,
    run_ask,
    thread_config,
    waiting_for_choice,
)
from asklineage.api.models import (
    AbstainResponse,
    AskRequest,
    ClarificationResponse,
    LineageResponse,
)
from asklineage.api.threads import create_thread, prune_threads, touch_thread, try_lock_thread
from asklineage.lineage.graph import Direction
from asklineage.lineage.resolve import Ambiguous, Candidate
from asklineage.llm.base import LLMError
from asklineage.llm.scripted import ScriptedProvider


@pytest.fixture(autouse=True)
def graph(load_graph) -> None:
    load_graph(NODES, EDGES)


@pytest.fixture
def saver() -> InMemorySaver:
    return InMemorySaver(serde=checkpoint_serde())


@pytest.fixture
def threads(saver):
    return build_graph(checkpointer=saver)


def turn(conn, threads, thread_id, provider=None, **fields):
    if "choose" not in fields:
        fields["project"] = TEST_PROJECT
    request = AskRequest(thread_id=thread_id, **fields)
    return run_ask(conn, provider, request, max_named=20, graph=threads, thread_id=thread_id)


def names(response) -> list[str]:
    return [row.display_name for row in response.table]


def test_a_shared_name_pauses_the_thread_and_resumes_with_the_choice(conn, threads):
    thread_id = uuid4()

    asked = turn(conn, threads, thread_id, node="orders")
    chosen = turn(conn, threads, thread_id, choose=RAW_ORDERS.unique_id)

    assert isinstance(asked, ClarificationResponse)
    assert (asked.reason, asked.resumable, asked.thread_id) == ("ambiguous", True, thread_id)
    assert "send choose" in asked.message
    assert isinstance(chosen, LineageResponse)
    assert chosen.node.unique_id == RAW_ORDERS.unique_id
    assert names(chosen) == ["stg_orders", "fct_orders", "mart_weekly"]
    assert not waiting_for_choice(threads, thread_id)


def test_resuming_does_not_read_the_question_again(conn, threads):
    thread_id = uuid4()
    provider = ScriptedProvider([parse_reply("orders", depth=1), LLMError("down")])

    turn(conn, threads, thread_id, provider, question="What depends on orders, one hop?")
    chosen = turn(conn, threads, thread_id, provider, choose=RAW_ORDERS.unique_id)

    assert (chosen.direction, chosen.depth) == (Direction.DOWNSTREAM, 1)
    assert names(chosen) == ["stg_orders"]
    assert [a.stage for a in chosen.attempts] == ["parse", "highlights"]
    assert len(provider.calls) == 2


def test_a_choice_that_was_not_offered_is_rejected(conn, threads):
    thread_id = uuid4()
    turn(conn, threads, thread_id, node="orders")

    rejected = turn(conn, threads, thread_id, choose=FCT_ORDERS.unique_id)

    assert isinstance(rejected, ClarificationResponse)
    assert (rejected.reason, rejected.resumable) == ("invalid_choice", False)
    assert not waiting_for_choice(threads, thread_id)


def test_a_suggestion_for_a_typo_can_be_chosen(conn, threads):
    thread_id = uuid4()

    asked = turn(conn, threads, thread_id, node="stg_ordrs")
    chosen = turn(conn, threads, thread_id, choose=STG_ORDERS.unique_id)

    assert (asked.reason, asked.resumable) == ("not_found", True)
    assert chosen.node.unique_id == STG_ORDERS.unique_id


def test_without_a_thread_a_clarification_is_not_resumable(conn):
    response = run_ask(conn, None, AskRequest(node="orders", project=TEST_PROJECT), max_named=20)

    assert response.resumable is False
    assert "node set to" in response.message
    assert response.thread_id is None


def test_a_follow_up_reuses_the_previous_node(conn, threads):
    thread_id = uuid4()
    follow_up = parse_reply(entity=None, direction="upstream", same_node=True)
    provider = ScriptedProvider([follow_up, LLMError("down")])

    turn(conn, threads, thread_id, node="stg_orders", depth=2)
    answer = turn(conn, threads, thread_id, provider, question="And upstream?")

    assert answer.node.unique_id == STG_ORDERS.unique_id
    assert (answer.direction, answer.depth) == (Direction.UPSTREAM, 2)
    assert names(answer) == ["raw.orders"]
    assert "`stg_orders` (downstream, 2 hops)" in provider.calls[0].prompt


def test_a_follow_up_about_another_node_keeps_direction_and_depth(conn, threads):
    thread_id = uuid4()
    provider = ScriptedProvider([parse_reply("fct_orders", direction=None), LLMError("down")])

    turn(conn, threads, thread_id, node="mart_weekly", direction="upstream", depth=1)
    answer = turn(conn, threads, thread_id, provider, question="What about fct_orders?")

    assert answer.node.unique_id == FCT_ORDERS.unique_id
    assert (answer.direction, answer.depth) == (Direction.UPSTREAM, 1)
    assert names(answer) == ["stg_orders"]


def test_a_new_question_starts_clean(conn, threads):
    thread_id = uuid4()

    abstained = turn(conn, threads, thread_id, ScriptedProvider([NOT_LINEAGE]), question="Hi?")
    answered = turn(conn, threads, thread_id, node="stg_orders")

    assert isinstance(abstained, AbstainResponse)
    assert len(abstained.attempts) == 1
    assert isinstance(answered, LineageResponse)
    assert answered.attempts == []


def test_a_new_question_abandons_a_pending_choice(conn, threads):
    thread_id = uuid4()
    turn(conn, threads, thread_id, node="orders")

    answered = turn(conn, threads, thread_id, node="stg_orders")

    assert isinstance(answered, LineageResponse)
    assert not waiting_for_choice(threads, thread_id)


def test_the_checkpoint_keeps_the_previous_node_not_the_walk(conn, threads):
    thread_id = uuid4()
    provider = ScriptedProvider([parse_reply(), HIGHLIGHTS])

    answer = turn(conn, threads, thread_id, provider, question=QUESTION)
    values = threads.get_state(thread_config(thread_id)).values

    assert answer.highlight_status is HighlightStatus.PASSED
    assert values["result"] is None
    previous = values["previous"]
    assert isinstance(previous, Remembered)
    assert previous.root.unique_id == STG_ORDERS.unique_id
    assert (previous.direction, previous.depth) == (Direction.DOWNSTREAM, 5)


def test_state_types_survive_the_checkpoint_serializer():
    serde = checkpoint_serde()
    candidate = Candidate(ORDERS.unique_id, "orders", "model", None, 1.0)
    state = {
        "unresolved": Ambiguous("orders", (candidate,)),
        "previous": Remembered(candidate, Direction.UPSTREAM, 3),
    }

    restored = serde.loads_typed(serde.dumps_typed(state))

    assert isinstance(restored["unresolved"], Ambiguous)
    assert list(restored["unresolved"].candidates) == [candidate]
    assert restored["previous"] == state["previous"]
    assert restored["previous"].direction is Direction.UPSTREAM


def test_postgres_saver_resumes_a_clarification(conn, database_url):
    thread_id = uuid4()
    with psycopg.connect(
        database_url, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as raw:
        saver = PostgresSaver(raw, serde=checkpoint_serde())
        saver.setup()
        threads = build_graph(checkpointer=saver)
        try:
            asked = turn(conn, threads, thread_id, node="orders")
            chosen = turn(conn, threads, thread_id, choose=RAW_ORDERS.unique_id)
        finally:
            saver.delete_thread(str(thread_id))
        assert threads.get_state(thread_config(thread_id)).values == {}

    assert asked.resumable
    assert names(chosen) == ["stg_orders", "fct_orders", "mart_weekly"]


def test_threads_exist_until_pruned(conn, threads, saver):
    thread_id = create_thread(conn)
    turn(conn, threads, thread_id, node="stg_orders")
    conn.execute(
        "UPDATE ask_threads SET last_used_at = now() - interval '1000 days' WHERE thread_id = %s",
        (thread_id,),
    )

    assert touch_thread(conn, uuid4()) is False
    assert prune_threads(conn, saver, timedelta(days=999)) == 1
    assert touch_thread(conn, thread_id) is False
    assert threads.get_state(thread_config(thread_id)).values == {}


def test_a_thread_in_use_cannot_be_locked_by_another_request(conn, database_url):
    thread_id = uuid4()
    assert try_lock_thread(conn, thread_id)

    with psycopg.connect(database_url, row_factory=dict_row) as other:
        assert try_lock_thread(other, thread_id) is False
        assert try_lock_thread(other, uuid4())
