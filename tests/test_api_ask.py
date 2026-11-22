from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from ask_examples import (
    EDGES,
    FCT_ORDERS,
    MART_WEEKLY,
    NODES,
    ORDERS,
    PARSE,
    RAW_ORDERS,
    STG_ORDERS,
)
from fastapi.testclient import TestClient
from helpers import TEST_PROJECT
from langgraph.checkpoint.memory import InMemorySaver
from psycopg.rows import dict_row

from asklineage.api.ask import get_conn, get_llm, get_thread_graph
from asklineage.api.flow import build_graph, checkpoint_serde
from asklineage.api.threads import create_thread, try_lock_thread
from asklineage.llm.scripted import ScriptedProvider
from asklineage.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    # No database, LLM or Postgres checkpointer unless a test overrides these again.
    threads = build_graph(checkpointer=InMemorySaver(serde=checkpoint_serde()))
    app.dependency_overrides[get_conn] = lambda: None
    app.dependency_overrides[get_llm] = lambda: None
    app.dependency_overrides[get_thread_graph] = lambda: threads
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def db_client(client, conn, load_graph) -> TestClient:
    load_graph(
        [STG_ORDERS, FCT_ORDERS, ORDERS, RAW_ORDERS],
        [(STG_ORDERS.unique_id, FCT_ORDERS.unique_id)],
    )
    app.dependency_overrides[get_conn] = lambda: conn
    return client


@pytest.fixture
def chain_client(client, conn, load_graph) -> TestClient:
    load_graph(NODES, EDGES)
    app.dependency_overrides[get_conn] = lambda: conn
    return client


def edge_pairs(data: dict) -> list[tuple[str, str]]:
    return [(edge["parent_id"], edge["child_id"]) for edge in data["edges"]]


def post(client, **body) -> dict:
    response = client.post("/api/ask", json=body)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"direction": "upstream", "question": "where does stg_orders come from?"},
        {"node": "stg_orders", "depth": 26},
        {"node": "stg_orders", "direction": "sideways"},
        {"question": "x" * 501},
        {"node": "stg_orders", "colour": "blue"},
        {"choose": "model.x.y"},
        {"choose": "model.x.y", "thread_id": str(uuid4()), "node": "stg_orders"},
        {"node": "stg_orders", "start_thread": True, "thread_id": str(uuid4())},
        {"node": "stg_orders", "thread_id": "not-a-uuid"},
    ],
)
def test_bad_requests_are_rejected(client, body):
    assert client.post("/api/ask", json=body).status_code == 422


def test_free_text_without_an_llm_abstains(client):
    data = post(client, question="what depends on stg_orders?")

    assert data["kind"] == "abstain"
    assert data["reason"] == "llm_off"
    assert data["thread_id"] is None


def test_structured_request_returns_the_lineage_contract(db_client):
    data = post(db_client, node="stg_orders", project=TEST_PROJECT)

    assert data["kind"] == "lineage"
    assert data["node"]["unique_id"] == STG_ORDERS.unique_id
    assert data["direction"] == "downstream"
    assert data["highlights"][0]["text"] == (
        "`fct_orders` (finance) reads directly from `stg_orders`."
    )
    assert data["facts"]["by_depth"] == {"1": 1}
    assert data["table"] == [
        {
            "unique_id": FCT_ORDERS.unique_id,
            "display_name": "fct_orders",
            "resource_type": "model",
            "depth": 1,
            "owner": "finance",
            "owner_source": "owners_file",
        }
    ]
    assert data["edges"] == [{"parent_id": STG_ORDERS.unique_id, "child_id": FCT_ORDERS.unique_id}]


def test_edges_join_the_root_and_the_table_rows(chain_client):
    data = post(chain_client, node="stg_orders", project=TEST_PROJECT)

    ids = {data["node"]["unique_id"]} | {row["unique_id"] for row in data["table"]}
    assert edge_pairs(data) == [
        (FCT_ORDERS.unique_id, MART_WEEKLY.unique_id),
        (STG_ORDERS.unique_id, FCT_ORDERS.unique_id),
    ]
    assert all(parent in ids and child in ids for parent, child in edge_pairs(data))


def test_an_upstream_walk_keeps_edges_parent_to_child(chain_client):
    data = post(chain_client, node="mart_weekly", direction="upstream", project=TEST_PROJECT)

    assert [row["display_name"] for row in data["table"]] == [
        "fct_orders",
        "stg_orders",
        "raw.orders",
    ]
    assert sorted(edge_pairs(data)) == sorted(EDGES)


def test_free_text_with_an_llm_returns_attempts_by_stage(db_client):
    app.dependency_overrides[get_llm] = lambda: ScriptedProvider([PARSE])

    data = post(db_client, question="What depends on stg_orders?", project=TEST_PROJECT)

    assert data["kind"] == "lineage"
    assert data["highlights"][0]["text"].startswith("`fct_orders` (finance)")
    assert [a["stage"] for a in data["attempts"]] == ["parse"]
    assert all(a["passed"] for a in data["attempts"])


def test_unknown_node_returns_a_clarification(db_client):
    data = post(db_client, node="stg_ordrs", project=TEST_PROJECT)

    assert data["kind"] == "clarification"
    assert data["reason"] == "not_found"
    assert data["resumable"] is False
    assert data["candidates"][0]["unique_id"] == STG_ORDERS.unique_id


def test_a_thread_clarifies_then_answers_the_choice(db_client):
    asked = post(db_client, node="orders", project=TEST_PROJECT, start_thread=True)
    thread_id = asked["thread_id"]

    chosen = post(db_client, thread_id=thread_id, choose=ORDERS.unique_id)

    assert (asked["kind"], asked["resumable"]) == ("clarification", True)
    assert chosen["kind"] == "lineage"
    assert chosen["node"]["unique_id"] == ORDERS.unique_id
    assert chosen["thread_id"] == thread_id
    assert chosen["edges"] == []


def test_a_threaded_answer_carries_its_edges(db_client):
    data = post(db_client, node="stg_orders", project=TEST_PROJECT, start_thread=True)

    assert data["thread_id"] is not None
    assert edge_pairs(data) == [(STG_ORDERS.unique_id, FCT_ORDERS.unique_id)]


def test_an_unknown_thread_is_not_found(db_client):
    body = {"thread_id": str(uuid4()), "node": "stg_orders"}

    assert db_client.post("/api/ask", json=body).status_code == 404


def test_choose_on_a_thread_that_is_not_waiting_is_a_conflict(db_client):
    started = post(db_client, node="stg_orders", project=TEST_PROJECT, start_thread=True)
    body = {"thread_id": started["thread_id"], "choose": STG_ORDERS.unique_id}

    assert db_client.post("/api/ask", json=body).status_code == 409


def test_a_busy_thread_is_a_conflict(db_client, conn, database_url):
    thread_id = create_thread(conn)
    body = {"thread_id": str(thread_id), "node": "stg_orders", "project": TEST_PROJECT}

    with psycopg.connect(database_url, row_factory=dict_row) as other:
        assert try_lock_thread(other, thread_id)
        assert db_client.post("/api/ask", json=body).status_code == 409
