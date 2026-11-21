from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from helpers import TEST_PROJECT, make_node

from asklineage.api.ask import get_conn, get_llm
from asklineage.llm.scripted import ScriptedProvider
from asklineage.main import app

STG_ORDERS = make_node("stg_orders", owner="finance")
FCT_ORDERS = make_node("fct_orders", owner="finance")


@pytest.fixture
def client() -> Iterator[TestClient]:
    # No database or LLM unless a test overrides these again.
    app.dependency_overrides[get_conn] = lambda: None
    app.dependency_overrides[get_llm] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def db_client(client, conn, load_graph) -> TestClient:
    load_graph([STG_ORDERS, FCT_ORDERS], [(STG_ORDERS.unique_id, FCT_ORDERS.unique_id)])
    app.dependency_overrides[get_conn] = lambda: conn
    return client


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"direction": "upstream", "question": "where does stg_orders come from?"},
        {"node": "stg_orders", "depth": 26},
        {"node": "stg_orders", "direction": "sideways"},
        {"question": "x" * 501},
        {"node": "stg_orders", "colour": "blue"},
    ],
)
def test_bad_requests_are_rejected(client, body):
    assert client.post("/api/ask", json=body).status_code == 422


def test_free_text_without_an_llm_abstains(client):
    response = client.post("/api/ask", json={"question": "what depends on stg_orders?"})

    assert response.status_code == 200
    assert response.json()["kind"] == "abstain"
    assert response.json()["reason"] == "llm_off"


def test_structured_request_returns_the_lineage_contract(db_client):
    body = {"node": "stg_orders", "project": TEST_PROJECT}

    data = db_client.post("/api/ask", json=body).json()

    assert data["kind"] == "lineage"
    assert data["node"]["unique_id"] == STG_ORDERS.unique_id
    assert data["direction"] == "downstream"
    assert data["highlight_status"] == "off"
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


def test_free_text_with_an_llm_returns_attempts_by_stage(db_client):
    parse = {"is_lineage": True, "entity": "stg_orders", "direction": "downstream", "depth": None}
    highlights = {
        "claims": [
            {
                "upstream": "stg_orders",
                "downstream": "fct_orders",
                "hops": "direct",
                "text": "`fct_orders` reads directly from `stg_orders`.",
            }
        ]
    }
    app.dependency_overrides[get_llm] = lambda: ScriptedProvider([parse, highlights])
    body = {"question": "What depends on stg_orders?", "project": TEST_PROJECT}

    data = db_client.post("/api/ask", json=body).json()

    assert data["kind"] == "lineage"
    assert data["highlights"][0]["text"].startswith("`fct_orders` (finance)")
    assert [a["stage"] for a in data["attempts"]] == ["parse", "highlights"]
    assert all(a["passed"] for a in data["attempts"])


def test_unknown_node_returns_a_clarification(db_client):
    body = {"node": "stg_ordrs", "project": TEST_PROJECT}

    data = db_client.post("/api/ask", json=body).json()

    assert data["kind"] == "clarification"
    assert data["reason"] == "not_found"
    assert data["candidates"][0]["unique_id"] == STG_ORDERS.unique_id
