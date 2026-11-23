from collections.abc import Iterator

import pytest
from ask_examples import FCT_ORDERS, STG_ORDERS
from fastapi.testclient import TestClient
from pydantic import ValidationError

from asklineage.api.ask import get_conn, get_llm
from asklineage.config import Settings, get_settings
from asklineage.main import app

KEY = "k" * 24


@pytest.fixture
def client(conn, load_graph) -> Iterator[TestClient]:
    load_graph([STG_ORDERS, FCT_ORDERS], [(STG_ORDERS.unique_id, FCT_ORDERS.unique_id)])
    app.dependency_overrides[get_conn] = lambda: conn
    app.dependency_overrides[get_llm] = lambda: None
    app.dependency_overrides[get_settings] = lambda: Settings(api_key=KEY)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def ask(client: TestClient, headers: dict[str, str]):
    return client.post("/api/ask", json={"node": FCT_ORDERS.unique_id}, headers=headers)


def test_open_while_no_key_is_set(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(api_key=None)

    assert client.get("/api/info").status_code == 200
    assert ask(client, {}).status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": KEY + "x"}])
def test_rejects_a_missing_or_wrong_key(client: TestClient, headers: dict[str, str]) -> None:
    for response in (client.get("/api/info", headers=headers), ask(client, headers)):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "APIKey"
        assert KEY not in response.text


def test_accepts_the_right_key(client: TestClient) -> None:
    assert client.get("/api/info", headers={"X-API-Key": KEY}).status_code == 200
    response = ask(client, {"X-API-Key": KEY})
    assert response.status_code == 200
    assert response.json()["kind"] == "lineage"


def test_health_and_page_stay_open(client: TestClient) -> None:
    for path in ("/health/live", "/health/ready", "/"):
        assert client.get(path).status_code == 200


@pytest.mark.parametrize("key", ["", "short"])
def test_a_short_key_fails_at_startup(key: str) -> None:
    with pytest.raises(ValidationError, match="API_KEY"):
        Settings(api_key=key)
