from collections.abc import Iterator

import pytest
from ask_examples import FCT_ORDERS, ORDERS, RAW_ORDERS, STG_ORDERS
from fastapi.testclient import TestClient
from helpers import TEST_PROJECT

from asklineage.api.ask import get_conn
from asklineage.config import Settings, get_settings
from asklineage.main import app


@pytest.fixture
def info_client(conn, load_graph) -> Iterator[TestClient]:
    load_graph([STG_ORDERS, FCT_ORDERS, ORDERS, RAW_ORDERS], [])
    app.dependency_overrides[get_conn] = lambda: conn
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_index_serves_the_page() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>asklineage</title>" in response.text


def test_page_never_uses_innerhtml() -> None:
    with TestClient(app) as client:
        page = client.get("/").text

    assert "innerHTML" not in page
    assert "outerHTML" not in page
    assert "insertAdjacentHTML" not in page


def test_info_has_no_secrets(info_client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        llm_provider="openai_compatible",
        llm_base_url="http://secret-host.internal:9999/v1",
        llm_api_key="sk-secret-value",
        llm_model="qwen2.5:7b",
    )

    response = info_client.get("/api/info")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"projects", "llm_provider", "llm_model"}
    assert TEST_PROJECT in body["projects"]
    assert body["llm_provider"] == "openai_compatible"
    assert body["llm_model"] == "qwen2.5:7b"
    for secret in ("sk-secret-value", "secret-host", "9999"):
        assert secret not in response.text


def test_info_without_an_llm(info_client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(llm_provider="none")

    body = info_client.get("/api/info").json()

    assert body["llm_provider"] == "none"
    assert body["llm_model"] is None
