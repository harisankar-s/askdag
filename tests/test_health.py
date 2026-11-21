from fastapi.testclient import TestClient

from asklineage.main import app


def test_live_returns_alive() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
