"""
API tests with FastAPI's TestClient (sends HTTP requests in memory, no server needed).

We inject a tiny FAKE model so the tests:
  - run in seconds,
  - work in CI where the real (gitignored, 250 MB) model does not exist,
  - test OUR code (routing, validation, response format), not the ML model.
"""
import pytest
from fastapi.testclient import TestClient

from api import main


class FakeModel:
    """Pretends to be NewsClassifier: same .predict() interface, hard-coded logic."""

    def predict(self, text: str) -> dict:
        label = "Sports" if "goal" in text.lower() else "World"
        return {"label": label, "confidence": 0.9}


@pytest.fixture
def client():
    main.state["model"] = FakeModel()   # set BEFORE startup -> real model is never loaded
    with TestClient(main.app) as c:
        yield c
    main.state["model"] = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_label_and_confidence(client):
    r = client.post("/predict", json={"text": "Late goal wins the match"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Sports"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_missing_field(client):
    assert client.post("/predict", json={"headline": "wrong key"}).status_code == 422


def test_predict_without_model_returns_503():
    main.state["model"] = None
    # No 'with' block -> startup (model loading) does not run
    r = TestClient(main.app).post("/predict", json={"text": "hello"})
    assert r.status_code == 503
