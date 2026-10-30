"""API contract tests. The scheduler is off so the test client does not start a background job."""

from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_current_history_and_dashboard(client, settings):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert health.json()["version"] == "v0001"

    current = client.get("/model/current")
    assert current.status_code == 200
    body = current.json()
    assert body["version"] == "v0001"
    assert body["status"] == "production"
    assert body["primary_metric"] == "f1_weighted"
    assert body["metrics"]["f1_weighted"] == pytest.approx(body["primary_metric_value"])
    assert body["feature_names"] == ["x1", "x2"]
    assert body["label_map"] == {"0": "low", "1": "high"}
    assert body["observation_count"] == 120
    assert body["artifact_available"] is True

    history = client.get("/model/history")
    assert history.status_code == 200
    payload = history.json()
    assert [item["version"] for item in payload["versions"]] == ["v0001"]
    assert payload["versions"][0]["created_at"]
    assert payload["runs"][0]["trigger"] == "startup"
    assert payload["runs"][0]["status"] == "promoted"

    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "v0001" in page.text
    assert "f1_weighted" in page.text
    assert "<svg" in page.text
    assert "Comparison" in page.text

    bundle = joblib.load(Path(settings.models_dir) / "production.joblib")
    assert bundle["version"] == body["version"]
    assert bundle["metrics"]["f1_weighted"] == pytest.approx(body["metrics"]["f1_weighted"])


def test_predict_uses_the_production_model(client):
    high = client.post("/predict", json={"features": {"x1": 3.0, "x2": 3.0, "ignored": 1.0}})
    low = client.post("/predict", json={"features": {"x2": -3.0, "x1": -3.0}})
    assert high.status_code == 200
    assert low.status_code == 200
    assert high.json()["label"] == "high"
    assert low.json()["label"] == "low"
    assert high.json()["model_version"] == "v0001"
    assert sum(high.json()["probabilities"].values()) == pytest.approx(1)

    missing = client.post("/predict", json={"features": {"x1": 1.0}})
    assert missing.status_code == 400
    assert "x2" in missing.json()["detail"]

    empty = client.post("/predict", json={"features": {}})
    assert empty.status_code == 422

    wrong_type = client.post("/predict", json={"features": {"x1": "nope", "x2": 1}})
    assert wrong_type.status_code == 422


def test_retrain_skips_without_new_rows_and_force_keeps_a_model(client):
    skipped = client.post("/model/retrain")
    assert skipped.status_code == 200
    assert skipped.json()["status"] == "skipped"

    forced = client.post("/model/retrain", params={"force": "true"})
    assert forced.status_code == 200
    assert forced.json()["status"] in {"promoted", "rejected"}

    current = client.get("/model/current")
    assert current.status_code == 200
    assert current.json()["is_production"] is True
    still_serves = client.post("/predict", json={"features": {"x1": 2.5, "x2": 1.0}})
    assert still_serves.status_code == 200
    assert still_serves.json()["model_version"] == current.json()["version"]
