import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import api.main as main_module
from api.main import app

FEATURES = [
    17.99,
    10.38,
    122.8,
    1001.0,
    0.1184,
    0.2776,
    0.3001,
    0.1471,
    0.2419,
    0.07871,
    1.095,
    0.9053,
    8.589,
    153.4,
    0.006399,
    0.04904,
    0.05373,
    0.01587,
    0.03003,
    0.006193,
    25.38,
    17.33,
    184.6,
    2019.0,
    0.1622,
    0.6656,
    0.7119,
    0.2654,
    0.4601,
    0.1189,
]


def _make_pipeline(prediction=0):
    proba = [[0.93, 0.07]] if prediction == 0 else [[0.07, 0.93]]
    p = MagicMock()
    p.predict.return_value = np.array([prediction])
    p.predict_proba.return_value = np.array(proba)
    return p


@pytest.fixture(autouse=True)
def reset_model():
    main_module._model = None
    yield
    main_module._model = None


@pytest.fixture
def client():
    with patch("mlflow.sklearn.load_model", return_value=_make_pipeline()), patch("psycopg2.connect"):
        yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "model" in resp.json()


def test_predict_valid(client):
    resp = client.post("/predict", json={"features": FEATURES})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"prediction", "confidence", "label", "timestamp"}
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["label"] in ("malignant", "benign")


def test_predict_label_malignant(client):
    resp = client.post("/predict", json={"features": FEATURES})
    body = resp.json()
    assert body["prediction"] == 0
    assert body["label"] == "malignant"


def test_predict_label_benign():
    with patch("mlflow.sklearn.load_model", return_value=_make_pipeline(1)), patch("psycopg2.connect"):
        resp = TestClient(app).post("/predict", json={"features": FEATURES})
    assert resp.json()["prediction"] == 1
    assert resp.json()["label"] == "benign"


def test_predict_too_few_features(client):
    resp = client.post("/predict", json={"features": [1.0] * 29})
    assert resp.status_code == 422


def test_predict_too_many_features(client):
    resp = client.post("/predict", json={"features": [1.0] * 31})
    assert resp.status_code == 422


def test_predict_db_failure_swallowed():
    with patch("mlflow.sklearn.load_model", return_value=_make_pipeline()), patch(
        "psycopg2.connect", side_effect=Exception("DB down")
    ):
        resp = TestClient(app).post("/predict", json={"features": FEATURES})
    assert resp.status_code == 200


def test_model_loaded_once():
    with patch("mlflow.sklearn.load_model", return_value=_make_pipeline()) as mock_load, patch(
        "psycopg2.connect"
    ):
        c = TestClient(app)
        c.post("/predict", json={"features": FEATURES})
        c.post("/predict", json={"features": FEATURES})
    assert mock_load.call_count == 1
