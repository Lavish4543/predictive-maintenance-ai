
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model_name"]


def test_predict_rejects_missing_fields(client):
    response = client.post(
        "/predict",
        json={"Type": "L"},
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_rejects_invalid_machine_type(client):
    response = client.post(
        "/predict",
        json={
            "Air temperature [K]": 300,
            "Process temperature [K]": 310,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40,
            "Tool wear [min]": 100,
            "Type": "X",
        },
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_rejects_non_json(client):
    response = client.post(
        "/predict",
        data="not json",
        content_type="text/plain",
    )

    assert response.status_code == 415
