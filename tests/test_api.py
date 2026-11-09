
import pytest
from fastapi.testclient import TestClient

from main import app


# Fixture runs FastAPI lifespan before each test
@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_home_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction_endpoint(client):

    payload = {
        "Type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]

    assert isinstance(
        data["failure_detected"], bool
    )

    assert (
        0 <= data["failure_probability"] <= 1
    )


def test_invalid_machine_type(client):

    payload = {
        "Type": "X",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_missing_feature(client):

    payload = {
        "Type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422
