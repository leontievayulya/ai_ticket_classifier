from fastapi.testclient import TestClient
from src.api import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_success():
    payload = {
        "subject": "Refund request",
        "description": "I want to return my product and get my money back."
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    assert "queue" in response.json()

def test_predict_empty_text():
    payload = {
        "subject": "   ",
        "description": "   "
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422

def test_predict_missing_field():
    payload = {"subject": "Some subject"}
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422

def test_predict_unsupported_language():
    payload = {
        "subject": "Mein Laptop ist kaputt",
        "description": "Der Bildschirm ist schwarz."
    }
    response = client.post("/v1/predict", json=payload)
    # Найденный риск: модель не понимает язык, но выдает предсказание
    assert response.status_code == 200