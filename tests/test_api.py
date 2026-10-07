"""
API integration tests for DASI REST interface using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    # Context manager triggers lifespan startup and shutdown
    with TestClient(app) as test_client:
        yield test_client


class TestAPIEndpoints:
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "DASI" in data["name"]
        assert data["version"] == "0.1.0"

    def test_health_endpoint(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "operational", "HEALTHY", "OPERATIONAL"]

    def test_status_endpoint(self, client):
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "signals_loaded" in data

    def test_analyze_scam_endpoint(self, client):
        payload = {
            "messages": [
                {
                    "role": "caller",
                    "text": "This is Inspector Sharma from Cyber Crime Department. Your Aadhaar is locked under PMLA.",
                    "timestamp_offset": 0
                },
                {
                    "role": "caller",
                    "text": "Non bailable arrest warrant issued. Transfer security deposit immediately to avoid jail.",
                    "timestamp_offset": 15
                }
            ],
            "language": "en",
            "region": "Delhi NCR",
            "caller_id": "+91-9876543210"
        }
        response = client.post("/api/v1/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "incident_id" in data
        assert "threat" in data
        assert 0.0 <= data["threat"]["threat_score"] <= 1.0
        assert data["threat"]["threat_level"] in ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_campaigns_endpoint(self, client):
        response = client.get("/api/v1/campaigns")
        assert response.status_code == 200
        data = response.json()
        assert "active_campaigns" in data

    def test_alerts_endpoint(self, client):
        response = client.get("/api/v1/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
