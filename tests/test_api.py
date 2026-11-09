"""
Unit tests for DataSentinel AI FastAPI Endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "DataSentinel AI"


def test_api_overview():
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert "monthly_revenue" in data
    assert "retention" in data
    assert "delivery" in data


def test_api_quality_report():
    res = client.get("/api/quality/report")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "results" in data
    assert data["summary"]["quality_score_pct"] == 100.0


def test_api_copilot_query():
    res = client.post("/api/copilot/query", json={"question": "What are top products?"})
    assert res.status_code == 200
    data = res.json()
    assert data["is_safe"] is True
    assert "generated_sql" in data
    assert "data" in data


def test_api_rca():
    res = client.get("/api/rca/2018-08")
    assert res.status_code == 200
    data = res.json()
    assert data["target_month"] == "2018-08"
    assert len(data["primary_findings"]) > 0


def test_api_executive_report():
    res = client.get("/api/reports/executive")
    assert res.status_code == 200
    data = res.json()
    assert "markdown" in data
    assert "# DataSentinel AI" in data["markdown"]
