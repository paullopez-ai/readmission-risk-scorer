"""
Tests for POST /predict and POST /predict/explain.
All tests use MOCK_MODEL=true (set in conftest.py).
"""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SCENARIO_1_PAYLOAD, SCENARIO_2_PAYLOAD, SCENARIO_4_PAYLOAD


def test_predict_scenario1_high_risk(client: TestClient):
    """Scenario 1: HF + SNF + high comorbidity → HIGH tier, requires_review=True."""
    response = client.post("/predict", json=SCENARIO_1_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    assert data["risk_tier"] == "HIGH"
    assert data["risk_score"] >= 0.65
    assert data["requires_review"] is True
    assert len(data["shap_factors"]) == 3
    assert data["imputed_features"] == []
    assert "prediction_id" in data
    assert "timestamp" in data
    assert "model_version" in data
    assert "feature_hash" in data
    assert data["inference_ms"] > 0
    assert data["cost_usd"] > 0


def test_predict_scenario2_low_risk(client: TestClient):
    """Scenario 2: KNEE_HIP + HOME + low comorbidity → LOW tier, no review."""
    response = client.post("/predict", json=SCENARIO_2_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    assert data["risk_tier"] == "LOW"
    assert data["risk_score"] < 0.35
    assert data["requires_review"] is False
    assert len(data["shap_factors"]) == 3


def test_predict_scenario4_missing_feature(client: TestClient):
    """Scenario 4: missing specialist_consult_ct → imputed_features populated."""
    response = client.post("/predict", json=SCENARIO_4_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    assert "specialist_consult_ct" in data["imputed_features"]
    assert data["risk_tier"] in ("HIGH", "MODERATE", "LOW")
    # Prediction must still be valid despite missing feature
    assert 0.0 <= data["risk_score"] <= 1.0


def test_predict_explicit_scenario_id(client: TestClient):
    """scenario_id query parameter overrides content-based matching."""
    response = client.post(
        "/predict?scenario_id=scenario-2-low-risk", json=SCENARIO_1_PAYLOAD
    )
    assert response.status_code == 200
    data = response.json()
    # Even though payload matches scenario-1, explicit ID returns scenario-2
    assert data["risk_tier"] == "LOW"


def test_predict_response_schema_complete(client: TestClient):
    """All required RiskAssessment fields are present in every response."""
    response = client.post("/predict", json=SCENARIO_1_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    required_fields = [
        "prediction_id", "timestamp", "model_version", "feature_hash",
        "risk_score", "risk_tier", "shap_factors", "recommended_actions",
        "requires_review", "imputed_features", "ood_warning", "low_confidence",
        "inference_ms", "cost_usd",
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_predict_shap_factors_structure(client: TestClient):
    """Each SHAP factor has the required sub-fields."""
    response = client.post("/predict", json=SCENARIO_1_PAYLOAD)
    data = response.json()

    for factor in data["shap_factors"]:
        assert "feature" in factor
        assert "display_name" in factor
        assert "feature_value" in factor
        assert "shap_value" in factor
        assert factor["direction"] in ("positive", "negative")


def test_predict_recommended_actions_high(client: TestClient):
    """HIGH-tier prediction includes social work consult action."""
    response = client.post("/predict", json=SCENARIO_1_PAYLOAD)
    data = response.json()
    assert data["risk_tier"] == "HIGH"
    actions_text = " ".join(data["recommended_actions"]).lower()
    assert "social work" in actions_text or "follow-up" in actions_text


def test_predict_recommended_actions_low(client: TestClient):
    """LOW-tier prediction includes standard discharge action."""
    response = client.post("/predict", json=SCENARIO_2_PAYLOAD)
    data = response.json()
    assert data["risk_tier"] == "LOW"
    actions_text = " ".join(data["recommended_actions"]).lower()
    assert "standard" in actions_text or "follow-up" in actions_text


def test_predict_explain_returns_narrative(client: TestClient):
    """POST /predict/explain returns all RiskAssessment fields plus narrative."""
    response = client.post("/predict/explain", json=SCENARIO_1_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    assert "narrative" in data
    assert isinstance(data["narrative"], str)
    assert len(data["narrative"]) > 20
    # All core fields still present
    assert "risk_score" in data
    assert "risk_tier" in data


def test_predict_invalid_comorbidity_index(client: TestClient):
    """Pydantic validation rejects comorbidity_index > 6."""
    bad_payload = {**SCENARIO_1_PAYLOAD, "comorbidity_index": 10}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_invalid_dx_group(client: TestClient):
    """Pydantic validation rejects unknown primary_dx_group."""
    bad_payload = {**SCENARIO_1_PAYLOAD, "primary_dx_group": "UNKNOWN_DX"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_health_endpoint(client: TestClient):
    """GET /health returns ok status with mock_model=true."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mock_model"] is True
