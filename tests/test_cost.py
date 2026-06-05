"""
Tests for src/utils/cost.py — inference cost computation and ROI model.
"""

import pytest

from src.utils.cost import (
    AZURE_ML_DS2V2_HOURLY_RATE_USD,
    LOCAL_COST_PER_PREDICTION_USD,
    build_cost_model,
    compute_inference_cost,
    compute_llm_narrative_cost,
)


def test_local_inference_cost_near_zero():
    """Local inference cost is a small positive number."""
    cost = compute_inference_cost(5.0, mode="local")
    assert cost > 0
    assert cost < 0.001  # less than 0.1 cent per prediction


def test_azure_ml_cost_proportional_to_latency():
    """Azure ML cost scales with inference duration."""
    cost_5ms = compute_inference_cost(5.0, mode="azure_ml")
    cost_15ms = compute_inference_cost(15.0, mode="azure_ml")
    assert cost_15ms > cost_5ms
    assert cost_15ms == pytest.approx(cost_5ms * 3, rel=1e-6)


def test_azure_ml_cost_formula():
    """Azure ML cost = inference_ms / 1000 / 3600 * hourly_rate."""
    inference_ms = 10.0
    expected = inference_ms / 1000 / 3600 * AZURE_ML_DS2V2_HOURLY_RATE_USD
    actual = compute_inference_cost(inference_ms, mode="azure_ml")
    assert actual == pytest.approx(expected, rel=1e-9)


def test_local_cost_is_constant():
    """Local cost does not vary with inference latency."""
    cost_1ms = compute_inference_cost(1.0, mode="local")
    cost_100ms = compute_inference_cost(100.0, mode="local")
    assert cost_1ms == cost_100ms == LOCAL_COST_PER_PREDICTION_USD


def test_llm_narrative_cost_positive():
    """LLM narrative cost is a small positive value."""
    cost = compute_llm_narrative_cost()
    assert cost > 0
    assert cost < 0.01  # less than 1 cent per narrative


def test_cost_model_structure():
    """build_cost_model returns all required ROI model keys."""
    model = build_cost_model(daily_discharge_volume=50)

    required_keys = [
        "azure_ml_hourly_rate_usd",
        "demo_daily_endpoint_cost_usd",
        "prod_daily_endpoint_cost_usd",
        "daily_discharge_volume",
        "demo_cost_per_prediction_usd",
        "llm_narrative_cost_per_call_usd",
        "cms_penalty_per_prevented_readmission_usd",
        "break_even_prevention_rate",
        "break_even_interpretation",
        "pricing_source",
    ]
    for key in required_keys:
        assert key in model, f"Missing cost model key: {key}"


def test_cost_model_roi_framing():
    """CMS penalty >> daily endpoint cost (the ROI math must work out)."""
    model = build_cost_model(daily_discharge_volume=50)
    # Daily cost should be well under $1 for a 4-hour demo session
    assert model["demo_daily_endpoint_cost_usd"] < 1.0
    # CMS penalty is at least $10K per prevented readmission
    assert model["cms_penalty_per_prevented_readmission_usd"] >= 10_000
    # Break-even rate must be very small (< 0.01 = less than 1%)
    assert model["break_even_prevention_rate"] < 0.01


def test_cost_model_daily_volume_parametric():
    """build_cost_model respects the daily_discharge_volume parameter."""
    model_50 = build_cost_model(daily_discharge_volume=50)
    model_100 = build_cost_model(daily_discharge_volume=100)
    # Cost per prediction halves when volume doubles
    assert model_100["demo_cost_per_prediction_usd"] == pytest.approx(
        model_50["demo_cost_per_prediction_usd"] / 2, rel=1e-6
    )
