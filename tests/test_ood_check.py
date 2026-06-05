"""
Tests for src/utils/ood_check.py — out-of-distribution detection.
"""

import pytest

from src.utils.ood_check import TRAINING_BOUNDS, check_ood, get_ood_features


def _base_record() -> dict:
    """A well-formed record within training bounds."""
    return {
        "primary_dx_group": "HF",
        "comorbidity_index": 3,
        "length_of_stay_days": 5,
        "age_group": "65-74",
        "prior_admissions_12m": 2,
        "procedure_count": 3,
        "discharge_disposition": "HOME",
        "icu_flag": False,
        "emergency_admit_flag": True,
        "insurance_type": "MEDICARE",
        "specialist_consult_ct": 2,
        "incomplete_dc_flag": False,
        "weekend_discharge": False,
    }


def test_in_distribution_record_no_warning():
    """A record with all features in range returns False."""
    assert check_ood(_base_record()) is False


def test_ood_comorbidity_index_too_high():
    """comorbidity_index > 6 triggers OOD warning."""
    record = {**_base_record(), "comorbidity_index": 8}
    assert check_ood(record) is True


def test_ood_comorbidity_index_too_low():
    """comorbidity_index < 0 triggers OOD warning."""
    record = {**_base_record(), "comorbidity_index": -1}
    assert check_ood(record) is True


def test_ood_length_of_stay_too_high():
    """length_of_stay_days > 30 triggers OOD warning."""
    record = {**_base_record(), "length_of_stay_days": 45}
    assert check_ood(record) is True


def test_ood_at_boundary_is_ok():
    """Values at exactly the boundary (max/min) are in-distribution."""
    low, high = TRAINING_BOUNDS["comorbidity_index"]
    record_low = {**_base_record(), "comorbidity_index": low}
    record_high = {**_base_record(), "comorbidity_index": high}
    assert check_ood(record_low) is False
    assert check_ood(record_high) is False


def test_get_ood_features_returns_names():
    """get_ood_features returns list of out-of-range feature names."""
    record = {**_base_record(), "comorbidity_index": 10, "length_of_stay_days": 50}
    ood = get_ood_features(record)
    assert "comorbidity_index" in ood
    assert "length_of_stay_days" in ood


def test_get_ood_features_empty_for_valid():
    """get_ood_features returns empty list for in-distribution record."""
    assert get_ood_features(_base_record()) == []


def test_ood_check_ignores_none_values():
    """None values (pre-imputation) do not trigger OOD warning."""
    record = {**_base_record(), "specialist_consult_ct": None}
    assert check_ood(record) is False


def test_ood_predict_response_includes_flag(client):
    """POST /predict response includes ood_warning field (always present)."""
    from tests.conftest import SCENARIO_1_PAYLOAD
    response = client.post("/predict", json=SCENARIO_1_PAYLOAD)
    assert response.status_code == 200
    assert "ood_warning" in response.json()
