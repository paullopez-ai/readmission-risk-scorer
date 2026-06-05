"""
Tests for src/utils/imputation.py — missing feature handling.
"""

import pytest

from src.utils.imputation import IMPUTATION_STRATEGY, impute_features


def test_no_imputation_when_all_features_present():
    """Records with all features populated are returned unchanged."""
    record = {
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
    imputed, imputed_fields = impute_features(record)
    assert imputed_fields == []
    assert imputed["specialist_consult_ct"] == 2


def test_specialist_consult_ct_imputed_when_none():
    """Missing specialist_consult_ct is replaced with the median value."""
    record = {
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
        "specialist_consult_ct": None,
        "incomplete_dc_flag": False,
        "weekend_discharge": False,
    }
    imputed, imputed_fields = impute_features(record)
    assert "specialist_consult_ct" in imputed_fields
    assert imputed["specialist_consult_ct"] == IMPUTATION_STRATEGY["specialist_consult_ct"]["value"]
    assert imputed["specialist_consult_ct"] is not None


def test_multiple_missing_features_all_imputed():
    """Multiple None fields are all imputed and disclosed."""
    record = {
        "primary_dx_group": "COPD",
        "comorbidity_index": None,
        "length_of_stay_days": None,
        "age_group": "75-84",
        "prior_admissions_12m": None,
        "procedure_count": None,
        "discharge_disposition": "SNF",
        "icu_flag": True,
        "emergency_admit_flag": True,
        "insurance_type": "MEDICARE",
        "specialist_consult_ct": None,
        "incomplete_dc_flag": True,
        "weekend_discharge": False,
    }
    imputed, imputed_fields = impute_features(record)

    expected_imputed = ["comorbidity_index", "length_of_stay_days", "prior_admissions_12m",
                        "procedure_count", "specialist_consult_ct"]
    for field in expected_imputed:
        assert field in imputed_fields
        assert imputed[field] is not None


def test_imputation_does_not_modify_original():
    """impute_features returns a copy; the original dict is not mutated."""
    record = {
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
        "specialist_consult_ct": None,
        "incomplete_dc_flag": False,
        "weekend_discharge": False,
    }
    original_value = record["specialist_consult_ct"]
    impute_features(record)
    assert record["specialist_consult_ct"] == original_value  # original unchanged


def test_imputation_values_are_non_negative():
    """All imputation values are non-negative integers (valid feature range)."""
    for feature, config in IMPUTATION_STRATEGY.items():
        assert config["value"] >= 0, f"{feature} imputation value is negative"


def test_imputation_strategy_has_rationale():
    """Every imputation strategy entry documents a clinical rationale."""
    for feature, config in IMPUTATION_STRATEGY.items():
        assert "rationale" in config, f"{feature} missing rationale"
        assert len(config["rationale"]) > 20, f"{feature} rationale too short"
