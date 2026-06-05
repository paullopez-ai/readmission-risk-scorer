"""
pytest configuration. Sets MOCK_MODEL=true before any test module imports
to ensure the FastAPI app initializes in mock mode.

All tests use mock mode. Never load a real model or call external APIs in tests.
"""

import os

# Must be set before importing the FastAPI app
os.environ["MOCK_MODEL"] = "true"
os.environ["MOCK_LLM"] = "true"
os.environ["INFERENCE_MODE"] = "local"

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI test client, shared across all tests in the session."""
    return TestClient(app)


# --- Shared request payloads ---

SCENARIO_1_PAYLOAD = {
    "primary_dx_group": "HF",
    "comorbidity_index": 5,
    "length_of_stay_days": 7,
    "age_group": "75-84",
    "prior_admissions_12m": 3,
    "procedure_count": 2,
    "discharge_disposition": "SNF",
    "icu_flag": True,
    "emergency_admit_flag": True,
    "insurance_type": "MEDICARE",
    "specialist_consult_ct": 1,
    "incomplete_dc_flag": True,
    "weekend_discharge": False,
}

SCENARIO_2_PAYLOAD = {
    "primary_dx_group": "KNEE_HIP",
    "comorbidity_index": 1,
    "length_of_stay_days": 3,
    "age_group": "45-64",
    "prior_admissions_12m": 0,
    "procedure_count": 2,
    "discharge_disposition": "HOME",
    "icu_flag": False,
    "emergency_admit_flag": False,
    "insurance_type": "COMMERCIAL",
    "specialist_consult_ct": 1,
    "incomplete_dc_flag": False,
    "weekend_discharge": False,
}

SCENARIO_4_PAYLOAD = {
    "primary_dx_group": "HF",
    "comorbidity_index": 5,
    "length_of_stay_days": 7,
    "age_group": "75-84",
    "prior_admissions_12m": 3,
    "procedure_count": 2,
    "discharge_disposition": "SNF",
    "icu_flag": True,
    "emergency_admit_flag": True,
    "insurance_type": "MEDICARE",
    # specialist_consult_ct deliberately omitted → triggers imputation
    "incomplete_dc_flag": True,
    "weekend_discharge": False,
}
