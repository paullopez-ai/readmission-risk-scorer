"""
XGBoost model loading and feature encoding for live inference.

Used only when MOCK_MODEL=false. Imports are lazy (inside functions)
so the module can be imported without xgboost/joblib installed.

Feature encoding: ordinal encoding with fixed category maps.
These maps are identical to those used in scripts/train.py —
any change here must be mirrored there.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# Ordinal encoding maps for categorical features.
# Order matters: these must match the order used in train.py.
ENCODING_MAPS: dict[str, dict[str, int]] = {
    "primary_dx_group": {
        "AMI": 0, "HF": 1, "PNEUMONIA": 2, "COPD": 3,
        "KNEE_HIP": 4, "STROKE": 5, "OTHER": 6,
    },
    "age_group": {
        "18-44": 0, "45-64": 1, "65-74": 2, "75-84": 3, "85+": 4,
    },
    "discharge_disposition": {
        "HOME": 0, "HOME_HEALTH": 1, "SNF": 2, "REHAB": 3, "AMA": 4,
    },
    "insurance_type": {
        "MEDICARE": 0, "MEDICAID": 1, "COMMERCIAL": 2, "SELF_PAY": 3,
    },
}

# Feature order for the numpy array passed to the model.
# Must be identical to the column order in train.py.
FEATURE_ORDER: list[str] = [
    "primary_dx_group",
    "comorbidity_index",
    "length_of_stay_days",
    "age_group",
    "prior_admissions_12m",
    "procedure_count",
    "discharge_disposition",
    "icu_flag",
    "emergency_admit_flag",
    "insurance_type",
    "specialist_consult_ct",
    "incomplete_dc_flag",
    "weekend_discharge",
]

# Module-level model cache — loaded once per process
_model = None
_scaler = None
_feature_names: list[str] | None = None


def load_model():
    """
    Load XGBoost model, StandardScaler, and feature names from models/.
    Caches after first load. Raises FileNotFoundError if artifacts are missing.
    """
    global _model, _scaler, _feature_names
    if _model is not None:
        return _model, _scaler, _feature_names

    import joblib

    model_path = MODEL_DIR / "xgb_readmission_v1.joblib"
    scaler_path = MODEL_DIR / "scaler_v1.joblib"
    feature_names_path = MODEL_DIR / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Run scripts/train.py to generate model artifacts."
        )

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    with open(feature_names_path) as f:
        _feature_names = json.load(f)

    return _model, _scaler, _feature_names


def encode_features(record_dict: dict[str, Any]) -> np.ndarray:
    """
    Encode a DischargeRecord dict to a numeric numpy array.

    Applies ordinal encoding to categorical features and converts booleans
    to integers. The resulting array shape is (1, 13) for a single record.

    Args:
        record_dict: DischargeRecord.model_dump() with imputation applied
                     (no None values).

    Returns:
        numpy array of shape (1, 13), dtype float64, ready for StandardScaler.
    """
    encoded = []
    for feature in FEATURE_ORDER:
        value = record_dict[feature]
        if feature in ENCODING_MAPS:
            encoded.append(float(ENCODING_MAPS[feature][value]))
        elif isinstance(value, bool):
            encoded.append(float(int(value)))
        else:
            encoded.append(float(value))
    return np.array(encoded, dtype=np.float64).reshape(1, -1)
