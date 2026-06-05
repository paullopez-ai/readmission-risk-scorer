"""
Azure ML managed online endpoint scoring script.

Entry point for the AML endpoint deployment. Loaded once at container
startup; `run()` is called per request. Mirrors the local FastAPI
inference path exactly (same ENCODING_MAPS, same SHAP approach).
"""

import json
import os
import time
import logging

import joblib
import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)

# Populated in init() from the mounted model directory
_model = None
_scaler = None
_feature_names = None

# Must match src/utils/inference.py and scripts/train.py exactly
ENCODING_MAPS = {
    "primary_dx_group": {
        "HF": 0, "PNEUMONIA": 1, "COPD": 2, "KNEE_HIP": 3,
        "CARDIAC": 4, "SEPSIS": 5, "OTHER": 6,
    },
    "age_group": {
        "18-44": 0, "45-54": 1, "55-64": 2, "65-74": 3,
        "75-84": 4, "85+": 5,
    },
    "discharge_disposition": {
        "HOME": 0, "HOME_HEALTH": 1, "SNF": 2, "REHAB": 3,
        "AMA": 4, "OTHER": 5,
    },
    "insurance_type": {
        "MEDICARE": 0, "MEDICAID": 1, "COMMERCIAL": 2, "SELF_PAY": 3,
    },
}

FEATURE_ORDER = [
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

HIGH_THRESHOLD = float(os.environ.get("HIGH_THRESHOLD", "0.65"))
MODERATE_THRESHOLD = float(os.environ.get("MODERATE_THRESHOLD", "0.35"))


def init():
    """Called once when the container starts. Loads model artifacts."""
    global _model, _scaler, _feature_names

    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "models", "xgb_readmission_v1.joblib")
    scaler_path = os.path.join(model_dir, "models", "scaler_v1.joblib")
    features_path = os.path.join(model_dir, "models", "feature_names.json")

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        _feature_names = json.load(f)

    logger.info("Model loaded from %s", model_dir)


def _encode(record: dict) -> np.ndarray:
    """Encode a discharge record dict → numpy array (1, 13)."""
    row = []
    for feat in FEATURE_ORDER:
        val = record.get(feat)
        if feat in ENCODING_MAPS:
            val = ENCODING_MAPS[feat].get(str(val), 0)
        else:
            val = float(val) if val is not None else 0.0
        row.append(val)
    return np.array(row, dtype=float).reshape(1, -1)


def _compute_shap(features_scaled: np.ndarray) -> list[dict]:
    """XGBoost native SHAP — avoids shap.TreeExplainer / XGBoost 3.x bug."""
    booster = _model.get_booster()
    dm = xgb.DMatrix(features_scaled, feature_names=_feature_names)
    contribs = booster.predict(dm, pred_contribs=True)[0, :-1]  # drop bias
    paired = sorted(zip(_feature_names, contribs), key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature": name,
            "shap_value": float(val),
            "direction": "positive" if val >= 0 else "negative",
        }
        for name, val in paired[:3]
    ]


def _classify(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "LOW"


def run(raw_data: str) -> str:
    """
    Called per inference request by the AML runtime.

    Input:  JSON string — single DischargeRecord dict or
            {"data": [DischargeRecord, ...]} for batch.
    Output: JSON string — RiskAssessment or list of RiskAssessments.
    """
    t0 = time.perf_counter()

    payload = json.loads(raw_data)

    # Support single record or {"data": [...]} batch envelope
    if isinstance(payload, list):
        records = payload
    elif "data" in payload:
        records = payload["data"]
    else:
        records = [payload]

    results = []
    for record in records:
        # Impute missing specialist_consult_ct
        if record.get("specialist_consult_ct") is None:
            record["specialist_consult_ct"] = 1

        features_raw = _encode(record)
        features_scaled = _scaler.transform(features_raw)
        risk_score = float(_model.predict_proba(features_scaled)[0, 1])
        shap_factors = _compute_shap(features_scaled)
        risk_tier = _classify(risk_score)

        inference_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "risk_score": round(risk_score, 4),
            "risk_tier": risk_tier,
            "shap_factors": shap_factors,
            "inference_ms": round(inference_ms, 2),
            "model_version": "xgb_readmission_v1",
        })

    return json.dumps(results if len(results) > 1 else results[0])
