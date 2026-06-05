"""
POST /predict — core readmission risk inference endpoint.
POST /predict/explain — same prediction + optional LLM narrative (Track 2).

MOCK_MODEL=true: returns deterministic fixture from data/mock-predictions.json.
  Matching logic: specialist_consult_ct=None → scenario-4; HF+SNF → scenario-1;
  KNEE_HIP+HOME → scenario-2; else → scenario-1 (default).
  Override with ?scenario_id=<key> query parameter.

MOCK_MODEL=false: runs real XGBoost inference + SHAP computation.
  Applies imputation → OOD check → encode → scale → predict_proba →
  SHAP values → tier classification → audit trail.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.models.discharge_record import DischargeRecord, FEATURE_DISPLAY_NAMES
from src.models.risk_assessment import RiskAssessment, RiskAssessmentWithNarrative, SHAPFactor
from src.utils.audit import compute_feature_hash, write_review_queue_entry
from src.utils.cost import compute_inference_cost
from src.utils.imputation import impute_features
from src.utils.ood_check import check_ood
from src.utils.risk_tier import classify_risk, get_recommended_actions

MOCK_MODEL: bool = os.getenv("MOCK_MODEL", "true").lower() == "true"
MOCK_LLM: bool = os.getenv("MOCK_LLM", "true").lower() == "true"
INFERENCE_MODE: str = os.getenv("INFERENCE_MODE", "local")

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
MOCK_FIXTURES_PATH = DATA_DIR / "mock-predictions.json"

router = APIRouter()
_mock_fixtures: dict | None = None


def _load_mock_fixtures() -> dict:
    global _mock_fixtures
    if _mock_fixtures is None:
        with open(MOCK_FIXTURES_PATH) as f:
            _mock_fixtures = json.load(f)
    return _mock_fixtures


def _get_mock_key(record: DischargeRecord, scenario_id: Optional[str]) -> str:
    if scenario_id:
        return scenario_id
    # Content-based matching for demo scenarios
    if record.specialist_consult_ct is None:
        return "scenario-4-missing-feature"
    if record.primary_dx_group == "HF" and record.discharge_disposition in ("SNF", "AMA"):
        return "scenario-1-high-risk"
    if record.primary_dx_group == "KNEE_HIP" and record.discharge_disposition == "HOME":
        return "scenario-2-low-risk"
    return "scenario-1-high-risk"


def _mock_response(record: DischargeRecord, scenario_id: Optional[str]) -> RiskAssessment:
    key = _get_mock_key(record, scenario_id)
    fixtures = _load_mock_fixtures()
    if key not in fixtures:
        key = "scenario-1-high-risk"
    fixture = fixtures[key].copy()
    # Refresh per-call fields so responses are unique
    fixture["prediction_id"] = f"mock-{uuid.uuid4().hex[:8]}"
    fixture["timestamp"] = datetime.now(timezone.utc).isoformat()
    return RiskAssessment(**fixture)


def _live_predict(record: DischargeRecord) -> RiskAssessment:
    """Full live inference path: imputation → OOD → XGBoost → SHAP → tier."""
    from src.utils.inference import encode_features, load_model
    from src.utils.shap_explainer import compute_shap_values

    start = time.perf_counter()

    record_dict = record.model_dump()

    # Feature hash of raw input (pre-imputation, for audit without PHI storage)
    feature_hash = compute_feature_hash(record_dict)

    # Imputation
    record_imputed, imputed_features = impute_features(record_dict)

    # OOD check
    ood_warning = check_ood(record_imputed)

    # Load model (cached after first call)
    model, scaler, feature_names = load_model()

    # Encode and scale
    features_raw = encode_features(record_imputed)
    features_scaled = scaler.transform(features_raw)

    # Predict
    proba = model.predict_proba(features_scaled)[0]  # [p_no_readmit, p_readmit]
    risk_score = float(proba[1])
    low_confidence = abs(proba[1] - proba[0]) < 0.25

    # SHAP top-3 factors
    shap_raw = compute_shap_values(model, features_scaled, feature_names)
    shap_factors = [
        SHAPFactor(
            feature=item["feature"],
            display_name=FEATURE_DISPLAY_NAMES.get(item["feature"], item["feature"]),
            feature_value=record_imputed.get(item["feature"]),
            shap_value=item["shap_value"],
            direction=item["direction"],
        )
        for item in shap_raw
    ]

    # Tier and actions
    risk_tier = classify_risk(risk_score)
    recommended_actions = get_recommended_actions(risk_tier)
    requires_review = risk_tier == "HIGH"

    # Latency and cost
    inference_ms = (time.perf_counter() - start) * 1000
    cost_usd = compute_inference_cost(inference_ms, mode=INFERENCE_MODE)

    assessment = RiskAssessment(
        prediction_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_version="xgb_readmission_v1",
        feature_hash=feature_hash,
        risk_score=risk_score,
        risk_tier=risk_tier,
        shap_factors=shap_factors,
        recommended_actions=recommended_actions,
        requires_review=requires_review,
        imputed_features=imputed_features,
        ood_warning=ood_warning,
        low_confidence=low_confidence,
        inference_ms=round(inference_ms, 2),
        cost_usd=cost_usd,
    )

    if requires_review:
        write_review_queue_entry(
            assessment.model_dump(),
            queue_path=DATA_DIR / "review-queue.jsonl",
        )

    return assessment


@router.post("/predict", response_model=RiskAssessment)
async def predict(
    record: DischargeRecord,
    scenario_id: Optional[str] = None,
):
    """
    Predict 30-day readmission risk for a patient at discharge.

    Returns risk score [0-1], tier (HIGH/MODERATE/LOW), top-3 SHAP factors,
    care coordination actions, and a full audit trail.

    HIGH-tier predictions are automatically written to the review queue and
    flagged with requires_review=true.

    Set MOCK_MODEL=true (default) for deterministic fixture responses.
    Set MOCK_MODEL=false to run real XGBoost + SHAP inference.
    """
    if MOCK_MODEL:
        return _mock_response(record, scenario_id)
    return _live_predict(record)


@router.post("/predict/explain", response_model=RiskAssessmentWithNarrative)
async def predict_explain(
    record: DischargeRecord,
    scenario_id: Optional[str] = None,
):
    """
    Predict readmission risk + generate a plain-English clinical narrative.

    Track 2 endpoint. Runs the same prediction as /predict, then translates
    the top SHAP factors into a 2-sentence discharge planning note using
    Claude Haiku (or a deterministic stub when MOCK_LLM=true).

    Requires ANTHROPIC_API_KEY when MOCK_LLM=false.
    """
    # Run core prediction (mock or live)
    if MOCK_MODEL:
        assessment = _mock_response(record, scenario_id)
    else:
        assessment = _live_predict(record)

    # Generate narrative
    if MOCK_LLM:
        top_factor = assessment.shap_factors[0] if assessment.shap_factors else None
        if top_factor:
            narrative = (
                f"This patient has a {assessment.risk_tier.lower()} risk of 30-day "
                f"readmission (score: {assessment.risk_score:.0%}). "
                f"The primary contributing factor is {top_factor.display_name} "
                f"({'+' if top_factor.shap_value > 0 else ''}{top_factor.shap_value:.2f}), "
                f"which {'increases' if top_factor.direction == 'positive' else 'decreases'} "
                f"readmission likelihood."
            )
        else:
            narrative = (
                f"This patient has a {assessment.risk_tier.lower()} risk of 30-day "
                f"readmission (score: {assessment.risk_score:.0%})."
            )
    else:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY not set. Set MOCK_LLM=true for demo mode.",
            )

        client = anthropic.Anthropic(api_key=api_key)
        factors_text = "\n".join(
            f"- {f.display_name}: {f.shap_value:+.2f} ({f.direction})"
            for f in assessment.shap_factors
        )
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Write a 2-sentence discharge planning note for a nurse. "
                        f"The patient has {assessment.risk_tier} readmission risk "
                        f"(score: {assessment.risk_score:.0%}). "
                        f"Key risk factors:\n{factors_text}\n"
                        "Be specific and actionable. Do not mention SHAP or machine learning."
                    ),
                }
            ],
        )
        narrative = message.content[0].text

    return RiskAssessmentWithNarrative(**assessment.model_dump(), narrative=narrative)
