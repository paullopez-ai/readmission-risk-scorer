"""
RiskAssessment: Pydantic output schema for /predict and /predict/explain.
Typed and versioned. Every field serves a specific purpose:
audit trail, clinical action, or trust/transparency signal.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class SHAPFactor(BaseModel):
    """
    A single SHAP feature contribution for one prediction.
    Positive shap_value increases readmission risk; negative decreases it.
    """

    feature: str = Field(
        description="Snake-case feature name matching DischargeRecord field names."
    )
    display_name: str = Field(
        description="Human-readable label for clinical staff display."
    )
    feature_value: Any = Field(
        description="Actual value of this feature from the input record (post-imputation)."
    )
    shap_value: float = Field(
        description=(
            "SHAP contribution in log-odds space. Positive = increases readmission risk, "
            "negative = decreases it. Magnitude indicates relative importance."
        )
    )
    direction: Literal["positive", "negative"] = Field(
        description="'positive' if this feature increases risk; 'negative' if it decreases risk."
    )


class RiskAssessment(BaseModel):
    """
    Complete readmission risk assessment returned by POST /predict.
    Includes risk score, tier, SHAP explanation, care coordination actions,
    and a full audit trail for compliance and traceability.
    """

    # --- Core prediction ---
    risk_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "XGBoost predicted probability of 30-day readmission [0-1]. "
            "Calibrated: a score of 0.75 means the model predicts ~75% "
            "likelihood of readmission based on the feature set."
        ),
    )
    risk_tier: Literal["HIGH", "MODERATE", "LOW"] = Field(
        description=(
            "Three-tier classification mapped to care coordination protocol. "
            "HIGH (>= 0.65): immediate social work consult + human review. "
            "MODERATE (0.35-0.64): telephonic follow-up within 7 days. "
            "LOW (< 0.35): standard discharge packet."
        )
    )

    # --- SHAP explanation ---
    shap_factors: list[SHAPFactor] = Field(
        description=(
            "Top 3 SHAP feature contributions for this prediction, sorted by "
            "absolute magnitude. These are the specific, auditable reasons "
            "the model assigned this risk score."
        )
    )

    # --- Care coordination ---
    recommended_actions: list[str] = Field(
        description="Actionable care coordination steps mapped to the risk tier."
    )

    # --- Trust signals ---
    requires_review: bool = Field(
        description=(
            "True for HIGH-tier predictions. Indicates a human review checkpoint "
            "should occur before care coordination actions are triggered. "
            "Prediction is written to the review queue when True."
        )
    )
    imputed_features: list[str] = Field(
        default_factory=list,
        description=(
            "Feature names for which median imputation was applied due to missing "
            "input values. Disclosed so callers can assess prediction reliability."
        ),
    )
    ood_warning: bool = Field(
        description=(
            "True if any numeric feature falls outside the training data range. "
            "Signals the model is extrapolating beyond its training distribution; "
            "downstream systems should apply additional scrutiny to this prediction."
        )
    )
    low_confidence: bool = Field(
        description=(
            "True when the gap between readmission and non-readmission probabilities "
            "is < 0.25, indicating model uncertainty. Use with caution in borderline "
            "MODERATE/LOW cases."
        )
    )

    # --- Audit trail ---
    prediction_id: str = Field(
        description="UUID identifying this specific prediction for traceability."
    )
    model_version: str = Field(
        description="Identifier of the model artifact used (e.g., 'xgb_readmission_v1')."
    )
    feature_hash: str = Field(
        description=(
            "SHA-256 hash (16-char prefix) of the input feature vector. "
            "Allows tracing which input produced a prediction without storing PHI."
        )
    )
    timestamp: str = Field(
        description="ISO 8601 UTC timestamp of when the prediction was made."
    )

    # --- Cost and latency instrumentation ---
    inference_ms: float = Field(
        description=(
            "Wall-clock latency from request receipt to response send, in milliseconds. "
            "Includes SHAP computation. Target: < 15ms at p95."
        )
    )
    cost_usd: float = Field(
        description=(
            "Estimated compute cost for this prediction in USD. Near-zero for local "
            "inference; computed from Azure ML endpoint pricing when INFERENCE_MODE=azure_ml."
        )
    )


class RiskAssessmentWithNarrative(RiskAssessment):
    """
    Extended response for POST /predict/explain (Track 2 only).
    Adds a plain-English clinical narrative generated from SHAP factors.
    """

    narrative: str = Field(
        description=(
            "Plain-English clinical narrative for non-technical discharge planning staff. "
            "Translates the top SHAP factors into actionable clinical language. "
            "Generated by Claude Haiku when MOCK_LLM=false; deterministic stub when MOCK_LLM=true."
        )
    )
