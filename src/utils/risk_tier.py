"""
Risk tier classification and care coordination action mapping.

Three tiers map directly to discharge planning protocols:
  HIGH (>= 0.65):     Immediate social work consult + human review queue
  MODERATE (0.35-0.64): Telephonic follow-up within 7 days
  LOW (< 0.35):       Standard discharge protocol

Thresholds are configurable via env vars (HIGH_THRESHOLD, MODERATE_THRESHOLD)
so clinical teams can tune the precision/recall tradeoff documented in the
eval report's threshold sensitivity table.
"""

import os

HIGH_THRESHOLD: float = float(os.getenv("HIGH_THRESHOLD", "0.65"))
MODERATE_THRESHOLD: float = float(os.getenv("MODERATE_THRESHOLD", "0.35"))

_RECOMMENDED_ACTIONS: dict[str, list[str]] = {
    "HIGH": [
        "Immediate social work consult before discharge",
        "48-hour telephonic follow-up by care coordinator",
        "Medication reconciliation with clinical pharmacist",
        "Schedule PCP follow-up within 7 days",
        "Route to high-risk patient review queue",
    ],
    "MODERATE": [
        "Telephonic follow-up within 7 days of discharge",
        "Medication review with discharge nurse",
        "Post-discharge support group referral",
        "Schedule PCP follow-up within 14 days",
    ],
    "LOW": [
        "Standard discharge education packet",
        "30-day telephonic follow-up call",
        "Schedule routine PCP follow-up",
    ],
}


def classify_risk(score: float) -> str:
    """
    Map a continuous risk score [0-1] to a three-tier classification.

    Thresholds are intentionally conservative on the HIGH tier side to
    maximize recall for high-risk patients (missing a high-risk patient
    is worse than a false positive triggering an unnecessary consult).

    Args:
        score: XGBoost predict_proba readmission probability [0-1]

    Returns:
        "HIGH", "MODERATE", or "LOW"
    """
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "LOW"


def get_recommended_actions(tier: str) -> list[str]:
    """
    Return the ordered list of care coordination actions for a given tier.

    Args:
        tier: "HIGH", "MODERATE", or "LOW"

    Returns:
        List of actionable strings for the discharge planning team.
    """
    return _RECOMMENDED_ACTIONS.get(tier, _RECOMMENDED_ACTIONS["LOW"])
