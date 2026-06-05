"""
Missing feature imputation strategy for discharge records.

Strategy: median imputation for numeric features.
Rationale: median is robust to outliers in clinical data and produces
defensible defaults (a patient with unknown specialist consult count
is assumed to have the median count, not zero or maximum).

All imputed features are disclosed in RiskAssessment.imputed_features
so callers are never surprised by what the model assumed.
"""

from typing import Any

# Imputation values and rationale per feature.
# Values are derived from the training data distribution;
# update these if the training distribution changes significantly.
IMPUTATION_STRATEGY: dict[str, dict[str, Any]] = {
    "comorbidity_index": {
        "value": 2,
        "rationale": (
            "Median Elixhauser score in the training cohort. "
            "A patient with unknown comorbidity burden is assumed to have "
            "moderate (not zero or maximum) chronic condition burden."
        ),
    },
    "length_of_stay_days": {
        "value": 4,
        "rationale": (
            "Median LOS in the training cohort (~4 days). "
            "Missing LOS is unusual; when absent, the median provides a "
            "conservative estimate for a typical inpatient stay."
        ),
    },
    "prior_admissions_12m": {
        "value": 1,
        "rationale": (
            "Median prior admissions in training cohort. "
            "When admission history is unavailable, we assume at least one "
            "prior admission (conservative; avoids underestimating risk for "
            "patients with missing history)."
        ),
    },
    "procedure_count": {
        "value": 2,
        "rationale": (
            "Median procedure count in training cohort. "
            "Missing procedure count typically indicates documentation gaps, "
            "not zero procedures; median is a safer default."
        ),
    },
    "specialist_consult_ct": {
        "value": 1,
        "rationale": (
            "Median specialist consult count in training cohort. "
            "This is the most commonly missing field in real EHR data "
            "(consultation notes may be in a separate system). "
            "Imputed to median rather than zero to avoid underestimating "
            "care complexity."
        ),
    },
}


def impute_features(record_dict: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Apply median imputation to any missing (None) numeric features.

    Args:
        record_dict: DischargeRecord.model_dump() output; may contain None values
                     for optional fields.

    Returns:
        Tuple of (imputed_record, list_of_imputed_field_names).
        The imputed_record has None values replaced with medians.
        imputed_field_names is empty if no imputation was needed.
    """
    imputed = record_dict.copy()
    imputed_fields: list[str] = []

    for feature, config in IMPUTATION_STRATEGY.items():
        if imputed.get(feature) is None:
            imputed[feature] = config["value"]
            imputed_fields.append(feature)

    return imputed, imputed_fields
