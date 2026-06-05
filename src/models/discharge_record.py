"""
DischargeRecord: Pydantic input schema for readmission risk prediction.
Each field is a structured EHR-derived feature with clinical definition.
This is the machine-readable input contract for the /predict endpoint.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class PrimaryDxGroup(str, Enum):
    """CMS HRRP target conditions plus catch-all OTHER."""
    AMI = "AMI"           # Acute Myocardial Infarction
    HF = "HF"             # Heart Failure
    PNEUMONIA = "PNEUMONIA"
    COPD = "COPD"         # Chronic Obstructive Pulmonary Disease
    KNEE_HIP = "KNEE_HIP" # Elective knee/hip replacement (low-risk baseline)
    STROKE = "STROKE"     # Ischemic stroke
    OTHER = "OTHER"       # All other primary diagnoses


class AgeGroup(str, Enum):
    """Age groups aligned with CMS HRRP reporting bands."""
    A18_44 = "18-44"
    A45_64 = "45-64"
    A65_74 = "65-74"
    A75_84 = "75-84"
    A85_PLUS = "85+"


class DischargeDisposition(str, Enum):
    """Post-discharge care setting. Key readmission predictor."""
    HOME = "HOME"               # Discharged to self-care at home
    HOME_HEALTH = "HOME_HEALTH" # Home with visiting nursing/therapy
    SNF = "SNF"                 # Skilled nursing facility
    REHAB = "REHAB"             # Inpatient rehabilitation facility
    AMA = "AMA"                 # Against medical advice (highest risk signal)


class InsuranceType(str, Enum):
    """Payer type; proxy for social determinants and care access."""
    MEDICARE = "MEDICARE"
    MEDICAID = "MEDICAID"
    COMMERCIAL = "COMMERCIAL"
    SELF_PAY = "SELF_PAY"


# Human-readable display names for feature labels in API responses and UI
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "primary_dx_group": "Primary Diagnosis",
    "comorbidity_index": "Comorbidity Index",
    "length_of_stay_days": "Length of Stay (days)",
    "age_group": "Age Group",
    "prior_admissions_12m": "Prior Admissions (12 months)",
    "procedure_count": "Procedure Count",
    "discharge_disposition": "Discharge Disposition",
    "icu_flag": "ICU Admission",
    "emergency_admit_flag": "Emergency Admission",
    "insurance_type": "Insurance Type",
    "specialist_consult_ct": "Specialist Consult Count",
    "incomplete_dc_flag": "Incomplete Discharge Instructions",
    "weekend_discharge": "Weekend Discharge",
}


class DischargeRecord(BaseModel):
    """
    Structured EHR-derived features collected at point of discharge.
    All 13 features are defined by the CMS HRRP feature set and academic
    readmission prediction literature. No free-text fields; purely tabular.
    """

    primary_dx_group: PrimaryDxGroup = Field(
        description=(
            "Primary discharge diagnosis group. CMS HRRP target conditions "
            "(HF, COPD, AMI, PNEUMONIA, STROKE) carry significantly higher "
            "readmission risk than elective procedures (KNEE_HIP)."
        )
    )

    comorbidity_index: int = Field(
        ge=0,
        le=6,
        description=(
            "Simplified Elixhauser comorbidity score (0-6). Counts concurrent "
            "chronic conditions (diabetes, renal failure, COPD, etc.). "
            "Higher scores correlate strongly with readmission risk."
        ),
    )

    length_of_stay_days: int = Field(
        ge=1,
        le=30,
        description=(
            "Inpatient length of stay in days. Longer stays signal greater "
            "clinical complexity; stays > 7 days are associated with elevated "
            "readmission risk independent of diagnosis."
        ),
    )

    age_group: AgeGroup = Field(
        description=(
            "Patient age group. Medicare patients (65+) have higher baseline "
            "readmission risk due to comorbidity burden and reduced functional "
            "reserve. The 85+ cohort has the highest age-related risk."
        )
    )

    prior_admissions_12m: int = Field(
        ge=0,
        le=10,
        description=(
            "Number of inpatient admissions in the preceding 12 months. "
            "The strongest individual predictor of 30-day readmission; "
            "patients with >= 2 prior admissions face markedly elevated risk."
        ),
    )

    procedure_count: int = Field(
        ge=0,
        le=15,
        description=(
            "Number of distinct procedures performed during this admission. "
            "Higher counts indicate clinical complexity. Combined with ICU "
            "flag provides a severity signal beyond diagnosis alone."
        ),
    )

    discharge_disposition: DischargeDisposition = Field(
        description=(
            "Care setting to which patient is discharged. AMA carries the "
            "highest readmission risk; SNF is elevated (care transition risk); "
            "HOME is the lowest-risk disposition for appropriate patients."
        )
    )

    icu_flag: bool = Field(
        description=(
            "True if patient required ICU-level care during this admission. "
            "ICU stays indicate acute severity and physiologic instability, "
            "both associated with higher post-discharge readmission risk."
        )
    )

    emergency_admit_flag: bool = Field(
        description=(
            "True if admission originated via the emergency department. "
            "Emergency admissions often reflect decompensated chronic conditions "
            "with less time for pre-admission optimization and care planning."
        )
    )

    insurance_type: InsuranceType = Field(
        description=(
            "Primary payer type. Medicaid and self-pay patients face social "
            "determinant barriers (medication access, follow-up care) that "
            "elevate readmission risk beyond clinical factors alone."
        )
    )

    specialist_consult_ct: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description=(
            "Number of specialist consultations during admission (0-10+). "
            "May be missing in records lacking consultation documentation. "
            "When absent, median imputation is applied and disclosed in "
            "the imputed_features response field."
        ),
    )

    incomplete_dc_flag: bool = Field(
        description=(
            "True if discharge instructions were not fully completed or delivered. "
            "Incomplete discharge education is a modifiable readmission risk factor; "
            "strong signal for care coordination intervention at discharge."
        )
    )

    weekend_discharge: bool = Field(
        description=(
            "True if discharged on Saturday or Sunday. Weekend discharges are "
            "associated with reduced access to follow-up care and pharmacy "
            "services in the immediate post-discharge window."
        )
    )

    model_config = {"use_enum_values": True}
