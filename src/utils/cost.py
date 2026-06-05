"""
Inference cost computation for readmission risk predictions.

Pricing constants are dated so they can be updated when Azure pricing changes.
Two modes:
  "local":    Near-zero CPU cost for local XGBoost inference
  "azure_ml": Azure ML managed online endpoint (Standard_DS2_v2)

The per-prediction cost from a managed endpoint is the fraction of the
hourly compute cost consumed by the inference duration. At 15ms and
$0.096/hr this is ~$4e-7 per prediction — essentially free per call.

The meaningful cost metric is endpoint uptime cost (documented in the
eval report ROI model), not per-prediction compute fractions.
The cost_usd field in each response enables cumulative cost tracking
in production.

Source: Azure pricing calculator, 2026-06-04
"""

# Azure ML Standard_DS2_v2 (2 vCPUs, 7 GB RAM) — cheapest AML online endpoint tier
# Price as of 2026-06-04 (East US region)
AZURE_ML_DS2V2_HOURLY_RATE_USD: float = 0.096

# Local inference: Python process CPU time is effectively zero-cost relative
# to any cloud or API-based alternative. Using a non-zero token to allow
# cumulative tracking without division-by-zero issues.
LOCAL_COST_PER_PREDICTION_USD: float = 0.000001

# LLM narrative layer (Claude Haiku 4.5 via /predict/explain endpoint)
# Input: ~200 tokens, Output: ~100 tokens
# Haiku pricing as of 2026-06-04 (approximate)
HAIKU_INPUT_COST_PER_1K_TOKENS_USD: float = 0.00025
HAIKU_OUTPUT_COST_PER_1K_TOKENS_USD: float = 0.00125

# Estimated tokens for the /predict/explain LLM call
LLM_ESTIMATED_INPUT_TOKENS: int = 200
LLM_ESTIMATED_OUTPUT_TOKENS: int = 100


def compute_inference_cost(inference_ms: float, mode: str = "local") -> float:
    """
    Compute the estimated USD cost for a single XGBoost inference call.

    Args:
        inference_ms: Wall-clock inference latency in milliseconds.
        mode: "local" or "azure_ml". Defaults to env var INFERENCE_MODE.

    Returns:
        Estimated cost in USD.
    """
    if mode == "azure_ml":
        # Cost = fraction of hour × hourly rate
        inference_hours = inference_ms / 1000 / 3600
        return inference_hours * AZURE_ML_DS2V2_HOURLY_RATE_USD
    return LOCAL_COST_PER_PREDICTION_USD


def compute_llm_narrative_cost() -> float:
    """
    Estimated cost for the Claude Haiku LLM narrative call in /predict/explain.

    Returns:
        Estimated cost in USD for one LLM narrative generation.
    """
    input_cost = (LLM_ESTIMATED_INPUT_TOKENS / 1000) * HAIKU_INPUT_COST_PER_1K_TOKENS_USD
    output_cost = (LLM_ESTIMATED_OUTPUT_TOKENS / 1000) * HAIKU_OUTPUT_COST_PER_1K_TOKENS_USD
    return input_cost + output_cost


def build_cost_model(daily_discharge_volume: int = 50) -> dict:
    """
    Build the ROI cost model for the eval report.
    Compares inference cost at typical hospital discharge volume against
    the CMS HRRP penalty avoided by one prevented readmission.

    Args:
        daily_discharge_volume: Expected daily prediction requests.

    Returns:
        Dict with cost model components for eval-report.json.
    """
    # Endpoint uptime: assume 4 hours/day (demo) or 24 hours (production)
    demo_daily_endpoint_cost = AZURE_ML_DS2V2_HOURLY_RATE_USD * 4
    prod_daily_endpoint_cost = AZURE_ML_DS2V2_HOURLY_RATE_USD * 24

    # CMS HRRP penalty: up to 3% of all Medicare DRG payments
    # Typical penalty for a 500-bed hospital: ~$500K-$1M/year
    # Conservative estimate: one prevented readmission avoids $15,000 in penalties
    cms_penalty_per_readmission_usd = 15_000

    # Break-even: how many predictions to prevent one readmission to cover costs
    demo_cost_per_prediction = demo_daily_endpoint_cost / daily_discharge_volume
    break_even_prevention_rate = demo_daily_endpoint_cost / cms_penalty_per_readmission_usd

    return {
        "azure_ml_hourly_rate_usd": AZURE_ML_DS2V2_HOURLY_RATE_USD,
        "demo_daily_endpoint_cost_usd": round(demo_daily_endpoint_cost, 3),
        "prod_daily_endpoint_cost_usd": round(prod_daily_endpoint_cost, 3),
        "daily_discharge_volume": daily_discharge_volume,
        "demo_cost_per_prediction_usd": round(demo_cost_per_prediction, 6),
        "llm_narrative_cost_per_call_usd": round(compute_llm_narrative_cost(), 6),
        "cms_penalty_per_prevented_readmission_usd": cms_penalty_per_readmission_usd,
        "break_even_prevention_rate": round(break_even_prevention_rate, 6),
        "break_even_interpretation": (
            f"At {daily_discharge_volume} discharges/day, the model costs "
            f"${demo_daily_endpoint_cost:.2f}/day to run. One prevented readmission "
            f"avoids ${cms_penalty_per_readmission_usd:,} in CMS penalties. "
            f"Break-even requires preventing {break_even_prevention_rate:.4%} of "
            "scored patients from readmitting — far below documented intervention rates."
        ),
        "pricing_source": "Azure pricing calculator, 2026-06-04, East US region",
    }
