"""
GET /health — liveness check for the inference API.

Returns model mode, version, and basic status. Used by the UI and
monitoring systems to confirm the API is running and in the expected mode.
"""

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

MOCK_MODEL: bool = os.getenv("MOCK_MODEL", "true").lower() == "true"
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"


@router.get("/health")
async def health():
    """
    Return API health status and inference mode.

    mock_model=true: demo mode, no model files required.
    mock_model=false: live inference, XGBoost loaded from models/.
    """
    model_artifacts_present = (MODEL_DIR / "xgb_readmission_v1.joblib").exists()

    return JSONResponse(
        content={
            "status": "ok",
            "version": "1.0.0",
            "mock_model": MOCK_MODEL,
            "model_artifacts_present": model_artifacts_present,
            "inference_mode": os.getenv("INFERENCE_MODE", "local"),
        }
    )
