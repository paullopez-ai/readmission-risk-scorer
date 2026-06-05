"""
readmission-risk-scorer FastAPI application entry point.

Demo track (MOCK_MODEL=true):
  FastAPI on localhost:8000; returns fixtures from data/mock-predictions.json.
  No model files required, no API keys needed.

Live inference (MOCK_MODEL=false):
  Loads committed models/ artifacts at startup; real XGBoost inference + SHAP.
  No cloud account needed; runs entirely locally.

Hyperscaler track:
  Azure Function wraps an Azure ML managed online endpoint; same response schema.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

MOCK_MODEL = os.getenv("MOCK_MODEL", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MOCK_MODEL:
        # Pre-load model artifacts at startup so first prediction is fast
        from src.utils.inference import load_model
        try:
            model, scaler, feature_names = load_model()
            app.state.model = model
            app.state.scaler = scaler
            app.state.feature_names = feature_names
            print(
                f"[startup] XGBoost model loaded: {len(feature_names)} features, "
                f"MOCK_MODEL={MOCK_MODEL}"
            )
        except FileNotFoundError as e:
            print(f"[startup] WARNING: {e}")
            print("[startup] Set MOCK_MODEL=true for demo mode without model files.")
            raise
    else:
        print(f"[startup] MOCK_MODEL=true — returning fixtures from data/mock-predictions.json")
    yield


app = FastAPI(
    title="readmission-risk-scorer",
    description=(
        "30-day hospital readmission risk prediction API. "
        "XGBoost + SHAP explainability. No LLM in the critical path. "
        "Addresses CMS HRRP penalty context: up to 3% Medicare DRG payment reduction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.routes import predict, eval, review, health  # noqa: E402

app.include_router(predict.router, tags=["inference"])
app.include_router(eval.router, tags=["evaluation"])
app.include_router(review.router, tags=["review-queue"])
app.include_router(health.router, tags=["health"])
