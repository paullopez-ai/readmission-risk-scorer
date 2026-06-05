#!/usr/bin/env python3
"""
XGBoost training pipeline for readmission-risk-scorer.

Fits a StandardScaler and XGBoost classifier on data/train.csv using
GridSearchCV for hyperparameter tuning. Saves committed model artifacts to models/.

Output:
  models/xgb_readmission_v1.joblib   — Trained XGBoost classifier
  models/scaler_v1.joblib            — Fitted StandardScaler
  models/feature_names.json          — Ordered feature name list

Feature encoding: ordinal encoding with fixed category maps identical to
src/utils/inference.py ENCODING_MAPS. Any change here must be mirrored there.

Expected AUROC on test set: 0.75-0.85 for this synthetic feature set.
Training time: ~2-5 minutes for GridSearchCV with 5-fold CV.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
SEED = 42

# Ordinal encoding maps — MUST be identical to src/utils/inference.py ENCODING_MAPS
ENCODING_MAPS = {
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

BOOL_FEATURES = {"icu_flag", "emergency_admit_flag", "incomplete_dc_flag", "weekend_discharge"}
LABEL_COLUMN = "readmitted_30d"


def encode_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Apply ordinal encoding to categorical features and return numeric array."""
    encoded = df[FEATURE_ORDER].copy()

    for col, mapping in ENCODING_MAPS.items():
        encoded[col] = encoded[col].map(mapping)

    for col in BOOL_FEATURES:
        encoded[col] = encoded[col].astype(int)

    return encoded.values.astype(np.float64)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            "Run: python scripts/generate-dataset.py (Gate 1)"
        )

    print(f"[train] Loading {train_path}")
    df = pd.read_csv(train_path)
    print(f"[train] {len(df)} records, readmission rate: {df[LABEL_COLUMN].mean():.1%}")

    X_raw = encode_dataframe(df)
    y = df[LABEL_COLUMN].values

    # Fit StandardScaler on training features
    print("[train] Fitting StandardScaler")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # XGBoost with class imbalance handling
    # scale_pos_weight = n_negative / n_positive (for imbalanced classes)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    default_scale_pos_weight = n_neg / n_pos
    print(f"[train] Class imbalance: {n_pos} positive, {n_neg} negative "
          f"(scale_pos_weight default: {default_scale_pos_weight:.1f})")

    # Hyperparameter grid
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.10, 0.20],
        "n_estimators": [100, 200],
        "scale_pos_weight": [1, default_scale_pos_weight],
    }

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("[train] Running GridSearchCV (this takes 2-5 minutes)...")
    t0 = time.time()
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_scaled, y)
    elapsed = time.time() - t0

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_auc = grid_search.best_score_

    print(f"\n[train] GridSearchCV complete ({elapsed:.1f}s)")
    print(f"[train] Best CV AUC: {best_cv_auc:.4f}")
    print(f"[train] Best params: {best_params}")

    # Quick training-set AUROC (sanity check; test-set AUROC in evaluate.py)
    from sklearn.metrics import roc_auc_score
    y_pred_proba = best_model.predict_proba(X_scaled)[:, 1]
    train_auc = roc_auc_score(y, y_pred_proba)
    print(f"[train] Train AUROC: {train_auc:.4f} (test AUROC computed in evaluate.py)")

    # Save artifacts
    model_path = MODELS_DIR / "xgb_readmission_v1.joblib"
    scaler_path = MODELS_DIR / "scaler_v1.joblib"
    feature_names_path = MODELS_DIR / "feature_names.json"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(feature_names_path, "w") as f:
        json.dump(FEATURE_ORDER, f)

    print(f"\n[train] Saved {model_path}")
    print(f"[train] Saved {scaler_path}")
    print(f"[train] Saved {feature_names_path}")
    print("[train] Done. Gate 2 complete.")
    print("[train] Next step: run scripts/evaluate.py (Gate 3)")


if __name__ == "__main__":
    main()
