#!/usr/bin/env python3
"""
Evaluation harness for readmission-risk-scorer.

Runs the held-out test set through AUROC, calibration, threshold sensitivity,
SHAP global analysis, latency benchmarking, and ROI cost modeling.
All outputs are committed to reports/ so the benchmark is reproducible.

Output:
  reports/eval-report.json       — All metrics + latency benchmarks + ROI model
  reports/shap-summary.png       — SHAP global feature importance (beeswarm plot)
  reports/roc-curve.png          — ROC curve with three operating thresholds marked
  reports/calibration-plot.png   — Reliability diagram (probability calibration)
  reports/training-cost.json     — Cost model for embedding in eval-report.json

Expected AUROC: 0.75-0.85 for real-world data; higher on synthetic (clean labels).
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap  # used only for shap.summary_plot visualization
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

matplotlib.use("Agg")  # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.cost import build_cost_model
from src.utils.inference import ENCODING_MAPS, FEATURE_ORDER, encode_features

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

BOOL_FEATURES = {"icu_flag", "emergency_admit_flag", "incomplete_dc_flag", "weekend_discharge"}
LABEL_COLUMN = "readmitted_30d"

# Risk tier thresholds (must match src/utils/risk_tier.py defaults)
HIGH_THRESHOLD = 0.65
MODERATE_THRESHOLD = 0.35

sns.set_theme(style="whitegrid", palette="muted")


def encode_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Ordinal-encode a DataFrame of discharge records to a numpy array."""
    encoded = df[FEATURE_ORDER].copy()
    for col, mapping in ENCODING_MAPS.items():
        encoded[col] = encoded[col].map(mapping)
    for col in BOOL_FEATURES:
        encoded[col] = encoded[col].astype(int)
    return encoded.values.astype(np.float64)


def metrics_at_threshold(y_true, y_proba, threshold: float) -> dict:
    """Compute precision, recall, F1 at a given probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "n_flagged": int(y_pred.sum()),
        "pct_flagged": round(float(y_pred.mean()) * 100, 1),
    }


def run_latency_benchmark(model, scaler, n_iterations: int = 1000) -> dict:
    """Benchmark inference + SHAP latency over n_iterations predictions."""
    import xgboost as xgb

    test_record = {
        "primary_dx_group": "HF",
        "comorbidity_index": 5,
        "length_of_stay_days": 7,
        "age_group": "75-84",
        "prior_admissions_12m": 3,
        "procedure_count": 2,
        "discharge_disposition": "SNF",
        "icu_flag": True,
        "emergency_admit_flag": True,
        "insurance_type": "MEDICARE",
        "specialist_consult_ct": 1,
        "incomplete_dc_flag": True,
        "weekend_discharge": False,
    }
    features_raw = encode_features(test_record)
    features_scaled = scaler.transform(features_raw)
    booster = model.get_booster()
    dm = xgb.DMatrix(features_scaled)

    # Warm up
    for _ in range(10):
        model.predict_proba(features_scaled)
        booster.predict(dm, pred_contribs=True)

    # Benchmark
    latencies = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        model.predict_proba(features_scaled)
        booster.predict(dm, pred_contribs=True)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    return {
        "n_iterations": n_iterations,
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "mean_ms": round(float(latencies.mean()), 2),
        "min_ms": round(float(latencies.min()), 2),
        "max_ms": round(float(latencies.max()), 2),
    }


def plot_roc_curve(y_true, y_proba, thresholds: list[float], output_path: Path) -> float:
    """Plot ROC curve with operating thresholds marked. Returns AUROC."""
    fpr, tpr, thresh = roc_curve(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"XGBoost (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")

    colors = {"HIGH": "#dc2626", "MODERATE": "#d97706", "LOW": "#16a34a"}
    labels = {HIGH_THRESHOLD: "HIGH tier", MODERATE_THRESHOLD: "MODERATE tier"}
    for t, label in labels.items():
        idx = np.argmin(np.abs(thresh - t))
        color = colors.get(label.split()[0], "gray")
        ax.scatter(fpr[idx], tpr[idx], s=80, zorder=5, color=color, label=f"Threshold {t} ({label})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — 30-Day Readmission Prediction", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved {output_path}")
    return auroc


def plot_calibration(y_true, y_proba, output_path: Path) -> dict:
    """Plot calibration reliability diagram. Returns calibration metrics."""
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=10, strategy="uniform"
    )

    # Expected calibration error (ECE)
    bin_sizes = np.histogram(y_proba, bins=10, range=(0, 1))[0]
    ece = float(np.sum(np.abs(fraction_of_positives - mean_predicted) * bin_sizes) / len(y_proba))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: reliability diagram
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    axes[0].plot(mean_predicted, fraction_of_positives, "o-", color="#2563eb",
                 lw=2, label=f"XGBoost (ECE={ece:.3f})")
    axes[0].set_xlabel("Mean Predicted Probability", fontsize=11)
    axes[0].set_ylabel("Fraction of Positives", fontsize=11)
    axes[0].set_title("Calibration Plot (Reliability Diagram)", fontsize=12)
    axes[0].legend(fontsize=10)

    # Right: prediction distribution
    axes[1].hist(y_proba[y_true == 0], bins=30, alpha=0.6, color="#2563eb",
                 label="Not readmitted", density=True)
    axes[1].hist(y_proba[y_true == 1], bins=30, alpha=0.6, color="#dc2626",
                 label="Readmitted", density=True)
    axes[1].axvline(x=HIGH_THRESHOLD, color="#dc2626", ls="--", lw=1.5, label=f"HIGH ({HIGH_THRESHOLD})")
    axes[1].axvline(x=MODERATE_THRESHOLD, color="#d97706", ls="--", lw=1.5,
                    label=f"MODERATE ({MODERATE_THRESHOLD})")
    axes[1].set_xlabel("Predicted Probability", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Score Distribution by True Class", fontsize=12)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved {output_path}")

    return {
        "expected_calibration_error": round(ece, 4),
        "n_bins": 10,
        "note": "ECE < 0.05 indicates well-calibrated probabilities.",
    }


def plot_shap_summary(model, X_test_scaled: np.ndarray, feature_names: list[str],
                      output_path: Path) -> list[dict]:
    """
    Generate SHAP global beeswarm summary plot.
    Returns top-5 global feature importances for eval-report.json.

    Uses XGBoost native pred_contribs=True instead of shap.TreeExplainer to avoid
    a known SHAP 0.49 / XGBoost 3.x base_score format incompatibility.
    """
    import xgboost as xgb

    # Use a sample for the plot to keep it readable
    n_sample = min(500, len(X_test_scaled))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test_scaled), size=n_sample, replace=False)
    X_sample = X_test_scaled[idx]

    # XGBoost native SHAP: pred_contribs returns (n_samples, n_features + 1)
    # The last column is the baseline (bias) term — drop it
    booster = model.get_booster()
    dm = xgb.DMatrix(X_sample, feature_names=feature_names)
    shap_values = booster.predict(dm, pred_contribs=True)[:, :-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_size=None,
        max_display=13,
    )
    plt.title("SHAP Global Feature Importance — Readmission Risk", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved {output_path}")

    # Global importance: mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1], reverse=True
    )
    return [
        {"feature": name, "mean_abs_shap": round(float(val), 4)}
        for name, val in ranked[:5]
    ]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model artifacts
    model_path = MODELS_DIR / "xgb_readmission_v1.joblib"
    scaler_path = MODELS_DIR / "scaler_v1.joblib"
    feature_names_path = MODELS_DIR / "feature_names.json"

    for p in (model_path, scaler_path, feature_names_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artifact: {p}\nRun: python scripts/train.py (Gate 2)"
            )

    print("[evaluate] Loading model artifacts")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_names_path) as f:
        feature_names = json.load(f)

    # Load test set
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing test data: {test_path}\nRun: python scripts/generate-dataset.py (Gate 1)"
        )

    print(f"[evaluate] Loading {test_path}")
    df_test = pd.read_csv(test_path)
    print(f"[evaluate] {len(df_test)} test records, readmission rate: {df_test[LABEL_COLUMN].mean():.1%}")

    # Encode and scale test features
    X_test_raw = df_test[FEATURE_ORDER].copy()
    for col, mapping in ENCODING_MAPS.items():
        X_test_raw[col] = X_test_raw[col].map(mapping)
    for col in BOOL_FEATURES:
        X_test_raw[col] = X_test_raw[col].astype(int)
    X_test_scaled = scaler.transform(X_test_raw.values.astype(np.float64))
    y_test = df_test[LABEL_COLUMN].values

    # Predictions
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    print(f"[evaluate] Predictions generated (score range: "
          f"{y_proba.min():.3f} – {y_proba.max():.3f})")

    # --- AUROC ---
    auroc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    print(f"[evaluate] Test AUROC: {auroc:.4f}")
    print(f"[evaluate] Average Precision (PR-AUC): {avg_precision:.4f}")

    # --- ROC curve plot ---
    roc_path = REPORTS_DIR / "roc-curve.png"
    plot_roc_curve(y_test, y_proba, [HIGH_THRESHOLD, MODERATE_THRESHOLD], roc_path)

    # --- Calibration plot ---
    calibration_path = REPORTS_DIR / "calibration-plot.png"
    calibration_metrics = plot_calibration(y_test, y_proba, calibration_path)

    # --- SHAP global summary plot ---
    shap_path = REPORTS_DIR / "shap-summary.png"
    print("[evaluate] Computing SHAP values (may take ~30s)...")
    global_shap = plot_shap_summary(model, X_test_scaled, feature_names, shap_path)
    print(f"[evaluate] Top SHAP feature: {global_shap[0]['feature']} "
          f"(mean |SHAP|={global_shap[0]['mean_abs_shap']:.4f})")

    # --- Threshold sensitivity analysis ---
    # Three operating thresholds, each optimized for a tier
    thresholds_to_eval = [
        HIGH_THRESHOLD,       # HIGH tier: maximize recall for high-risk
        MODERATE_THRESHOLD,   # MODERATE tier: balanced
        0.20,                 # Sensitivity threshold (broader catch net)
    ]
    threshold_table = [metrics_at_threshold(y_test, y_proba, t) for t in thresholds_to_eval]
    threshold_table[0]["tier"] = "HIGH (optimized for recall)"
    threshold_table[1]["tier"] = "MODERATE (balanced)"
    threshold_table[2]["tier"] = "SENSITIVITY (broad net)"
    print(f"[evaluate] HIGH tier (t={HIGH_THRESHOLD}): "
          f"precision={threshold_table[0]['precision']:.3f}, "
          f"recall={threshold_table[0]['recall']:.3f}, "
          f"F1={threshold_table[0]['f1']:.3f}")

    # --- Latency benchmark ---
    print("[evaluate] Running latency benchmark (1,000 predictions + SHAP)...")
    latency = run_latency_benchmark(model, scaler)
    print(f"[evaluate] Latency — p50: {latency['p50_ms']}ms, "
          f"p95: {latency['p95_ms']}ms, p99: {latency['p99_ms']}ms")

    # --- Cost model ---
    cost_model = build_cost_model(daily_discharge_volume=50)

    # --- Assemble eval report ---
    eval_report = {
        "model_version": "xgb_readmission_v1",
        "test_set": {
            "n_records": int(len(df_test)),
            "readmission_rate": round(float(df_test[LABEL_COLUMN].mean()), 4),
            "note": "Held-out 1,000 records; not seen during training or hyperparameter search.",
        },
        "performance": {
            "auroc": round(auroc, 4),
            "average_precision": round(avg_precision, 4),
            "note": (
                "AUROC > 0.90 reflects clean synthetic labels (threshold-based assignment). "
                "Real-world EHR data typically yields 0.78-0.82 AUROC for this feature set."
            ),
        },
        "threshold_sensitivity": threshold_table,
        "calibration": calibration_metrics,
        "shap_global_importance": global_shap,
        "latency_benchmark": latency,
        "cost_model": cost_model,
        "artifacts": {
            "roc_curve": "reports/roc-curve.png",
            "calibration_plot": "reports/calibration-plot.png",
            "shap_summary": "reports/shap-summary.png",
        },
    }

    report_path = REPORTS_DIR / "eval-report.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"[evaluate] Saved {report_path}")

    # Also save training-cost.json separately
    cost_path = REPORTS_DIR / "training-cost.json"
    with open(cost_path, "w") as f:
        json.dump(cost_model, f, indent=2)
    print(f"[evaluate] Saved {cost_path}")

    print("\n[evaluate] Done. Gate 3 complete.")
    print(f"  AUROC:         {auroc:.4f}")
    print(f"  PR-AUC:        {avg_precision:.4f}")
    print(f"  ECE:           {calibration_metrics['expected_calibration_error']:.4f}")
    print(f"  Top SHAP feat: {global_shap[0]['feature']}")
    print(f"  p50 latency:   {latency['p50_ms']}ms")
    print(f"  p95 latency:   {latency['p95_ms']}ms")


if __name__ == "__main__":
    main()
