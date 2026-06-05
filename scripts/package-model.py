"""
scripts/package-model.py

Package model artifacts for upload to Azure ML.

Creates model.tar.gz containing:
  models/xgb_readmission_v1.joblib
  models/scaler_v1.joblib
  models/feature_names.json
  infra/azure-ml/score.py
  infra/azure-ml/conda.yml

Run from repo root:
  python scripts/package-model.py

Output: model.tar.gz (excluded from git via .gitignore)
"""

import os
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OUTPUT = REPO_ROOT / "model.tar.gz"

FILES = [
    ("models/xgb_readmission_v1.joblib", "models/xgb_readmission_v1.joblib"),
    ("models/scaler_v1.joblib", "models/scaler_v1.joblib"),
    ("models/feature_names.json", "models/feature_names.json"),
    ("infra/azure-ml/score.py", "score.py"),
    ("infra/azure-ml/conda.yml", "conda.yml"),
]


def main():
    missing = [src for src, _ in FILES if not (REPO_ROOT / src).exists()]
    if missing:
        print("ERROR: missing files:")
        for f in missing:
            print(f"  {f}")
        print("\nRun scripts/train.py first to generate model artifacts.")
        sys.exit(1)

    with tarfile.open(OUTPUT, "w:gz") as tar:
        for src, arcname in FILES:
            full = REPO_ROOT / src
            tar.add(full, arcname=arcname)
            print(f"  added {src} -> {arcname}")

    size_kb = OUTPUT.stat().st_size // 1024
    print(f"\nCreated {OUTPUT.name} ({size_kb} KB)")
    print("Next: python scripts/deploy-model.py --help")


if __name__ == "__main__":
    main()
