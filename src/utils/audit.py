"""
Prediction audit trail utilities.

Every prediction carries:
  - prediction_id: UUID for traceability
  - feature_hash: SHA-256 of input features (traceable without storing PHI)
  - model_version: which model artifact produced this prediction
  - timestamp: ISO 8601 UTC

HIGH-tier predictions are written to data/review-queue.jsonl for human
review before care coordination actions are triggered.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REVIEW_QUEUE_PATH = Path("data/review-queue.jsonl")


def compute_feature_hash(features: dict[str, Any]) -> str:
    """
    Compute a SHA-256 hash of the input feature vector.

    Produces a stable 16-character hex identifier for the exact input
    that generated a prediction. Allows auditors to verify which features
    produced a given score without storing the raw PHI-containing record.

    Args:
        features: DischargeRecord.model_dump() output (pre-imputation).

    Returns:
        First 16 characters of the SHA-256 hex digest.
    """
    serialized = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def write_review_queue_entry(
    assessment_dict: dict[str, Any],
    queue_path: str | Path | None = None,
) -> None:
    """
    Append a HIGH-tier prediction to the human review queue.

    The review queue is a newline-delimited JSON file. Each line is a
    complete RiskAssessment dict. The /review-queue endpoint reads this
    file and surfaces pending HIGH-tier patients to the discharge planning team.

    Args:
        assessment_dict: RiskAssessment.model_dump() for a HIGH-tier prediction.
        queue_path: Path to the review queue file. Defaults to
                    data/review-queue.jsonl (relative to CWD).
    """
    path = Path(queue_path) if queue_path else DEFAULT_REVIEW_QUEUE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        **assessment_dict,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "reviewed": False,
    }

    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_review_queue(
    queue_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Read all entries from the review queue file.

    Args:
        queue_path: Path to the review queue file.

    Returns:
        List of prediction dicts, newest first. Empty list if no file.
    """
    path = Path(queue_path) if queue_path else DEFAULT_REVIEW_QUEUE_PATH

    if not path.exists():
        return []

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return list(reversed(entries))  # newest first
