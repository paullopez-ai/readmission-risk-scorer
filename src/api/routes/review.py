"""
GET /review-queue — surface HIGH-tier patients pending human review.

Reads from data/review-queue.jsonl, which is appended to by POST /predict
whenever a HIGH-tier prediction is made. Returns entries newest-first.

HIGH-tier predictions are flagged requires_review=true and must be reviewed
by a discharge planning team member before care coordination actions trigger.
"""

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.utils.audit import read_review_queue

router = APIRouter()

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
REVIEW_QUEUE_PATH = DATA_DIR / "review-queue.jsonl"


@router.get("/review-queue")
async def get_review_queue():
    """
    Return pending HIGH-tier predictions from the review queue.

    Returns all entries in data/review-queue.jsonl, sorted newest-first.
    Returns an empty list if no HIGH-tier predictions have been made yet.

    In production, this endpoint would filter to unreviewed entries only
    and support marking entries as reviewed (PATCH /review-queue/{id}).
    """
    queue_path_override = os.getenv("REVIEW_QUEUE_PATH")
    path = Path(queue_path_override) if queue_path_override else REVIEW_QUEUE_PATH

    entries = read_review_queue(queue_path=path)

    return JSONResponse(
        content={
            "total": len(entries),
            "pending_review": sum(1 for e in entries if not e.get("reviewed", False)),
            "entries": entries,
        }
    )
