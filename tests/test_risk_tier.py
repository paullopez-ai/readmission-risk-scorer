"""
Tests for src/utils/risk_tier.py — threshold routing and action mapping.
"""

import pytest

from src.utils.risk_tier import classify_risk, get_recommended_actions


# --- classify_risk ---

def test_high_tier_at_threshold():
    """Score exactly at HIGH_THRESHOLD → HIGH."""
    assert classify_risk(0.65) == "HIGH"


def test_high_tier_above_threshold():
    """Score well above threshold → HIGH."""
    assert classify_risk(0.81) == "HIGH"
    assert classify_risk(0.99) == "HIGH"
    assert classify_risk(1.0) == "HIGH"


def test_moderate_tier_in_range():
    """Score in MODERATE range → MODERATE."""
    assert classify_risk(0.35) == "MODERATE"
    assert classify_risk(0.50) == "MODERATE"
    assert classify_risk(0.64) == "MODERATE"


def test_low_tier_below_threshold():
    """Score below MODERATE_THRESHOLD → LOW."""
    assert classify_risk(0.12) == "LOW"
    assert classify_risk(0.00) == "LOW"
    assert classify_risk(0.34) == "LOW"


def test_boundary_just_below_high():
    """Score just below HIGH threshold → MODERATE, not HIGH."""
    assert classify_risk(0.6499) == "MODERATE"


def test_boundary_just_below_moderate():
    """Score just below MODERATE threshold → LOW, not MODERATE."""
    assert classify_risk(0.3499) == "LOW"


# --- get_recommended_actions ---

def test_high_actions_not_empty():
    """HIGH tier returns a non-empty action list."""
    actions = get_recommended_actions("HIGH")
    assert len(actions) > 0


def test_moderate_actions_not_empty():
    """MODERATE tier returns a non-empty action list."""
    actions = get_recommended_actions("MODERATE")
    assert len(actions) > 0


def test_low_actions_not_empty():
    """LOW tier returns a non-empty action list."""
    actions = get_recommended_actions("LOW")
    assert len(actions) > 0


def test_high_actions_mention_urgency():
    """HIGH-tier actions reference urgent care coordination."""
    actions = " ".join(get_recommended_actions("HIGH")).lower()
    assert any(word in actions for word in ["immediate", "social work", "48-hour", "review queue"])


def test_low_actions_mention_standard():
    """LOW-tier actions reference standard discharge protocol."""
    actions = " ".join(get_recommended_actions("LOW")).lower()
    assert any(word in actions for word in ["standard", "follow-up", "routine"])


def test_tiers_have_different_actions():
    """HIGH, MODERATE, and LOW tiers return distinct action sets."""
    high = get_recommended_actions("HIGH")
    moderate = get_recommended_actions("MODERATE")
    low = get_recommended_actions("LOW")
    assert high != moderate
    assert moderate != low
    assert high != low


def test_unknown_tier_returns_fallback():
    """Unknown tier falls back to LOW actions rather than raising."""
    actions = get_recommended_actions("UNKNOWN_TIER")
    assert isinstance(actions, list)
    assert len(actions) > 0
