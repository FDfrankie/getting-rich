"""Backward-compatibility alias for worldquant_brain.submission_pipeline."""
from __future__ import annotations

from worldquant_brain.submission_pipeline import PolicyDecision, SubmitAllAvailablePolicy

__all__ = ["PolicyDecision", "SubmitAllAvailablePolicy"]
