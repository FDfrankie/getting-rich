"""Backward-compatibility alias for worldquant_brain.submission_pipeline."""
from __future__ import annotations

from worldquant_brain.submission_pipeline import (
    KILL_SWITCH_ENV,
    REQUIRED_SUBMIT_ACK,
    SUBMIT_ACK_ENV,
    AlphaStatus,
    BrainAutoSubmitGate,
    logger,
    migrate_alpha_status,
)

__all__ = [
    "AlphaStatus",
    "BrainAutoSubmitGate",
    "KILL_SWITCH_ENV",
    "REQUIRED_SUBMIT_ACK",
    "SUBMIT_ACK_ENV",
    "logger",
    "migrate_alpha_status",
]
