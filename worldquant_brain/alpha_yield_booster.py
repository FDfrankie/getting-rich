"""Backward-compatibility alias for worldquant_brain.alpha_miner."""
from __future__ import annotations

from worldquant_brain.alpha_miner import (
    _COMBO_PAIRS,
    _HIGH_YIELD_ATOMS,
    generate_orthogonal_combo_candidate,
    generate_smart_settings,
    precision_repair_near_miss,
)

__all__ = [
    "_COMBO_PAIRS",
    "_HIGH_YIELD_ATOMS",
    "generate_orthogonal_combo_candidate",
    "generate_smart_settings",
    "precision_repair_near_miss",
]
