"""Backward-compatibility alias for worldquant_brain.brain_result_parser."""
from __future__ import annotations

from worldquant_brain.brain_result_parser import (
    parse_and_evaluate_brain_result,
    parsed_to_alpha_metrics,
)

__all__ = ["parse_and_evaluate_brain_result", "parsed_to_alpha_metrics"]
