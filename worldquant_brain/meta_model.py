"""Backward-compatibility alias for worldquant_brain.alpha_repository."""
from __future__ import annotations

from worldquant_brain.alpha_repository import (
    MODEL_DESIGNATION,
    compute_ast_family_hash,
    extract_expression_features,
    split_family_chronological,
)

__all__ = [
    "MODEL_DESIGNATION",
    "compute_ast_family_hash",
    "extract_expression_features",
    "split_family_chronological",
]
