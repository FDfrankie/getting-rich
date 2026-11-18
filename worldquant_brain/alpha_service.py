"""High-level Alpha lifecycle service orchestrating Utility V1 evaluation and experience memory."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from worldquant_brain.alpha_lifecycle import AlphaState
from worldquant_brain.alpha_repository import AlphaRepository, compute_ast_family_hash
from worldquant_brain.candidate_utility import UtilityResult, evaluate_candidate


class AlphaService:
    def __init__(self, db_path: str | Path) -> None:
        self.repo = AlphaRepository(db_path)

    def register_and_score(self, candidate: Mapping[str, Any]) -> tuple[dict[str, Any], UtilityResult]:
        saved = self.repo.save_candidate(candidate)
        enriched = dict(candidate) | {
            "expression_hash": saved["expression_hash"],
            "ast_family_hash": compute_ast_family_hash(saved["expression"]),
            "duplicate_expression": bool(saved.get("idempotent_duplicate")),
            "already_submitted": bool(saved.get("submitted_at")),
        }
        util = evaluate_candidate(enriched)
        return saved, util
