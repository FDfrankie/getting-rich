"""AlphaRepository for canonical Alpha lifecycle & experience memory persistence."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from data_persistence.repositories import SQLiteRepository
from worldquant_brain.alpha_lifecycle import AlphaState, normalize_legacy_state, require_transition


def compute_ast_family_hash(expression: str) -> str:
    """Delegate AST family clustering to the single Canonical FastExpr AST source of truth."""
    from worldquant_brain.canonical_ast import parse_fastexpr

    return parse_fastexpr(expression).ast_family_hash


class AlphaRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.store = SQLiteRepository(db_path)

    def save_candidate(self, alpha: Mapping[str, Any]) -> dict[str, Any]:
        expr = str(alpha.get("expression") or "").strip()
        payload = dict(alpha)
        try:
            from worldquant_brain.canonical_ast import parse_fastexpr
            fp = parse_fastexpr(expr)
            payload["expression_hash"] = fp.exact_hash or hashlib.sha256(expr.encode("utf-8")).hexdigest()
            payload["ast_family_hash"] = fp.ast_family_hash
        except Exception:
            payload["expression_hash"] = hashlib.sha256(expr.encode("utf-8")).hexdigest()
            payload["ast_family_hash"] = compute_ast_family_hash(expr)
        return self.store.upsert_alpha(payload)

    def transition_state(
        self,
        alpha_id: str,
        target_state: AlphaState,
        *,
        actor: str = "system",
        reason: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM alphas WHERE alpha_id=?", (alpha_id,)).fetchone()
            if not row:
                raise KeyError(f"Alpha not found: {alpha_id}")
            current = normalize_legacy_state(
                row["status"],
                has_official_simulation=bool(row["brain_simulation_id"]),
            )
            require_transition(current, target_state)
            ev = dict(evidence or {})
            conn.execute(
                """
                UPDATE alphas
                SET status=?,
                    brain_alpha_id=COALESCE(?, brain_alpha_id),
                    brain_simulation_id=COALESCE(?, brain_simulation_id),
                    submitted_at=COALESCE(?, submitted_at),
                    updated_at=?
                WHERE alpha_id=?
                """,
                (
                    target_state.value,
                    ev.get("brain_alpha_id"),
                    ev.get("brain_simulation_id"),
                    now if target_state == AlphaState.SUBMITTED else None,
                    now,
                    alpha_id,
                ),
            )
            updated = dict(conn.execute("SELECT * FROM alphas WHERE alpha_id=?", (alpha_id,)).fetchone())
        self.store.record_audit_event(
            "ALPHA_STATE_TRANSITION",
            actor,
            alpha_id,
            current.value,
            target_state.value,
            reason,
            evidence,
        )
        return updated


class AlphaService:
    """High-level Alpha lifecycle service orchestrating Utility V1 evaluation and experience memory."""

    def __init__(self, db_path: str | Path) -> None:
        self.repo = AlphaRepository(db_path)

    def register_and_score(self, candidate: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        from worldquant_brain.candidate_utility import evaluate_candidate

        saved = self.repo.save_candidate(candidate)
        enriched = dict(candidate) | {
            "expression_hash": saved["expression_hash"],
            "ast_family_hash": compute_ast_family_hash(saved["expression"]),
            "duplicate_expression": bool(saved.get("idempotent_duplicate")),
            "already_submitted": bool(saved.get("submitted_at")),
        }
        util = evaluate_candidate(enriched)
        return saved, util


MODEL_DESIGNATION = "BRAIN_ALPHA_META_MODEL"


def extract_expression_features(record: Mapping[str, Any]) -> dict[str, Any]:
    expr = str(record.get("expression") or "").strip()
    ops = re.findall(r"([a-z_][a-z0-9_]*)\s*\(", expr.lower())
    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return {
        "ast_family_hash": str(record.get("ast_family_hash") or compute_ast_family_hash(expr)),
        "operator_count": len(ops),
        "ast_depth": max_depth,
        "expression_complexity": len(expr),
        "has_group_neutralize": int("group_neutralize" in ops),
        "has_decay": int("ts_decay_linear" in ops or "hump" in ops),
        "decay": float(record.get("decay") or 4.0),
        "truncation": float(record.get("truncation") or 0.08),
        "completed_at": str(record.get("completed_at") or record.get("created_at") or "1970-01-01T00:00:00Z"),
    }


def split_family_chronological(
    records: Iterable[Mapping[str, Any]],
    *,
    train_frac: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split chronologically by completed_at while forcing same AST family into the same split."""
    rows = [dict(r) | extract_expression_features(r) for r in records]
    rows.sort(key=lambda r: r["completed_at"])
    if not rows:
        return [], []
    cutoff_idx = max(1, int(len(rows) * train_frac))
    train_families: set[str] = set()
    for r in rows[:cutoff_idx]:
        train_families.add(r["ast_family_hash"])

    train_split: list[dict[str, Any]] = []
    test_split: list[dict[str, Any]] = []
    for r in rows:
        if r["ast_family_hash"] in train_families:
            train_split.append(r)
        else:
            test_split.append(r)
    return train_split, test_split
