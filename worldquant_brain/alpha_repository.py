"""AlphaRepository for canonical Alpha lifecycle & experience memory persistence."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from data_persistence.repositories import SQLiteRepository
from worldquant_brain.alpha_lifecycle import AlphaState, normalize_legacy_state, require_transition


def compute_ast_family_hash(expression: str) -> str:
    """Normalize numeric window literals so variants like ts_decay_linear(X,3/5/8) share one AST family cluster."""
    expr = re.sub(r"\s+", "", (expression or "").strip().lower())
    skeleton = re.sub(r"(?<=[,(])[-+]?\d+(?:\.\d+)?(?=[,)])", "N", expr)
    return hashlib.sha256(skeleton.encode("utf-8")).hexdigest()[:16]


class AlphaRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.store = SQLiteRepository(db_path)

    def save_candidate(self, alpha: Mapping[str, Any]) -> dict[str, Any]:
        expr = str(alpha.get("expression") or "").strip()
        payload = dict(alpha)
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
