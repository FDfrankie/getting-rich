"""Shadow Trading Order Intent & Counterfactual PnL Engine (Phase 16).

Generates OrderIntent records only (no live exchange execution) and enforces fail-closed gates:
- No signal on stale data
- No signal on invalid manifest
- No signal on reconciliation mismatch
- No duplicate OrderIntent
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from data_persistence.repositories import SQLiteRepository


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    strategy: str
    model_version: str
    symbol: str
    features_hash: str
    signal: float
    confidence: float
    target_position: float
    risk_decision: str
    cost_estimate_bps: float
    expected_fill_price: float
    counterfactual_pnl: float
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShadowTradingService:
    def __init__(self, db_path: str | Path) -> None:
        self.repo = db_path if isinstance(db_path, SQLiteRepository) else SQLiteRepository(db_path)

    def create_order_intent(
        self,
        *,
        strategy: str,
        model_version: str,
        symbol: str,
        features: Mapping[str, Any],
        signal: float,
        confidence: float,
        target_position: float,
        current_price: float,
        next_realized_price: float | None = None,
        cost_estimate_bps: float = 6.0,
        data_stale: bool = False,
        manifest_valid: bool = True,
        reconciliation_ok: bool = True,
    ) -> OrderIntent:
        f_hash = hashlib.sha256(json.dumps(dict(features), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()

        if data_stale:
            risk_decision = "BLOCKED_STALE_DATA"
            eff_sig, eff_pos = 0.0, 0.0
        elif not manifest_valid:
            risk_decision = "BLOCKED_INVALID_MANIFEST"
            eff_sig, eff_pos = 0.0, 0.0
        elif not reconciliation_ok:
            risk_decision = "BLOCKED_RECONCILIATION_MISMATCH"
            eff_sig, eff_pos = 0.0, 0.0
        else:
            risk_decision = "SHADOW_APPROVED"
            eff_sig = float(signal)
            eff_pos = float(target_position)

        cf_pnl = 0.0
        if next_realized_price is not None and current_price > 0 and eff_pos != 0.0:
            gross = eff_pos * ((float(next_realized_price) - float(current_price)) / float(current_price))
            cf_pnl = round(gross - abs(eff_pos) * (cost_estimate_bps / 10000.0), 6)

        intent = OrderIntent(
            intent_id=uuid.uuid4().hex,
            strategy=strategy,
            model_version=model_version,
            symbol=symbol,
            features_hash=f_hash,
            signal=eff_sig,
            confidence=float(confidence),
            target_position=eff_pos,
            risk_decision=risk_decision,
            cost_estimate_bps=float(cost_estimate_bps),
            expected_fill_price=float(current_price),
            counterfactual_pnl=cf_pnl,
            created_at=now,
        )

        # Single-transaction idempotent insert with UNIQUE index (eliminates TOCTOU race)
        with self.repo.connect() as conn:
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_order_intents_dedup "
                    "ON order_intents (strategy, symbol, features_hash)"
                )
            except sqlite3.OperationalError:
                pass
            dup = conn.execute(
                "SELECT * FROM order_intents WHERE strategy=? AND symbol=? AND features_hash=? LIMIT 1",
                (strategy, symbol, f_hash),
            ).fetchone()
            if dup:
                return OrderIntent(
                    intent_id=dup["intent_id"],
                    strategy=dup["strategy"],
                    model_version=dup["model_version"],
                    symbol=dup["symbol"],
                    features_hash=dup["features_hash"],
                    signal=float(dup["signal"]),
                    confidence=float(dup["confidence"]),
                    target_position=float(dup["target_position"]),
                    risk_decision="BLOCKED_DUPLICATE_INTENT",
                    cost_estimate_bps=float(dup["cost_estimate_bps"]),
                    expected_fill_price=float(dup["expected_fill_price"] or current_price),
                    counterfactual_pnl=float(dup["counterfactual_pnl"] or 0.0),
                    created_at=dup["created_at"],
                )
            try:
                conn.execute(
                    """
                    INSERT INTO order_intents (
                        intent_id, strategy, model_version, symbol, features_hash,
                        signal, confidence, target_position, risk_decision,
                        cost_estimate_bps, expected_fill_price, counterfactual_pnl, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent.intent_id,
                        intent.strategy,
                        intent.model_version,
                        intent.symbol,
                        intent.features_hash,
                        intent.signal,
                        intent.confidence,
                        intent.target_position,
                        intent.risk_decision,
                        intent.cost_estimate_bps,
                        intent.expected_fill_price,
                        intent.counterfactual_pnl,
                        intent.created_at,
                    ),
                )
            except sqlite3.IntegrityError:
                return OrderIntent(
                    intent_id=intent.intent_id,
                    strategy=strategy,
                    model_version=model_version,
                    symbol=symbol,
                    features_hash=f_hash,
                    signal=eff_sig,
                    confidence=float(confidence),
                    target_position=eff_pos,
                    risk_decision="BLOCKED_DUPLICATE_INTENT",
                    cost_estimate_bps=float(cost_estimate_bps),
                    expected_fill_price=float(current_price),
                    counterfactual_pnl=cf_pnl,
                    created_at=now,
                )
        return intent

ShadowOrderIntent = OrderIntent
