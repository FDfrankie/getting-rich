"""PnL-based Correlation Semantics & Evidence Repository.

Correlation in WorldQuant BRAIN is PnL Graph Correlation — NOT expression similarity.
Same-expression D0 vs D1 Alphas must be evaluated via PnL correlation rather than assumed redundant.
Expression, field, and operator similarity are strictly Research Novelty / Pre-screen proxies and
NEVER substitute for official Self-correlation.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator


class CorrelationType(StrEnum):
    SELF = "SELF"
    PROD = "PROD"
    INNER = "INNER"
    PARENT_CHILD = "PARENT_CHILD"
    D0_D1 = "D0_D1"
    LOCAL_PNL_PROXY = "LOCAL_PNL_PROXY"
    STRUCTURAL_PROXY = "STRUCTURAL_PROXY"


class CorrelationSource(StrEnum):
    BRAIN_SELF_CORRELATION = "BRAIN_SELF_CORRELATION"
    BRAIN_PROD_CORRELATION = "BRAIN_PROD_CORRELATION"
    BRAIN_ALPHA_LIST = "BRAIN_ALPHA_LIST"
    BRAIN_GENERATE_SELF_CORRELATION = "BRAIN_GENERATE_SELF_CORRELATION"
    LOCAL_PNL = "LOCAL_PNL"
    EXPRESSION_FINGERPRINT = "EXPRESSION_FINGERPRINT"
    UNKNOWN = "UNKNOWN"


class CorrelationScope(StrEnum):
    OWN_ALPHA_POOL = "OWN_ALPHA_POOL"
    BRAIN_ALPHA_POOL = "BRAIN_ALPHA_POOL"


@dataclass(frozen=True, slots=True)
class AlphaCorrelationEvidence:
    candidate_alpha_id: str
    compared_alpha_id: str | None
    correlation_type: CorrelationType
    correlation_source: CorrelationSource
    correlation_value: float | None
    candidate_delay: int | None
    compared_delay: int | None
    pnl_period_start: str | None
    pnl_period_end: str | None
    observations: int | None
    official: bool
    checked_at: str
    expression_similarity: float | None = None
    field_overlap: float | None = None
    operator_overlap: float | None = None
    scope: CorrelationScope = CorrelationScope.OWN_ALPHA_POOL
    expired: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["correlation_type"] = self.correlation_type.value
        d["correlation_source"] = self.correlation_source.value
        d["scope"] = self.scope.value
        return d


def official_self_correlation_passed(evidence: AlphaCorrelationEvidence | None) -> bool:
    """Only non-expired official SELF correlation < 0.70 can pass the Submission Gate."""
    if evidence is None or evidence.expired:
        return False
    return (
        bool(evidence.official)
        and evidence.correlation_type == CorrelationType.SELF
        and evidence.correlation_source
        in {
            CorrelationSource.BRAIN_SELF_CORRELATION,
            CorrelationSource.BRAIN_GENERATE_SELF_CORRELATION,
        }
        and evidence.correlation_value is not None
        and float(evidence.correlation_value) < 0.70
    )


def resolve_correlation_scope(*, is_consultant: bool) -> CorrelationScope:
    return CorrelationScope.BRAIN_ALPHA_POOL if is_consultant else CorrelationScope.OWN_ALPHA_POOL


@dataclass(frozen=True, slots=True)
class DelayPairMetrics:
    expression_hash: str
    d0_alpha_id: str
    d1_alpha_id: str
    d0_sharpe: float
    d1_sharpe: float
    d0_fitness: float
    d1_fitness: float
    pnl_correlation: float | None
    pnl_correlation_official: bool
    d0_portfolio_utility: float
    d1_portfolio_utility: float


@dataclass(frozen=True, slots=True)
class DelayPairDecision:
    decision: str
    preferred_alpha_id: str | None
    reason: str


def evaluate_delay_pair(metrics: DelayPairMetrics) -> DelayPairDecision:
    if metrics.pnl_correlation is None or not metrics.pnl_correlation_official:
        return DelayPairDecision(
            decision="CORRELATION_NOT_VERIFIED",
            preferred_alpha_id=None,
            reason="Same-expression D0 and D1 must be judged by official PnL correlation, not expression similarity.",
        )

    if metrics.pnl_correlation < 0.50:
        return DelayPairDecision(
            decision="KEEP_BOTH_FOR_PORTFOLIO_REVIEW",
            preferred_alpha_id=None,
            reason="D0 and D1 PnL series show low correlation (<0.50); review both for Merged Performance.",
        )

    preferred = (
        metrics.d0_alpha_id
        if metrics.d0_portfolio_utility > metrics.d1_portfolio_utility
        else metrics.d1_alpha_id
    )

    if metrics.pnl_correlation < 0.70:
        return DelayPairDecision(
            decision="KEEP_PREFERRED_HOLD_OTHER",
            preferred_alpha_id=preferred,
            reason="D0 and D1 are moderately correlated (<0.70); prioritize higher Portfolio Utility.",
        )

    return DelayPairDecision(
        decision="HIGHLY_CORRELATED_KEEP_ONE",
        preferred_alpha_id=preferred,
        reason="D0 and D1 PnL correlation >= 0.70; keep only the higher Portfolio Utility candidate.",
    )


def evaluate_parent_child_repair(
    *,
    parent_child_pnl_corr: float | None,
    official_corr: bool,
    target_gate_resolved: bool,
    parent_sharpe: float,
    child_sharpe: float,
    performance_improvement_ratio: float,
) -> tuple[bool, str]:
    """Repair is only successful when official corr/gate improves without severe Sharpe degradation."""
    if parent_child_pnl_corr is None or not official_corr:
        return False, "NOT_VERIFIED"
    if child_sharpe < parent_sharpe - 0.25 or child_sharpe < 1.25:
        return False, "SHARPE_SEVERELY_DEGRADED"
    if parent_child_pnl_corr >= 0.70 and performance_improvement_ratio < 0.10:
        return False, "INSUFFICIENT_PERFORMANCE_OVERRIDE_BELOW_10PCT"
    if not target_gate_resolved:
        return False, "TARGET_GATE_NOT_RESOLVED"
    return True, "REPAIR_SUCCEEDED"


CORRELATION_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS alpha_correlation_evidence (
    correlation_evidence_id TEXT PRIMARY KEY,
    candidate_alpha_id TEXT NOT NULL,
    compared_alpha_id TEXT,
    correlation_type TEXT NOT NULL CHECK (
        correlation_type IN (
            'SELF', 'PROD', 'INNER', 'PARENT_CHILD',
            'D0_D1', 'LOCAL_PNL_PROXY', 'STRUCTURAL_PROXY'
        )
    ),
    correlation_source TEXT NOT NULL,
    correlation_value REAL CHECK (
        correlation_value IS NULL OR correlation_value BETWEEN -1.0 AND 1.0
    ),
    candidate_delay INTEGER CHECK (
        candidate_delay IS NULL OR candidate_delay IN (0, 1)
    ),
    compared_delay INTEGER CHECK (
        compared_delay IS NULL OR compared_delay IN (0, 1)
    ),
    pnl_period_start TEXT,
    pnl_period_end TEXT,
    observations INTEGER CHECK (observations IS NULL OR observations >= 2),
    official INTEGER NOT NULL DEFAULT 0 CHECK (official IN (0, 1)),
    expression_similarity REAL,
    field_overlap REAL,
    operator_overlap REAL,
    scope TEXT NOT NULL DEFAULT 'OWN_ALPHA_POOL',
    raw_response_json TEXT NOT NULL DEFAULT '{}',
    checked_at TEXT NOT NULL,
    UNIQUE (
        candidate_alpha_id,
        compared_alpha_id,
        correlation_type,
        correlation_source,
        pnl_period_start,
        pnl_period_end
    )
);

CREATE INDEX IF NOT EXISTS idx_correlation_candidate
ON alpha_correlation_evidence (candidate_alpha_id, correlation_type, checked_at);

CREATE INDEX IF NOT EXISTS idx_correlation_pair
ON alpha_correlation_evidence (candidate_alpha_id, compared_alpha_id, correlation_type);
"""


class CorrelationEvidenceRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.database_path, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def migrate(self) -> None:
        with self.connection() as con:
            con.executescript(CORRELATION_SCHEMA_SQL)

    def save_evidence(
        self,
        ev: AlphaCorrelationEvidence,
        *,
        raw_response: dict[str, Any] | None = None,
    ) -> str:
        self.migrate()
        eid = f"corr_{uuid.uuid4().hex[:16]}"
        raw_json = json.dumps(raw_response or {}, ensure_ascii=False, sort_keys=True)
        with self.connection() as con:
            con.execute(
                """
                INSERT INTO alpha_correlation_evidence (
                    correlation_evidence_id, candidate_alpha_id, compared_alpha_id,
                    correlation_type, correlation_source, correlation_value,
                    candidate_delay, compared_delay, pnl_period_start, pnl_period_end,
                    observations, official, expression_similarity, field_overlap,
                    operator_overlap, scope, raw_response_json, checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    candidate_alpha_id, compared_alpha_id, correlation_type,
                    correlation_source, pnl_period_start, pnl_period_end
                ) DO UPDATE SET
                    correlation_value = excluded.correlation_value,
                    official = excluded.official,
                    observations = excluded.observations,
                    raw_response_json = excluded.raw_response_json,
                    checked_at = excluded.checked_at
                """,
                (
                    eid,
                    ev.candidate_alpha_id,
                    ev.compared_alpha_id,
                    ev.correlation_type.value,
                    ev.correlation_source.value,
                    ev.correlation_value,
                    ev.candidate_delay,
                    ev.compared_delay,
                    ev.pnl_period_start,
                    ev.pnl_period_end,
                    ev.observations,
                    1 if ev.official else 0,
                    ev.expression_similarity,
                    ev.field_overlap,
                    ev.operator_overlap,
                    ev.scope.value,
                    raw_json,
                    ev.checked_at,
                ),
            )
        return eid

    def list_for_candidate(
        self,
        candidate_alpha_id: str,
        *,
        correlation_type: CorrelationType | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        self.migrate()
        with self.connection() as con:
            if correlation_type is not None:
                rows = con.execute(
                    """
                    SELECT * FROM alpha_correlation_evidence
                    WHERE candidate_alpha_id = ? AND correlation_type = ?
                    ORDER BY correlation_value DESC, checked_at DESC LIMIT ?
                    """,
                    (candidate_alpha_id, correlation_type.value, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT * FROM alpha_correlation_evidence
                    WHERE candidate_alpha_id = ?
                    ORDER BY checked_at DESC LIMIT ?
                    """,
                    (candidate_alpha_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]
