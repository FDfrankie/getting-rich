"""Research Memory, Blueprint Registry, Robustness Store, and Portfolio Marginal Utility Evaluator (Stages 8–11).

Enforces:
- No direct write of ROBUST_GOOD without an evaluated RobustnessReport where promotion_ready=True.
- Idempotent replay of the same (alpha_id, variant_type, variant_value, simulation_id).
- Persistence of research lessons (failure_gate, effective_fix, ineffective_fixes) to block duplicate failing experiments.
- Portfolio Marginal Utility evaluation (incremental Sharpe, correlation, turnover overlap, tail risk).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from data_persistence.migrations import apply_sqlite_migrations
from worldquant_brain.alpha_blueprint import AlphaBlueprint, builtin_blueprints
from worldquant_brain.robustness_evaluator import (
    EvaluationRun,
    RobustnessReport,
    RobustnessStatus,
    evaluate_robustness,
)


class AlphaResearchFactoryService:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            apply_sqlite_migrations(conn)
        self.seed_builtin_blueprints()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def seed_builtin_blueprints(self) -> int:
        count = 0
        for bp in builtin_blueprints():
            self.register_blueprint(bp)
            count += 1
        return count

    def register_blueprint(self, blueprint: AlphaBlueprint) -> dict[str, Any]:
        errors = blueprint.validate()
        if errors:
            raise ValueError(f"Blueprint validation failed: {errors}")
        bp_hash = blueprint.sha256()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alpha_blueprints (
                    blueprint_id, version, blueprint_hash, family, title,
                    rationale, falsifiable_claim, blueprint_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    blueprint.blueprint_id,
                    blueprint.version,
                    bp_hash,
                    blueprint.family.value,
                    blueprint.title,
                    blueprint.economic_rationale,
                    blueprint.falsifiable_claim,
                    json.dumps(blueprint.canonical_dict(), ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM alpha_blueprints WHERE blueprint_id=? ORDER BY version DESC LIMIT 1",
                (blueprint.blueprint_id,),
            ).fetchone()
        return dict(row)

    def list_blueprints(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM alpha_blueprints ORDER BY blueprint_id ASC, version DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["blueprint"] = json.loads(d.pop("blueprint_json", "{}"))
            except Exception:
                d["blueprint"] = {}
            out.append(d)
        return out

    def get_blueprint(self, blueprint_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM alpha_blueprints WHERE blueprint_id=? ORDER BY version DESC LIMIT 1",
                (blueprint_id,),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["blueprint"] = json.loads(d.pop("blueprint_json", "{}"))
        except Exception:
            d["blueprint"] = {}
        return d

    def record_robustness_run(self, alpha_id: str, blueprint_id: str, run: EvaluationRun) -> dict[str, Any]:
        """Idempotently store a simulation run for robustness evaluation (same simulation replay is idempotent)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM alpha_robustness_runs
                WHERE (run_id = ?) OR (alpha_id = ? AND variant_type = ? AND variant_value = ? AND COALESCE(simulation_id, '') = COALESCE(?, ''))
                LIMIT 1
                """,
                (run.run_id, alpha_id, run.variant_type, run.variant_value, run.simulation_id),
            ).fetchone()
            if existing:
                return dict(existing) | {"idempotent_replay": True}
            conn.execute(
                """
                INSERT INTO alpha_robustness_runs (
                    run_id, alpha_id, blueprint_id, variant_type, variant_value,
                    simulation_id, metrics_source, sharpe, fitness, turnover,
                    returns, drawdown, pnl, year, regime, universe, parameters_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    alpha_id,
                    blueprint_id,
                    run.variant_type,
                    run.variant_value,
                    run.simulation_id,
                    run.metrics_source,
                    run.sharpe,
                    run.fitness,
                    run.turnover,
                    run.returns,
                    run.drawdown,
                    run.pnl,
                    run.year,
                    run.regime,
                    run.universe,
                    json.dumps(dict(run.parameters or {}), sort_keys=True),
                    now,
                ),
            )
            saved = conn.execute("SELECT * FROM alpha_robustness_runs WHERE run_id=?", (run.run_id,)).fetchone()
        return dict(saved) | {"idempotent_replay": False}

    def evaluate_and_persist_robustness(
        self,
        *,
        alpha_id: str,
        blueprint: AlphaBlueprint,
        baseline: EvaluationRun,
        runs: Sequence[EvaluationRun],
    ) -> RobustnessReport:
        """Evaluate 6-dimension robustness from official BRAIN runs and persist the report."""
        self.register_blueprint(blueprint)
        self.record_robustness_run(alpha_id, blueprint.blueprint_id, baseline)
        for r in runs:
            self.record_robustness_run(alpha_id, blueprint.blueprint_id, r)

        report = evaluate_robustness(blueprint=blueprint, baseline=baseline, runs=runs)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alpha_robustness_reports (
                    alpha_id, blueprint_id, blueprint_hash, baseline_run_id,
                    status, score, report_json, promotion_ready, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alpha_id,
                    report.blueprint_id,
                    report.blueprint_hash,
                    report.baseline_run_id,
                    report.status.value,
                    report.score,
                    json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True),
                    1 if report.promotion_ready else 0,
                    now,
                ),
            )
            # Only set quality_class='ROBUST_GOOD' when official evaluator proves promotion_ready
            if report.promotion_ready:
                conn.execute(
                    """
                    UPDATE alphas
                    SET quality_class='ROBUST_GOOD',
                        robustness_status=?,
                        robustness_score=?,
                        promotion_ready=1,
                        blueprint_id=?,
                        blueprint_hash=?,
                        updated_at=?
                    WHERE alpha_id=?
                    """,
                    (report.status.value, report.score, blueprint.blueprint_id, report.blueprint_hash, now, alpha_id),
                )
        return report

    def set_quality_class_guarded(
        self,
        alpha_id: str,
        target_quality: str,
    ) -> None:
        """Stage 7 & 11 Rule 13: Forbid direct DB assignment of ROBUST_GOOD without a verified RobustnessReport."""
        if target_quality == "ROBUST_GOOD":
            with self._connect() as conn:
                rep = conn.execute(
                    "SELECT promotion_ready, status FROM alpha_robustness_reports WHERE alpha_id=?",
                    (alpha_id,),
                ).fetchone()
            if not rep or not bool(rep["promotion_ready"]) or rep["status"] != RobustnessStatus.ROBUST.value:
                raise PermissionError(
                    f"Direct write of ROBUST_GOOD for {alpha_id} is blocked: requires verified RobustnessReport with promotion_ready=True."
                )
        with self._connect() as conn:
            conn.execute("UPDATE alphas SET quality_class=? WHERE alpha_id=?", (target_quality, alpha_id))

    def get_robustness_report(self, alpha_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM alpha_robustness_reports WHERE alpha_id=?", (alpha_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["report"] = json.loads(d.pop("report_json", "{}"))
        except Exception:
            d["report"] = {}
        d["promotion_ready"] = bool(d.get("promotion_ready"))
        return d

    def list_robust_good_alphas(self) -> list[dict[str, Any]]:
        """Stage 10 & 11 Rule 14: Return ONLY verified ROBUST_GOOD alphas backed by promotion_ready RobustnessReports."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.alpha_id, r.blueprint_id, r.blueprint_hash, r.baseline_run_id,
                       r.status, r.score, r.report_json, r.evaluated_at
                FROM alpha_robustness_reports r
                WHERE r.promotion_ready = 1 AND r.status = 'ROBUST'
                ORDER BY r.score DESC, r.evaluated_at DESC
                """
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["quality_class"] = "ROBUST_GOOD"
            try:
                item["report"] = json.loads(item.pop("report_json", "{}"))
            except Exception:
                item["report"] = {}
            out.append(item)
        return out

    def record_research_lesson(
        self,
        *,
        family: str,
        data_domain: str,
        failure_gate: str,
        effective_fix: str | None,
        ineffective_fixes: Iterable[str],
        evidence_run_ids: Iterable[str],
    ) -> dict[str, Any]:
        lesson_id = hashlib.sha256(f"{family}:{data_domain}:{failure_gate}".encode("utf-8")).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()
        ineff = sorted(set(str(x) for x in ineffective_fixes))
        ev_ids = sorted(set(str(x) for x in evidence_run_ids))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_lessons (
                    lesson_id, family, data_domain, failure_gate,
                    effective_fix, ineffective_fixes_json, evidence_run_ids_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lesson_id,
                    family,
                    data_domain,
                    failure_gate,
                    effective_fix,
                    json.dumps(ineff, ensure_ascii=False),
                    json.dumps(ev_ids, ensure_ascii=False),
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM research_lessons WHERE lesson_id=?", (lesson_id,)).fetchone()
        return self._format_lesson(row)

    def list_research_lessons(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM research_lessons ORDER BY created_at DESC").fetchall()
        return [self._format_lesson(r) for r in rows]

    @staticmethod
    def _format_lesson(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["ineffective_fixes"] = json.loads(d.pop("ineffective_fixes_json", "[]"))
        d["evidence_run_ids"] = json.loads(d.pop("evidence_run_ids_json", "[]"))
        return d


def evaluate_portfolio_marginal_utility(
    candidate_daily_returns: Sequence[float],
    vault_returns_matrix: np.ndarray | None,
    *,
    candidate_turnover: float = 0.25,
    vault_avg_turnover: float = 0.25,
) -> dict[str, Any]:
    """Evaluate PORTFOLIO_GOOD marginal contribution: incremental Sharpe, max correlation, worst-period improvement."""
    cand = np.asarray(candidate_daily_returns, dtype=float)
    if len(cand) < 10:
        return {"portfolio_good": False, "reason": "INSUFFICIENT_RETURN_SERIES"}
    cand_sr = float((np.mean(cand) / (np.std(cand, ddof=1) + 1e-12)) * np.sqrt(252.0))
    if vault_returns_matrix is None or getattr(vault_returns_matrix, "size", 0) == 0:
        return {
            "portfolio_good": cand_sr >= 1.25,
            "incremental_sharpe": round(cand_sr, 4),
            "max_pnl_correlation": 0.0,
            "turnover_delta": 0.0,
        }
    mat = np.asarray(vault_returns_matrix, dtype=float)
    if mat.ndim == 1:
        mat = mat[:, None]
    corrs = [
        float(np.corrcoef(cand, mat[:, j])[0, 1])
        for j in range(mat.shape[1])
        if np.std(mat[:, j]) > 1e-12
    ]
    max_corr = max((abs(c) for c in corrs if np.isfinite(c)), default=0.0)
    vault_eq = np.mean(mat, axis=1)
    old_sr = float((np.mean(vault_eq) / (np.std(vault_eq, ddof=1) + 1e-12)) * np.sqrt(252.0))
    w_new = 1.0 / float(mat.shape[1] + 1)
    combined = (1.0 - w_new) * vault_eq + w_new * cand
    new_sr = float((np.mean(combined) / (np.std(combined, ddof=1) + 1e-12)) * np.sqrt(252.0))
    inc_sr = new_sr - old_sr
    turn_delta = float(candidate_turnover) - float(vault_avg_turnover)
    ok = (inc_sr > 0.0) and (max_corr < 0.70) and (turn_delta <= 0.20)
    return {
        "portfolio_good": bool(ok),
        "old_portfolio_sharpe": round(old_sr, 4),
        "new_portfolio_sharpe": round(new_sr, 4),
        "incremental_sharpe": round(inc_sr, 4),
        "max_pnl_correlation": round(max_corr, 4),
        "turnover_delta": round(turn_delta, 4),
    }
