"""Complete Trial Registry & Deflated Sharpe Ratio (DSR) multiple-testing guard."""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from worldquant_brain.sqlite_resilience import ensure_schema_from_ddl


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Rational approximation (Peter J. Acklam) for standard normal inverse CDF."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


@dataclass(frozen=True, slots=True)
class DSRResult:
    observed_sharpe: float
    trial_count: int
    expected_max_sharpe: float
    dsr_probability: float
    passed: bool


def deflated_sharpe_ratio(
    observed_sharpe: float,
    trial_count: int,
    *,
    n_observations: int = 252,
    sharpe_std_across_trials: float = 0.5,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    threshold: float = 0.95,
    avg_trial_corr: float = 0.0,
) -> DSRResult:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio accounting for correlated effective trials N_eff."""
    n_trials = max(1, int(trial_count))
    n_obs = max(10, int(n_observations))
    sr = float(observed_sharpe)
    sr_daily = sr / math.sqrt(252.0)
    rho = min(max(float(avg_trial_corr), 0.0), 1.0)
    n_eff = max(1.0, rho + (1.0 - rho) * float(n_trials))

    if n_eff <= 1.0 + 1e-9:
        sr0 = 0.0
    else:
        euler_gamma = 0.5772156649
        z1 = _norm_ppf(1.0 - 1.0 / n_eff)
        z2 = _norm_ppf(1.0 - 1.0 / (n_eff * math.e))
        sr0_annual = float(sharpe_std_across_trials) * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
        sr0 = sr0_annual / math.sqrt(252.0)

    denom_sq = max(1e-9, 1.0 - skewness * sr_daily + ((kurtosis - 1.0) / 4.0) * (sr_daily ** 2))
    z_stat = ((sr_daily - sr0) * math.sqrt(n_obs - 1.0)) / math.sqrt(denom_sq)
    prob = _norm_cdf(z_stat)
    return DSRResult(
        observed_sharpe=round(sr, 4),
        trial_count=n_trials,
        expected_max_sharpe=round(sr0 * math.sqrt(252.0), 4),
        dsr_probability=round(prob, 4),
        passed=prob >= float(threshold),
    )


class TrialRegistry:
    """Persists EVERY alpha trial (winners and losers) so DSR reflects true multiple-testing count."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path, timeout=15.0)) as conn:
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=15000")
            with conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS alpha_trial_registry (
                        trial_id TEXT PRIMARY KEY,
                        parent_trial_id TEXT,
                        blueprint_id TEXT,
                        expression_hash TEXT NOT NULL,
                        settings_hash TEXT NOT NULL,
                        objective TEXT NOT NULL,
                        sharpe REAL,
                        fitness REAL,
                        turnover REAL,
                        accepted INTEGER NOT NULL DEFAULT 0,
                        metrics_json TEXT,
                        created_at TEXT NOT NULL
                    )"""
                )
                ensure_schema_from_ddl(conn, 'CREATE TABLE IF NOT EXISTS alpha_trial_registry (\n                        trial_id TEXT PRIMARY KEY,\n                        parent_trial_id TEXT,\n                        blueprint_id TEXT,\n                        expression_hash TEXT NOT NULL,\n                        settings_hash TEXT NOT NULL,\n                        objective TEXT NOT NULL,\n                        sharpe REAL,\n                        fitness REAL,\n                        turnover REAL,\n                        accepted INTEGER NOT NULL DEFAULT 0,\n                        metrics_json TEXT,\n                        created_at TEXT NOT NULL\n                    )')


    def record_trial(
        self,
        *,
        expression_hash: str,
        settings_hash: str,
        objective: str = "BALANCED",
        blueprint_id: str | None = None,
        parent_trial_id: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        accepted: bool = False,
    ) -> str:
        tid = f"trl_{uuid.uuid4().hex[:16]}"
        m = dict(metrics or {})
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        try:
            with conn:
                conn.execute(
                    """INSERT INTO alpha_trial_registry
                    (trial_id, parent_trial_id, blueprint_id, expression_hash, settings_hash, objective, sharpe, fitness, turnover, accepted, metrics_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        tid,
                        parent_trial_id,
                        blueprint_id,
                        expression_hash,
                        settings_hash,
                        objective,
                        m.get("sharpe"),
                        m.get("fitness"),
                        m.get("turnover"),
                        1 if accepted else 0,
                        json.dumps(m, ensure_ascii=False, default=str),
                        now,
                    ),
                )
        finally:
            conn.close()
        return tid

    def count_trials(self, *, blueprint_id: str | None = None) -> int:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        try:
            if blueprint_id:
                row = conn.execute("SELECT COUNT(*) FROM alpha_trial_registry WHERE blueprint_id = ?", (blueprint_id,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM alpha_trial_registry").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
