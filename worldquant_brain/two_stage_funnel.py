"""RL-Guided Two-Stage Alpha Discovery Funnel with Reference DB Repulsion & Attraction.

Design principles:
  1. Reference DB (`funnel_reference_db`) is for RL learning & directional guidance,
     NEVER for re-mining old seeds:
       - REJECTED / SUBMIT_FAILED records train negative RL rewards and act as
         repulsion zones (new formulas too structurally similar are skipped).
       - PASSED / SUBMITTED records train positive RL rewards for (dataset, probe)
         selection, while blocking parameter-only clones of already-submitted alphas.
  2. Immediate In-Session Near-Miss Refinement:
       - When a Stage-1 simulation lands a NEAR_MISS (|Sharpe| >= 0.90), fine-tuning
         variants are queued immediately in the current session (not re-mined from
         historical DB later).
  3. UCB1 Contextual Bandit RL (`funnel_rl_policy`) balances exploring fresh
     (dataset, probe) pairs and exploiting high-yield directions.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from worldquant_brain.alpha_miner import validate_candidate_expression
from worldquant_brain.multi_sim_client import DEFAULT_SIM_SETTINGS

STAGE1_PROBES: tuple[tuple[str, str], ...] = (
    ("rank_bf", "rank(ts_backfill({f}, 20))"),
    ("zscore60", "ts_zscore(ts_backfill({f}, 20), 60)"),
    ("grp_zscore", "group_zscore(ts_backfill({f}, 20), industry)"),
    ("grp_delta20", "group_rank(ts_delta(ts_backfill({f}, 20), 20), subindustry)"),
    ("tsrank120", "winsorize(ts_rank(ts_backfill({f}, 20), 120), std=4)"),
    ("decay5_bf", "winsorize(rank(ts_decay_linear(ts_backfill({f}, 20), 5)), std=4)"),
    ("trade_vol", "trade_when(volume > adv20, rank(ts_backfill({f}, 20)), -1)"),
    ("corr_ret20", "ts_corr(ts_backfill({f}, 20), returns, 20)"),
    ("spow_rank", "signed_power(rank(ts_backfill({f}, 20)) - 0.5, 2)"),
    ("cap_norm_gauss", "quantile(divide(ts_backfill({f}, 20), cap), driver=\"gaussian\")"),
    ("cap_bucket_neut", "group_neutralize(rank(ts_backfill({f}, 20)), bucket(rank(cap), range=\"0, 1, 0.1\"))"),
    ("last_revision", "rank(ts_backfill({f}, 20) - last_diff_value(ts_backfill({f}, 20), 252))"),
    ("ts_av_diff60", "quantile(ts_av_diff(ts_backfill({f}, 20), 60), driver=\"gaussian\")"),
    ("reg_resid60", "rank(ts_regression(ts_backfill({f}, 20), returns, 60, lag=0, rettype=0))"),
)


def _format_probe(template: str, raw_field: str) -> str:
    if raw_field.startswith("VEC:"):
        base_f = f"vec_avg({raw_field[4:]})"
    else:
        base_f = raw_field
    return template.format(f=base_f)

STAGE2_NEUTRALIZATIONS: tuple[str, ...] = (
    "SUBINDUSTRY",
    "INDUSTRY",
    "SECTOR",
    "MARKET",
    "CROWDING",
    "STATISTICAL",
)
STAGE2_DECAYS: tuple[int, ...] = (0, 4, 8, 15)
STAGE2_TRUNCATIONS: tuple[float, ...] = (0.05, 0.08, 0.10)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _sim_hash(expression: str, settings: Mapping[str, Any]) -> str:
    from worldquant_brain.canonical_ast import parse_fastexpr

    norm = {
        "expr": parse_fastexpr(expression).canonical_expression,
        "neutralization": str(settings.get("neutralization", "SUBINDUSTRY")).upper(),
        "decay": int(settings.get("decay", 4) or 0),
        "truncation": round(float(settings.get("truncation", 0.08) or 0.08), 4),
        "universe": str(settings.get("universe", "TOP3000")).upper(),
        "delay": int(settings.get("delay", 1) or 1),
        "nanHandling": str(settings.get("nanHandling", "OFF")).upper(),
    }
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode("utf-8")).hexdigest()


def _structural_skeleton(expression: str) -> str:
    """Delegate structural skeleton extraction to Canonical FastExpr AST."""
    from worldquant_brain.canonical_ast import parse_fastexpr

    return parse_fastexpr(expression).normalized_expression


def _token_set(expression: str) -> frozenset[str]:
    return frozenset(t.lower() for t in _TOKEN_RE.findall(str(expression or "")))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _negate_expression(expr: str) -> str:
    s = str(expr).strip()
    if s.startswith("-(") and s.endswith(")"):
        return s[2:-1].strip()
    if s.startswith("-"):
        return s[1:].strip()
    return f"-({s})"


NON_RESCUABLE_CHECKS: tuple[str, ...] = (
    "LOW_UNIVERSE",
    "COVERAGE",
    "DEAD",
    "NO_DATA",
    "SYNTAX",
    "ERROR",
)


def classify_verdict(
    result: Mapping[str, Any],
    *,
    min_sharpe: float = 1.25,
    min_fitness: float = 1.00,
    min_turnover: float = 0.01,
    max_turnover: float = 0.70,
    near_miss_sharpe: float = 0.90,
) -> tuple[str, str]:
    """Multi-factor delay-aware classification into (PASSED | NEAR_MISS | REJECTED, reason)."""
    sharpe = result.get("sharpe")
    fitness = result.get("fitness")
    turnover = result.get("turnover")
    margin = result.get("margin")
    checks_passed = bool(result.get("checks_passed", False))
    failed_checks = [str(c).upper() for c in (result.get("failed_checks") or []) if c]
    sett = result.get("settings") if isinstance(result.get("settings"), Mapping) else {}
    delay = int(sett["delay"]) if sett.get("delay") is not None else 1

    if sharpe is None or fitness is None:
        return "REJECTED", str(result.get("error") or "MISSING_METRICS")

    eff_min_sharpe = 2.00 if delay == 0 else float(min_sharpe)
    eff_nm_sharpe = max(float(near_miss_sharpe), 1.45) if delay == 0 else float(near_miss_sharpe)

    sh = float(sharpe)
    fit = float(fitness)
    to = float(turnover) if turnover is not None else 0.25
    norm_margin = (
        float(margin) / 10000.0
        if (margin is not None and float(margin) > 0.5)
        else (float(margin) if margin is not None else None)
    )

    reasons: list[str] = []
    if sh < eff_min_sharpe:
        reasons.append(f"LOW_SHARPE({sh:.2f}<{eff_min_sharpe:.2f})")
    if fit < min_fitness:
        reasons.append(f"LOW_FITNESS({fit:.2f}<{min_fitness:.2f})")
    if not (min_turnover < to < max_turnover):
        reasons.append(f"TURNOVER_OUT_OF_RANGE({to:.3f})")
    if norm_margin is not None and norm_margin < 0.0010:
        reasons.append(f"LOW_MARGIN({norm_margin:.5f})")
    if not checks_passed:
        reasons.append("FAILED_CHECKS(" + (",".join(failed_checks) if failed_checks else "UNKNOWN") + ")")

    if not reasons:
        return "PASSED", "QUALIFIED"

    # Multi-factor NEAR_MISS eligibility (must be genuinely rescuable)
    if any(bad in chk for chk in failed_checks for bad in NON_RESCUABLE_CHECKS):
        return "REJECTED", ";".join(reasons)
    if len(failed_checks) >= 4:
        return "REJECTED", ";".join(reasons)
    if abs(sh) < eff_nm_sharpe:
        return "REJECTED", ";".join(reasons)
    min_nm_fit = 0.50 if abs(sh) >= eff_min_sharpe else 0.55
    eff_fit = abs(fit) if (sh < 0 and fit < 0) else fit
    if eff_fit < min_nm_fit:
        return "REJECTED", ";".join(reasons)
    if not (0.005 < to < 0.95):
        return "REJECTED", ";".join(reasons)
    if norm_margin is not None and norm_margin < 0.0002 and abs(sh) < eff_min_sharpe:
        return "REJECTED", ";".join(reasons)

    return "NEAR_MISS", ";".join(reasons)


class TwoStageAlphaFunnel:
    """SQLite-backed RL-guided alpha discovery funnel with immediate session refinement."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        seed_sharpe_threshold: float = 0.90,
        base_settings: Mapping[str, Any] | None = None,
        repulsion_jaccard: float = 0.92,
    ) -> None:
        self.db_path = str(db_path)
        self.seed_sharpe_threshold = float(seed_sharpe_threshold)
        self.repulsion_jaccard = float(repulsion_jaccard)
        self.base_settings = dict(DEFAULT_SIM_SETTINGS)
        if base_settings:
            self.base_settings.update(dict(base_settings))
        # In-session only: near-miss seeds are refined immediately in this run, never re-mined from DB
        self._session_refinement_queue: list[dict[str, Any]] = []
        self._session_seen_hashes: set[str] = set()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS funnel_tried (
                        sim_hash TEXT PRIMARY KEY,
                        expression TEXT NOT NULL,
                        settings_json TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        dataset_id TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS funnel_dataset_stats (
                        dataset_id TEXT PRIMARY KEY,
                        screened_count INTEGER NOT NULL DEFAULT 0,
                        seeds_found INTEGER NOT NULL DEFAULT 0,
                        passed_count INTEGER NOT NULL DEFAULT 0,
                        submit_failed_count INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS funnel_reference_db (
                        sim_hash TEXT PRIMARY KEY,
                        expression TEXT NOT NULL,
                        skeleton TEXT NOT NULL DEFAULT '',
                        settings_json TEXT NOT NULL,
                        dataset_id TEXT NOT NULL DEFAULT '',
                        probe TEXT NOT NULL DEFAULT '',
                        stage TEXT NOT NULL DEFAULT 'STAGE1',
                        alpha_id TEXT,
                        outcome TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        sharpe REAL,
                        fitness REAL,
                        turnover REAL,
                        margin REAL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS funnel_rl_policy (
                        arm_key TEXT PRIMARY KEY,
                        dataset_id TEXT NOT NULL,
                        probe TEXT NOT NULL,
                        pulls INTEGER NOT NULL DEFAULT 0,
                        reward_sum REAL NOT NULL DEFAULT 0.0,
                        last_reward REAL NOT NULL DEFAULT 0.0,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                # Backfill skeleton column if upgrading from an earlier schema
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(funnel_reference_db)").fetchall()}
                if "skeleton" not in cols:
                    conn.execute("ALTER TABLE funnel_reference_db ADD COLUMN skeleton TEXT NOT NULL DEFAULT ''")
            finally:
                conn.commit()

    def _is_tried(self, conn: sqlite3.Connection, sim_hash: str) -> bool:
        if sim_hash in self._session_seen_hashes:
            return True
        if conn.execute("SELECT 1 FROM funnel_tried WHERE sim_hash = ?", (sim_hash,)).fetchone() is not None:
            return True
        if conn.execute("SELECT 1 FROM funnel_reference_db WHERE sim_hash = ?", (sim_hash,)).fetchone() is not None:
            return True
        return False

    def _mark_tried(
        self,
        conn: sqlite3.Connection,
        sim_hash: str,
        expression: str,
        settings: Mapping[str, Any],
        stage: str,
        dataset_id: str,
    ) -> None:
        self._session_seen_hashes.add(sim_hash)
        conn.execute(
            """
            INSERT OR IGNORE INTO funnel_tried (sim_hash, expression, settings_json, stage, dataset_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                sim_hash,
                expression,
                json.dumps(dict(settings), sort_keys=True),
                stage,
                dataset_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def _update_rl_arm(self, conn: sqlite3.Connection, dataset_id: str, probe: str, reward: float) -> None:
        if not dataset_id and not probe:
            return
        arm = f"{dataset_id}::{probe or 'DEFAULT'}"
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO funnel_rl_policy (arm_key, dataset_id, probe, pulls, reward_sum, last_reward, updated_at)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(arm_key) DO UPDATE SET
                pulls = pulls + 1,
                reward_sum = reward_sum + excluded.reward_sum,
                last_reward = excluded.last_reward,
                updated_at = excluded.updated_at
            """,
            (arm, dataset_id, probe or "DEFAULT", float(reward), float(reward), now_iso),
        )

    def _ucb_rank_arms(
        self,
        conn: sqlite3.Connection,
        dataset_ids: Sequence[str],
        probes: Sequence[tuple[str, str]],
    ) -> list[tuple[str, str, str]]:
        """Rank (dataset_id, probe_name, template) arms using UCB1 over historical RL rewards."""
        rows = conn.execute("SELECT arm_key, pulls, reward_sum FROM funnel_rl_policy").fetchall()
        arm_map = {str(r["arm_key"]): (int(r["pulls"]), float(r["reward_sum"])) for r in rows}
        total_pulls = max(1, sum(p for p, _ in arm_map.values()))
        scored: list[tuple[float, str, str, str]] = []
        for ds_id in dataset_ids:
            for probe_name, template in probes:
                pulls, r_sum = arm_map.get(f"{ds_id}::{probe_name}", (0, 0.0))
                if pulls == 0:
                    ucb = 10.0  # untried arms get priority exploration
                else:
                    mean_r = r_sum / pulls
                    ucb = mean_r + 1.25 * math.sqrt(math.log(total_pulls + 1.0) / pulls)
                scored.append((ucb, ds_id, probe_name, template))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(ds, pn, tmpl) for _, ds, pn, tmpl in scored]

    def _load_reference_guards(
        self, conn: sqlite3.Connection
    ) -> tuple[set[str], list[frozenset[str]]]:
        """Load already-submitted/passed skeletons (prevent param-only clones) and recent failure token sets (repulsion)."""
        rows = conn.execute(
            """
            SELECT expression, skeleton, outcome
            FROM funnel_reference_db
            ORDER BY updated_at DESC
            LIMIT 400
            """
        ).fetchall()
        submitted_or_passed_skeletons: set[str] = set()
        failure_tokens: list[frozenset[str]] = []
        for r in rows:
            outc = str(r["outcome"] or "")
            skel = str(r["skeleton"] or "") or _structural_skeleton(str(r["expression"] or ""))
            if outc in ("PASSED", "SUBMITTED"):
                if skel:
                    submitted_or_passed_skeletons.add(skel)
            elif outc in ("REJECTED", "SUBMIT_FAILED"):
                toks = _token_set(str(r["expression"] or ""))
                if toks:
                    failure_tokens.append(toks)
        return submitted_or_passed_skeletons, failure_tokens

    def _is_repelled_by_reference(
        self,
        expression: str,
        submitted_skeletons: set[str],
        failure_tokens: list[frozenset[str]],
    ) -> bool:
        # 1. Never generate parameter-tweaked clones of already PASSED/SUBMITTED alphas
        if _structural_skeleton(expression) in submitted_skeletons:
            return True
        # 2. Stay away from expressions structurally too close to recent REJECTED/SUBMIT_FAILED ones
        toks = _token_set(expression)
        for ftoks in failure_tokens:
            if _jaccard(toks, ftoks) >= self.repulsion_jaccard:
                return True
        return False

    def pending_refinement_count(self) -> int:
        return len(self._session_refinement_queue)

    def _enqueue_immediate_refinement(self, result: Mapping[str, Any], reason: str) -> int:
        """Immediately build Stage-2 fine-tuning variants for a current-session NEAR_MISS."""
        expr = str(result.get("expression") or "").strip()
        if not expr:
            return 0
        sharpe = result.get("sharpe")
        if sharpe is not None and float(sharpe) < 0:
            expr = _negate_expression(expr)
        base_sett = dict(self.base_settings)
        if isinstance(result.get("settings"), Mapping):
            base_sett.update(dict(result["settings"]))
        ds_id = str(result.get("dataset_id") or "")
        src_to = result.get("turnover")

        candidates: list[tuple[str, dict[str, Any]]] = []
        # Target 12.5% turnover sweet spot (Fitness floor) and Sub-Universe / Weight robustness via hump()
        if "TURNOVER" in reason or "FITNESS" in reason or (src_to is not None and float(src_to) > 0.15):
            candidates.append((f"hump(winsorize({expr}, std=4), hump=0.01)", dict(base_sett, decay=4)))
            candidates.append((f"ts_decay_linear({expr}, 8)", dict(base_sett, decay=6)))
            candidates.append((f"winsorize(ts_mean({expr}, 5), std=4)", dict(base_sett, decay=6)))
        if "WEIGHT" in reason or "SUB_UNIVERSE" in reason:
            candidates.append((f"hump({expr}, hump=0.01)", dict(base_sett, truncation=0.06)))
            for trunc in (0.05, 0.06):
                candidates.append((f"winsorize({expr}, std=3)", dict(base_sett, truncation=trunc)))
        for neut in STAGE2_NEUTRALIZATIONS:
            candidates.append((expr, dict(base_sett, neutralization=neut)))
        for dec in STAGE2_DECAYS:
            candidates.append((expr, dict(base_sett, decay=dec)))
        for trunc in STAGE2_TRUNCATIONS:
            candidates.append((expr, dict(base_sett, truncation=trunc)))
        if str(base_sett.get("nanHandling", "OFF")).upper() != "ON":
            candidates.append((expr, dict(base_sett, nanHandling="ON")))

        added = 0
        with self._connect() as conn:
            for c_expr, c_sett in candidates:
                sh = _sim_hash(c_expr, c_sett)
                if self._is_tried(conn, sh):
                    continue
                self._mark_tried(conn, sh, c_expr, c_sett, "STAGE2", ds_id)
                self._session_refinement_queue.append(
                    {
                        "expression": c_expr,
                        "settings": c_sett,
                        "stage": "STAGE2",
                        "dataset_id": ds_id,
                        "probe": str(result.get("probe") or ""),
                    }
                )
                added += 1
        return added

    def next_stage2_batch(self, batch_size: int = 8) -> list[dict[str, Any]]:
        """Pop immediate in-session near-miss refinement variants (never mines old DB seeds)."""
        if not self._session_refinement_queue:
            return []
        batch = self._session_refinement_queue[:batch_size]
        del self._session_refinement_queue[:batch_size]
        return batch

    def next_stage1_batch(
        self,
        dataset_id: str,
        fields: Sequence[str],
        *,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Generate a Stage-1 batch for a specific dataset guided by RL probe rankings and reference repulsion."""
        return self.next_batch({dataset_id: fields}, batch_size=batch_size)

    def next_batch(
        self,
        datasets_fields: Mapping[str, Sequence[str]],
        *,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Return immediate session near-miss refinements first; otherwise RL-guided Stage-1 batch."""
        s2 = self.next_stage2_batch(batch_size=batch_size)
        if s2:
            return s2

        out: list[dict[str, Any]] = []
        with self._connect() as conn:
            arms = self._ucb_rank_arms(conn, list(datasets_fields.keys()), STAGE1_PROBES)
            submitted_skels, failure_toks = self._load_reference_guards(conn)
            for ds_id, probe_name, template in arms:
                for field in datasets_fields.get(ds_id, ()):
                    if len(out) >= batch_size:
                        return out
                    expr = _format_probe(template, field)
                    ok_expr, _ = validate_candidate_expression(expr)
                    if not ok_expr:
                        continue
                    sett = dict(self.base_settings)
                    sh = _sim_hash(expr, sett)
                    if self._is_tried(conn, sh):
                        continue
                    if self._is_repelled_by_reference(expr, submitted_skels, failure_toks):
                        self._mark_tried(conn, sh, expr, sett, "REPELLED", ds_id)
                        continue
                    self._mark_tried(conn, sh, expr, sett, "STAGE1", ds_id)
                    out.append(
                        {
                            "expression": expr,
                            "settings": sett,
                            "stage": "STAGE1",
                            "dataset_id": ds_id,
                            "probe": probe_name,
                            "field": field,
                        }
                    )
        return out

    def mark_submitted(
        self,
        *,
        alpha_id: str,
        expression: str = "",
        settings: Mapping[str, Any] | None = None,
        dataset_id: str = "",
        probe: str = "",
    ) -> None:
        """Record successful official submission in reference DB and boost RL reward."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            if expression:
                sett = dict(self.base_settings)
                if settings:
                    sett.update(dict(settings))
                sh = _sim_hash(expression, sett)
                conn.execute(
                    """
                    INSERT INTO funnel_reference_db (
                        sim_hash, expression, skeleton, settings_json, dataset_id, probe,
                        alpha_id, outcome, reason, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'SUBMITTED', 'OFFICIAL_SUBMITTED', ?)
                    ON CONFLICT(sim_hash) DO UPDATE SET
                        outcome = 'SUBMITTED',
                        reason = 'OFFICIAL_SUBMITTED',
                        updated_at = excluded.updated_at
                    """,
                    (sh, expression, _structural_skeleton(expression), json.dumps(sett, sort_keys=True), dataset_id, probe, str(alpha_id), now_iso),
                )
            else:
                conn.execute(
                    "UPDATE funnel_reference_db SET outcome = 'SUBMITTED', reason = 'OFFICIAL_SUBMITTED', updated_at = ? WHERE alpha_id = ?",
                    (now_iso, str(alpha_id)),
                )
            self._update_rl_arm(conn, dataset_id, probe, +3.0)

    def mark_submit_failed(
        self,
        *,
        alpha_id: str,
        reason: str,
        expression: str = "",
        settings: Mapping[str, Any] | None = None,
        dataset_id: str = "",
        probe: str = "",
    ) -> None:
        """Record submission failure with explicit reason in reference DB and penalize RL arm."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            if expression:
                sett = dict(self.base_settings)
                if settings:
                    sett.update(dict(settings))
                sh = _sim_hash(expression, sett)
                conn.execute(
                    """
                    INSERT INTO funnel_reference_db (
                        sim_hash, expression, skeleton, settings_json, dataset_id, probe,
                        alpha_id, outcome, reason, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'SUBMIT_FAILED', ?, ?)
                    ON CONFLICT(sim_hash) DO UPDATE SET
                        outcome = 'SUBMIT_FAILED',
                        reason = excluded.reason,
                        updated_at = excluded.updated_at
                    """,
                    (sh, expression, _structural_skeleton(expression), json.dumps(sett, sort_keys=True), dataset_id, probe, str(alpha_id), reason, now_iso),
                )
            else:
                row = conn.execute("SELECT dataset_id, probe FROM funnel_reference_db WHERE alpha_id = ?", (str(alpha_id),)).fetchone()
                if row:
                    dataset_id = dataset_id or str(row["dataset_id"] or "")
                    probe = probe or str(row["probe"] or "")
                conn.execute(
                    "UPDATE funnel_reference_db SET outcome = 'SUBMIT_FAILED', reason = ?, updated_at = ? WHERE alpha_id = ?",
                    (reason, now_iso, str(alpha_id)),
                )
            if dataset_id:
                conn.execute(
                    """
                    INSERT INTO funnel_dataset_stats (dataset_id, screened_count, seeds_found, passed_count, submit_failed_count, updated_at)
                    VALUES (?, 0, 0, 0, 1, ?)
                    ON CONFLICT(dataset_id) DO UPDATE SET
                        submit_failed_count = submit_failed_count + 1,
                        updated_at = excluded.updated_at
                    """,
                    (dataset_id, now_iso),
                )
            self._update_rl_arm(conn, dataset_id, probe, -1.0)

    def ingest_results(self, results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        """Record all outcomes in reference DB, update RL policy, and queue immediate session refinements."""
        promoted = 0
        passed = 0
        rejected = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        near_misses_to_queue: list[tuple[Mapping[str, Any], str]] = []

        with self._connect() as conn:
            for r in results:
                ds_id = str(r.get("dataset_id") or "")
                stage = str(r.get("stage") or "STAGE1")
                probe = str(r.get("probe") or "")
                expr = str(r.get("expression") or "").strip()
                sett = dict(r.get("settings") or self.base_settings)
                sharpe = r.get("sharpe")
                fitness = r.get("fitness")
                turnover = r.get("turnover")
                margin = r.get("margin")
                alpha_id = r.get("alpha_id")

                if ds_id:
                    conn.execute(
                        """
                        INSERT INTO funnel_dataset_stats (dataset_id, screened_count, seeds_found, passed_count, submit_failed_count, updated_at)
                        VALUES (?, 1, 0, 0, 0, ?)
                        ON CONFLICT(dataset_id) DO UPDATE SET
                            screened_count = screened_count + 1,
                            updated_at = excluded.updated_at
                        """,
                        (ds_id, now_iso),
                    )

                verdict, reason = classify_verdict(r, near_miss_sharpe=self.seed_sharpe_threshold)
                sh_hash = _sim_hash(expr, sett) if expr else hashlib.sha256(str(alpha_id or now_iso).encode()).hexdigest()

                conn.execute(
                    """
                    INSERT INTO funnel_reference_db (
                        sim_hash, expression, skeleton, settings_json, dataset_id, probe, stage,
                        alpha_id, outcome, reason, sharpe, fitness, turnover, margin, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sim_hash) DO UPDATE SET
                        alpha_id = COALESCE(excluded.alpha_id, funnel_reference_db.alpha_id),
                        outcome = excluded.outcome,
                        reason = excluded.reason,
                        sharpe = COALESCE(excluded.sharpe, funnel_reference_db.sharpe),
                        fitness = COALESCE(excluded.fitness, funnel_reference_db.fitness),
                        turnover = COALESCE(excluded.turnover, funnel_reference_db.turnover),
                        margin = COALESCE(excluded.margin, funnel_reference_db.margin),
                        updated_at = excluded.updated_at
                    """,
                    (
                        sh_hash,
                        expr,
                        _structural_skeleton(expr) if expr else "",
                        json.dumps(sett, sort_keys=True),
                        ds_id,
                        probe,
                        stage,
                        alpha_id,
                        verdict,
                        reason,
                        float(sharpe) if sharpe is not None else None,
                        float(fitness) if fitness is not None else None,
                        float(turnover) if turnover is not None else None,
                        float(margin) if margin is not None else None,
                        now_iso,
                    ),
                )

                if verdict == "PASSED":
                    passed += 1
                    self._update_rl_arm(conn, ds_id, probe, +2.0)
                    if ds_id:
                        conn.execute(
                            "UPDATE funnel_dataset_stats SET passed_count = passed_count + 1, updated_at = ? WHERE dataset_id = ?",
                            (now_iso, ds_id),
                        )
                elif verdict == "NEAR_MISS":
                    self._update_rl_arm(conn, ds_id, probe, +0.8)
                    if stage == "STAGE1":
                        promoted += 1
                        near_misses_to_queue.append((r, reason))
                        if ds_id:
                            conn.execute(
                                "UPDATE funnel_dataset_stats SET seeds_found = seeds_found + 1, updated_at = ? WHERE dataset_id = ?",
                                (now_iso, ds_id),
                            )
                else:
                    rejected += 1
                    self._update_rl_arm(conn, ds_id, probe, -0.4)

        for nm_res, nm_reason in near_misses_to_queue:
            self._enqueue_immediate_refinement(nm_res, nm_reason)

        return {"promoted": promoted, "passed": passed, "rejected": rejected}

    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            tried = int(conn.execute("SELECT COUNT(*) FROM funnel_tried").fetchone()[0])
            ref_rows = conn.execute(
                "SELECT outcome, COUNT(*) AS cnt FROM funnel_reference_db GROUP BY outcome"
            ).fetchall()
            outcomes = {str(r["outcome"]): int(r["cnt"]) for r in ref_rows}
            rl_rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT arm_key, pulls, ROUND(reward_sum, 2) AS reward_sum FROM funnel_rl_policy ORDER BY reward_sum DESC LIMIT 10"
                ).fetchall()
            ]
            ds_rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM funnel_dataset_stats ORDER BY passed_count DESC, seeds_found DESC"
                ).fetchall()
            ]
        return {
            "tried_total": tried,
            "session_refinement_queue": len(self._session_refinement_queue),
            "reference_outcomes": outcomes,
            "top_rl_arms": rl_rows,
            "datasets": ds_rows,
        }
