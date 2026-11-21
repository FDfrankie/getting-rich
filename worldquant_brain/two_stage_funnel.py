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
import os
import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from worldquant_brain.multi_sim_client import DEFAULT_SIM_SETTINGS

STAGE1_PROBES: tuple[tuple[str, str], ...] = (
    ("rank_bf", "rank(ts_backfill({f}, 20))"),
    ("zscore60", "winsorize(ts_zscore(ts_backfill({f}, 20), 60), std=4)"),
    ("grp_zscore", "winsorize(group_zscore(ts_backfill({f}, 20), industry), std=4)"),
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
    ("reg_neut_cap", "regression_neut(rank(ts_backfill({f}, 20)), log(cap + 1.0))"),
    ("trade_when_vol", "trade_when(volume > ts_mean(volume, 20), group_neutralize(winsorize(rank(ts_backfill({f}, 20)), std=4), industry), -1)"),
    ("cartesian_cap_sub", "group_neutralize(rank(ts_backfill({f}, 20)), group_cartesian_product(subindustry, bucket(rank(cap), range=\"0, 1, 0.25\")))"),
    ("jump_decay_neut", "group_neutralize(jump_decay(rank(ts_backfill({f}, 20)), 5, sensitivity=0.5), subindustry)"),
    ("target_tvr_decay", "ts_target_tvr_decay(group_neutralize(rank(ts_backfill({f}, 20)), industry), lambda_min=0, lambda_max=1, target_tvr=0.15)"),
)


def _format_probe(template: str, raw_field: str) -> str:
    if raw_field.startswith("VEC:"):
        base_f = f"vec_avg({raw_field[4:]})"
    else:
        base_f = raw_field
    return template.format(f=base_f)

_DEFAULT_STAGE2_NEUTRALIZATIONS: tuple[str, ...] = (
    "SUBINDUSTRY",
    "INDUSTRY",
    "SECTOR",
    "SLOW",
    "FAST",
    "MARKET",
)
_env_neuts = [
    tok.strip().upper()
    for tok in os.getenv("BRAIN_STAGE2_NEUTRALIZATIONS", "").split(",")
    if tok.strip()
]
STAGE2_NEUTRALIZATIONS: tuple[str, ...] = tuple(_env_neuts) if _env_neuts else _DEFAULT_STAGE2_NEUTRALIZATIONS
_UNAVAILABLE_NEUT_RE = re.compile(
    r"Neutralization\s+([A-Z0-9_]+)\s+is\s+not\s+available",
    re.IGNORECASE,
)
STAGE2_DECAYS: tuple[int, ...] = (0, 4, 8, 15)
STAGE2_TRUNCATIONS: tuple[float, ...] = (0.05, 0.08, 0.10)


_LOG_WINDOWS: tuple[int, ...] = (5, 20, 60, 120, 252)
_TS_WIN_RE = re.compile(r"\b(ts_(?:zscore|rank|delta|mean|std_dev|av_diff|decay_linear|corr|regression)\s*\([^()]*?,\s*)(\d+)(\s*(?:,|\)))")
_FIELD_IN_BF_RE = re.compile(r"ts_backfill\(\s*(?:vec_avg\()?([A-Za-z_][A-Za-z0-9_]*)")


def _field_family_prefix(raw_field: str) -> str:
    """Group related fields (e.g., news18_sent_7d / news18_sent_30d) by prefix for early-stopping."""
    f = raw_field[4:] if raw_field.startswith("VEC:") else raw_field
    parts = [tok for tok in f.lower().split("_") if tok]
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0] if parts else ""


def _extract_primary_field(expression: str) -> str:
    m = _FIELD_IN_BF_RE.search(expression or "")
    return m.group(1) if m else ""


_TS_OP_HEAD_RE = re.compile(r"\b(ts_(?:zscore|rank|delta|mean|std_dev|av_diff|decay_linear|corr|regression))\s*\(")


def _log_window_variants(expression: str, *, prefer_larger: bool = False) -> list[str]:
    """Generate integer-only logarithmic window jump variants for nested ts_* operators."""
    expr = expression or ""
    m = _TS_OP_HEAD_RE.search(expr)
    if not m:
        return []
    # Find matching closing paren for this ts_* call
    depth = 1
    idx = m.end()
    last_comma = -1
    while idx < len(expr) and depth > 0:
        ch = expr[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        elif ch == "," and depth == 1:
            last_comma = idx
        idx += 1
    if depth != 0 or last_comma < 0:
        return []
    # Scan top-level comma-separated arguments after first comma for an integer literal window
    args_span = expr[m.end() : idx]
    win_start = win_end = cur_w = -1
    cur_depth = 0
    seg_start = 0
    for pos, ch in enumerate(args_span + ","):
        if ch == "(":
            cur_depth += 1
        elif ch == ")":
            cur_depth -= 1
        elif ch == "," and cur_depth == 0:
            if seg_start > 0:
                raw_arg = args_span[seg_start:pos]
                stripped = raw_arg.strip()
                if stripped.isdigit():
                    lead = len(raw_arg) - len(raw_arg.lstrip())
                    win_start = m.end() + seg_start + lead
                    win_end = win_start + len(stripped)
                    cur_w = int(stripped)
                    break
            seg_start = pos + 1
    if cur_w < 0:
        return []
    targets = [w for w in _LOG_WINDOWS if w != cur_w and (not prefer_larger or w > cur_w)]
    out: list[str] = []
    for w in targets[:3]:
        replaced = expr[:win_start] + str(w) + expr[win_end:]
        if replaced != expr:
            out.append(replaced)
    return out

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
    near_miss_sharpe: float = 1.15,
    near_miss_min_fitness: float | None = None,
    near_miss_min_turnover: float | None = None,
    near_miss_max_turnover: float | None = None,
    near_miss_min_margin: float | None = None,
    near_miss_max_failed_checks: int | None = None,
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

    if sharpe is None or fitness is None or turnover is None:
        return "REJECTED", str(result.get("error") or "MISSING_METRICS")

    eff_min_sharpe = 2.00 if delay == 0 else float(min_sharpe)
    eff_nm_sharpe = max(float(near_miss_sharpe), 1.80) if delay == 0 else float(near_miss_sharpe)

    sh = float(sharpe)
    fit = float(fitness)
    raw_to = float(turnover)
    to = raw_to / 100.0 if 2.0 < raw_to <= 100.0 else raw_to
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
    if not (min_turnover <= to <= max_turnover):
        reasons.append(f"TURNOVER_OUT_OF_RANGE({to:.3f})")
    if norm_margin is not None and norm_margin < 0.0010:
        reasons.append(f"LOW_MARGIN({norm_margin:.5f})")
    if not checks_passed:
        reasons.append("FAILED_CHECKS(" + (",".join(failed_checks) if failed_checks else "UNKNOWN") + ")")

    if not reasons:
        return "PASSED", "QUALIFIED"

    # Multi-factor NEAR_MISS eligibility (P14 tight gate: only genuinely rescuable seeds enter Stage 2)
    max_fc = int(
        os.getenv("BRAIN_NEAR_MISS_MAX_FAILED_CHECKS", "2") or "2"
        if near_miss_max_failed_checks is None
        else near_miss_max_failed_checks
    )
    if any(bad in chk for chk in failed_checks for bad in NON_RESCUABLE_CHECKS):
        return "REJECTED", ";".join(reasons)
    if len(failed_checks) > max_fc:
        return "REJECTED", ";".join(reasons)
    # P22 turnover-aware rescue: high-turnover raw signals (0.25 < to <= 0.78) have strong decay leverage
    # (decay reduces turnover and lifts Fitness by ~1.3x-1.6x), so allow slightly lower Stage-1 Sharpe/Fitness.
    high_to_leverage = (to > 0.25) and (delay != 0) and (near_miss_min_fitness is None)
    adaptive_nm_sharpe = min(eff_nm_sharpe, 1.05) if high_to_leverage else eff_nm_sharpe
    if abs(sh) < adaptive_nm_sharpe:
        return "REJECTED", ";".join(reasons)
    # When Sharpe is strongly negative (e.g. -1.48), BRAIN Fitness and Margin are also negative;
    # evaluate their sign-flipped magnitude so Stage-2 negation (-1 * expr) can rescue them.
    eff_fit = -fit if sh < 0 else fit
    eff_margin = (-norm_margin if (norm_margin is not None and sh < 0) else norm_margin)
    base_nm_fit = float(
        os.getenv("BRAIN_NEAR_MISS_MIN_FITNESS", "0.88") or "0.88"
        if near_miss_min_fitness is None
        else near_miss_min_fitness
    )
    if high_to_leverage:
        min_nm_fit = min(base_nm_fit, 0.72)
    else:
        min_nm_fit = max(0.82, base_nm_fit - 0.06) if abs(sh) >= eff_min_sharpe else base_nm_fit
    if eff_fit < min_nm_fit:
        return "REJECTED", ";".join(reasons)
    nm_min_to = float(
        os.getenv("BRAIN_NEAR_MISS_MIN_TO", "0.008") or "0.008"
        if near_miss_min_turnover is None
        else near_miss_min_turnover
    )
    nm_max_to = float(
        os.getenv("BRAIN_NEAR_MISS_MAX_TO", "0.78") or "0.78"
        if near_miss_max_turnover is None
        else near_miss_max_turnover
    )
    if not (nm_min_to <= to <= nm_max_to):
        return "REJECTED", ";".join(reasons)
    base_nm_margin = float(
        os.getenv("BRAIN_NEAR_MISS_MIN_MARGIN", "0.0007") or "0.0007"
        if near_miss_min_margin is None
        else near_miss_min_margin
    )
    eff_nm_margin = min(base_nm_margin, 0.0005) if abs(sh) >= eff_min_sharpe else base_nm_margin
    if eff_margin is not None and eff_margin < eff_nm_margin:
        return "REJECTED", ";".join(reasons)

    return "NEAR_MISS", ";".join(reasons)


class TwoStageAlphaFunnel:
    """SQLite-backed RL-guided alpha discovery funnel with immediate session refinement."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        seed_sharpe_threshold: float | None = None,
        base_settings: Mapping[str, Any] | None = None,
        repulsion_jaccard: float = 0.92,
        max_stage2_variants_per_seed: int | None = None,
        stage1_to_stage2_ratio: int | None = None,
    ) -> None:
        self.db_path = str(db_path)
        default_thresh = float(os.getenv("BRAIN_SEED_SHARPE_THRESHOLD", "1.15") or "1.15")
        self.seed_sharpe_threshold = float(default_thresh if seed_sharpe_threshold is None else seed_sharpe_threshold)
        self.max_stage2_variants_per_seed = max(
            1,
            int(
                os.getenv("BRAIN_MAX_STAGE2_VARIANTS", "4") or "4"
                if max_stage2_variants_per_seed is None
                else max_stage2_variants_per_seed
            ),
        )
        self.stage1_to_stage2_ratio = max(
            0,
            int(
                os.getenv("BRAIN_STAGE1_PER_STAGE2", "3") or "3"
                if stage1_to_stage2_ratio is None
                else stage1_to_stage2_ratio
            ),
        )
        self._stage1_batches_since_stage2: int = 0
        self._weak_prefix_counts: dict[tuple[str, str], int] = {}
        self._prefix_early_stop_limit: int = max(2, int(os.getenv('BRAIN_PREFIX_EARLY_STOP', '3') or '3'))
        self._strong_fields: dict[str, dict[str, float]] = {}
        self._paired_fields: set[tuple[str, str, str]] = set()
        self.repulsion_jaccard = float(repulsion_jaccard)
        self.base_settings = dict(DEFAULT_SIM_SETTINGS)
        if base_settings:
            self.base_settings.update(dict(base_settings))
        # In-session only: near-miss seeds are refined immediately in this run, never re-mined from DB
        self._session_refinement_queue: list[dict[str, Any]] = []
        self._unavailable_neutralizations: set[str] = set()
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
                stat_cols = {r["name"] for r in conn.execute("PRAGMA table_info(funnel_dataset_stats)").fetchall()}
                if "submit_failed_count" not in stat_cols:
                    conn.execute("ALTER TABLE funnel_dataset_stats ADD COLUMN submit_failed_count INTEGER NOT NULL DEFAULT 0")
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
        is_neg_sharpe = sharpe is not None and float(sharpe) < 0
        if is_neg_sharpe:
            expr = _negate_expression(expr)
        base_sett = dict(self.base_settings)
        if isinstance(result.get("settings"), Mapping):
            base_sett.update(dict(result["settings"]))
        ds_id = str(result.get("dataset_id") or "")
        src_to = result.get("turnover")

        candidates: list[tuple[str, dict[str, Any]]] = []
        # P22: Always try pure sign-flipped -(expr) with original settings FIRST before any mutation
        if is_neg_sharpe:
            candidates.append((expr, dict(base_sett)))
        src_margin = result.get("margin")
        norm_margin = (
            float(src_margin) / 10000.0
            if (src_margin is not None and float(src_margin) > 0.5)
            else (float(src_margin) if src_margin is not None else None)
        )
        # Diagnostic tuning: target the specific bottleneck instead of blind grid search
        if "TURNOVER" in reason or ("FITNESS" in reason and (src_to is None or float(src_to) > 0.25)) or (src_to is not None and float(src_to) > 0.40):
            for lw_expr in _log_window_variants(expr, prefer_larger=True):
                candidates.append((lw_expr, dict(base_sett)))
            candidates.append((f"hump(winsorize({expr}, std=4), hump=0.01)", dict(base_sett, decay=4)))
            candidates.append((f"ts_decay_linear({expr}, 8)", dict(base_sett, decay=6)))
        if "MARGIN" in reason or (norm_margin is not None and norm_margin < 0.0010):
            candidates.append((f"signed_power(rank({expr}) - 0.5, 2)", dict(base_sett, decay=0)))
        if "WEIGHT" in reason or "SUB_UNIVERSE" in reason:
            # Official BRAIN Weight/Coverage FAQ: rank/group_rank balances long-short counts + lower truncation
            candidates.append((f"group_rank({expr}, subindustry)", dict(base_sett, truncation=0.05)))
            candidates.append((f"winsorize({expr}, std=3)", dict(base_sett, truncation=0.05)))
            candidates.append((f"hump({expr}, hump=0.01)", dict(base_sett, truncation=0.06)))
        if src_to is not None and float(src_to) < 0.08 and ("FITNESS" in reason or "SHARPE" in reason or "TURNOVER" in reason):
            # Official BRAIN Returns/Turnover FAQ: boost stale low-turnover signals via lower decay, liquid TOP1000, shorter window
            candidates.append((expr, dict(base_sett, decay=0, universe="TOP1000")))
            for sw_expr in _log_window_variants(expr, prefer_larger=False):
                candidates.append((sw_expr, dict(base_sett, decay=0)))
        if "2Y_SHARPE" in reason or "LOW_2Y" in reason:
            candidates.append((expr, dict(base_sett, decay=2)))
            for sw_expr in _log_window_variants(expr, prefer_larger=False):
                candidates.append((sw_expr, dict(base_sett, decay=2)))
        base_neut = str(base_sett.get("neutralization") or "").upper()
        base_decay = int(base_sett.get("decay", 4) or 0)
        base_trunc = round(float(base_sett.get("truncation", 0.08) or 0.08), 4)
        low_turnover = (src_to is not None and float(src_to) < 0.08 and "TURNOVER" not in reason)
        weight_issue = ("WEIGHT" in reason or "SUB_UNIVERSE" in reason)

        for neut in STAGE2_NEUTRALIZATIONS:
            if neut == base_neut or neut in self._unavailable_neutralizations:
                continue
            candidates.append((expr, dict(base_sett, neutralization=neut)))
        for dec in STAGE2_DECAYS:
            if dec == base_decay:
                continue
            if low_turnover and dec > base_decay:
                continue
            candidates.append((expr, dict(base_sett, decay=dec)))
        for trunc in STAGE2_TRUNCATIONS:
            if round(float(trunc), 4) == base_trunc:
                continue
            if not weight_issue and trunc < 0.06:
                continue
            candidates.append((expr, dict(base_sett, truncation=trunc)))

        priority_sharpe = abs(float(sharpe)) if sharpe is not None else 0.0
        added = 0
        with self._connect() as conn:
            for c_expr, c_sett in candidates:
                if added >= self.max_stage2_variants_per_seed:
                    break
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
                        "_priority_sharpe": priority_sharpe,
                    }
                )
                added += 1
        if self._session_refinement_queue:
            self._session_refinement_queue.sort(
                key=lambda item: float(item.get("_priority_sharpe", 0.0)),
                reverse=True,
            )
            if len(self._session_refinement_queue) > 60:
                del self._session_refinement_queue[60:]
        return added

    def _maybe_enqueue_pairwise_synergy(self, ds_id: str, new_field: str, new_abs_sharpe: float) -> int:
        """Pair newly discovered strong field with the strongest distinct peer field in the same dataset."""
        ds_strong = self._strong_fields.get(ds_id) or {}
        peers = sorted(
            ((sc, f) for f, sc in ds_strong.items() if f != new_field),
            reverse=True,
        )
        if not peers:
            return 0
        peer_sh, peer_field = peers[0]
        pair_key = (ds_id, min(new_field, peer_field), max(new_field, peer_field))
        if pair_key in self._paired_fields:
            return 0
        self._paired_fields.add(pair_key)

        synergy_exprs = [
            f"rank(ts_backfill({new_field}, 20)) - rank(ts_backfill({peer_field}, 20))",
            f"rank(ts_regression(ts_backfill({new_field}, 20), ts_backfill({peer_field}, 20), 60, lag=0, rettype=0))",
            f"winsorize(rank(divide(ts_backfill({new_field}, 20), ts_std_dev(ts_backfill({peer_field}, 20), 20) + 1e-6)), std=4)",
        ]
        prio = max(new_abs_sharpe, peer_sh) + 0.05
        added = 0
        with self._connect() as conn:
            for expr in synergy_exprs:
                sett = dict(self.base_settings)
                sh = _sim_hash(expr, sett)
                if self._is_tried(conn, sh):
                    continue
                self._mark_tried(conn, sh, expr, sett, "STAGE2", ds_id)
                self._session_refinement_queue.append(
                    {
                        "expression": expr,
                        "settings": sett,
                        "stage": "STAGE2",
                        "dataset_id": ds_id,
                        "probe": "pairwise_synergy",
                        "_priority_sharpe": prio,
                    }
                )
                added += 1
        if added and self._session_refinement_queue:
            self._session_refinement_queue.sort(
                key=lambda item: float(item.get("_priority_sharpe", 0.0)),
                reverse=True,
            )
            if len(self._session_refinement_queue) > 60:
                del self._session_refinement_queue[60:]
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
        """Interleave Stage-1 exploration and priority-sorted Stage-2 refinement so Stage-2 never starves discovery."""
        if (
            self._session_refinement_queue
            and self._stage1_batches_since_stage2 >= self.stage1_to_stage2_ratio
        ):
            s2 = self.next_stage2_batch(batch_size=batch_size)
            if s2:
                self._stage1_batches_since_stage2 = 0
                return s2

        out: list[dict[str, Any]] = []
        with self._connect() as conn:
            arms = self._ucb_rank_arms(conn, list(datasets_fields.keys()), STAGE1_PROBES)
            submitted_skels, failure_toks = self._load_reference_guards(conn)
            for ds_id, probe_name, template in arms:
                for field in datasets_fields.get(ds_id, ()):
                    if len(out) >= batch_size:
                        self._stage1_batches_since_stage2 += 1
                        return out
                    prefix = _field_family_prefix(field)
                    if prefix and self._weak_prefix_counts.get((ds_id, prefix), 0) >= self._prefix_early_stop_limit:
                        continue
                    ds_low = str(ds_id or "").lower()
                    if probe_name == "cap_norm_gauss" and not any(
                        k in ds_low for k in ("fundamental", "fnd", "balance", "income", "cashflow")
                    ):
                        continue
                    if "fundamental" in ds_low or ds_low.startswith("fnd"):
                        eff_template = template.replace("ts_backfill({f}, 20)", "ts_backfill({f}, 60)")
                    elif ds_low.startswith("pv") or "option" in ds_low or ds_low.startswith("opt"):
                        eff_template = template.replace("ts_backfill({f}, 20)", "ts_backfill({f}, 2)")
                    elif any(k in ds_low for k in ("news", "nws", "social", "analyst", "anl", "model", "mdl", "sentiment")):
                        eff_template = template.replace("ts_backfill({f}, 20)", "ts_backfill({f}, 5)")
                    else:
                        eff_template = template
                    expr = _format_probe(eff_template, field)
                    sett = dict(self.base_settings)
                    if any(
                        k in ds_low
                        for k in ("fundamental", "fnd", "analyst", "anl", "news", "nws", "social", "option", "opt", "model", "mdl", "sentiment")
                    ):
                        sett["nanHandling"] = "ON"
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
        if out:
            self._stage1_batches_since_stage2 += 1
            return out
        if self._session_refinement_queue:
            self._stage1_batches_since_stage2 = 0
            return self.next_stage2_batch(batch_size=batch_size)
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
            self._update_rl_arm(conn, dataset_id, probe, -0.3 if "CORR" in str(reason or "").upper() else -1.0)

        # P23: When a good Alpha is rejected specifically for SELF/PROD correlation, enqueue 4-pillar de-correlation rescue
        if "CORR" in str(reason or "").upper():
            res_expr = expression
            res_sett = dict(self.base_settings)
            if settings:
                res_sett.update(dict(settings))
            if not res_expr:
                with self._connect() as conn2:
                    r_row = conn2.execute(
                        "SELECT expression, settings_json, dataset_id, probe, sharpe FROM funnel_reference_db WHERE alpha_id = ?",
                        (str(alpha_id),),
                    ).fetchone()
                    if r_row:
                        res_expr = str(r_row["expression"] or "")
                        dataset_id = dataset_id or str(r_row["dataset_id"] or "")
                        probe = probe or str(r_row["probe"] or "")
                        try:
                            res_sett.update(json.loads(str(r_row["settings_json"] or "{}")))
                        except Exception:
                            pass
            if res_expr:
                self.enqueue_decorrelation_rescue(
                    expression=res_expr,
                    settings=res_sett,
                    dataset_id=dataset_id,
                    probe=probe,
                    priority_sharpe=1.50,
                )

    def enqueue_decorrelation_rescue(
        self,
        *,
        expression: str,
        settings: Mapping[str, Any] | None = None,
        dataset_id: str = "",
        probe: str = "",
        priority_sharpe: float = 1.50,
        max_variants: int = 3,
    ) -> int:
        """Enqueue WorldQuant BRAIN 4-pillar de-correlation variants for a good Alpha blocked by correlation."""
        from worldquant_brain.alpha_yield_booster import generate_decorrelation_variants

        base_sett = dict(self.base_settings)
        if settings:
            base_sett.update(dict(settings))
        variants = generate_decorrelation_variants(
            expression,
            base_sett,
            max_variants=max_variants,
            unavailable_neutralizations=self._unavailable_neutralizations,
        )
        added = 0
        with self._connect() as conn:
            for v in variants:
                c_expr = str(v.get("expression") or "").strip()
                c_sett = dict(v.get("settings") or base_sett)
                if not c_expr:
                    continue
                sh = _sim_hash(c_expr, c_sett)
                if self._is_tried(conn, sh):
                    continue
                self._mark_tried(conn, sh, c_expr, c_sett, "STAGE2", dataset_id)
                self._session_refinement_queue.append(
                    {
                        "expression": c_expr,
                        "settings": c_sett,
                        "stage": "STAGE2",
                        "dataset_id": dataset_id,
                        "probe": probe or str(v.get("reason") or "decorr_rescue"),
                        "_priority_sharpe": float(priority_sharpe),
                    }
                )
                added += 1
        if added and self._session_refinement_queue:
            self._session_refinement_queue.sort(
                key=lambda item: float(item.get("_priority_sharpe", 0.0)),
                reverse=True,
            )
            if len(self._session_refinement_queue) > 60:
                del self._session_refinement_queue[60:]
        return added

    def ingest_results(self, results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        """Record all outcomes in reference DB, update RL policy, and queue immediate session refinements."""
        promoted = 0
        passed = 0
        rejected = 0
        unsimulated_released = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        near_misses_to_queue: list[tuple[Mapping[str, Any], str]] = []

        _PERMANENT_SYNTAX_TOKENS = (
            "SYNTAX", "INVALID FIELD", "UNKNOWN FIELD", "UNEXPECTED TOKEN",
            "PARSE ERROR", "NOT AVAILABLE FOR", "OPERATOR", "TYPE ERROR",
        )

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
                sh_hash = _sim_hash(expr, sett) if expr else hashlib.sha256(str(alpha_id or now_iso).encode()).hexdigest()

                # P18: Do NOT burn funnel_tried or penalize RL when simulation never completed (API/auth/429/timeout error)
                if not r.get("success") and sharpe is None:
                    err_up = str(r.get("error") or r.get("status") or "").upper()
                    for m_neut in _UNAVAILABLE_NEUT_RE.finditer(err_up):
                        self._unavailable_neutralizations.add(m_neut.group(1).upper())
                    is_syntax_err = any(tok in err_up for tok in _PERMANENT_SYNTAX_TOKENS)
                    if not is_syntax_err:
                        conn.execute("DELETE FROM funnel_tried WHERE sim_hash = ?", (sh_hash,))
                        self._session_seen_hashes.discard(sh_hash)
                        unsimulated_released += 1
                        continue

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
                err_blob = f"{r.get('error') or ''} {reason or ''}"
                for m_neut in _UNAVAILABLE_NEUT_RE.finditer(err_blob):
                    self._unavailable_neutralizations.add(m_neut.group(1).upper())

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

        # Update prefix early-stop stats & collect strong fields for pairwise synergy
        for r in results:
            if str(r.get("stage") or "STAGE1") != "STAGE1":
                continue
            ds_id = str(r.get("dataset_id") or "")
            fld = _extract_primary_field(str(r.get("expression") or ""))
            sh_val = r.get("sharpe")
            if not ds_id or not fld or sh_val is None:
                continue
            abs_sh = abs(float(sh_val))
            pref = _field_family_prefix(fld)
            if pref:
                if abs_sh < 0.30:
                    self._weak_prefix_counts[(ds_id, pref)] = self._weak_prefix_counts.get((ds_id, pref), 0) + 1
                elif abs_sh >= 0.70:
                    self._weak_prefix_counts[(ds_id, pref)] = 0
            if abs_sh >= 0.85:
                ds_strong = self._strong_fields.setdefault(ds_id, {})
                if abs_sh > ds_strong.get(fld, 0.0):
                    ds_strong[fld] = abs_sh
                self._maybe_enqueue_pairwise_synergy(ds_id, fld, abs_sh)

        for nm_res, nm_reason in near_misses_to_queue:
            self._enqueue_immediate_refinement(nm_res, nm_reason)

        return {"promoted": promoted, "passed": passed, "rejected": rejected, "unsimulated_released": unsimulated_released}


    def purge_false_failures(self) -> dict[str, int]:
        """Release funnel_tried & funnel_reference_db entries that never actually simulated (sharpe IS NULL)."""
        with self._connect() as conn:
            before_tried = int(conn.execute("SELECT COUNT(*) FROM funnel_tried").fetchone()[0] or 0)
            before_ref = int(conn.execute("SELECT COUNT(*) FROM funnel_reference_db").fetchone()[0] or 0)

            # 1. Remove REJECTED rows in funnel_reference_db that have NULL sharpe (unsimulated API failures)
            cur_ref = conn.execute(
                "DELETE FROM funnel_reference_db WHERE UPPER(outcome) = 'REJECTED' AND sharpe IS NULL"
            )
            deleted_ref = int(cur_ref.rowcount or 0)

            # 2. Remove funnel_tried rows that do NOT have a real simulated entry (sharpe IS NOT NULL) in funnel_reference_db
            cur_tried = conn.execute(
                """DELETE FROM funnel_tried
                   WHERE sim_hash NOT IN (
                       SELECT sim_hash FROM funnel_reference_db WHERE sharpe IS NOT NULL
                   )"""
            )
            deleted_tried = int(cur_tried.rowcount or 0)

            # 3. Reset heavily negative RL arms that only had false failures (seeds_found=0 and passed_count=0)
            conn.execute(
                """DELETE FROM funnel_rl_policy
                   WHERE reward_sum < -5.0 AND dataset_id IN (
                       SELECT dataset_id FROM funnel_dataset_stats WHERE seeds_found = 0 AND passed_count = 0
                   )"""
            )

            after_tried = int(conn.execute("SELECT COUNT(*) FROM funnel_tried").fetchone()[0] or 0)
            after_ref = int(conn.execute("SELECT COUNT(*) FROM funnel_reference_db").fetchone()[0] or 0)

        self._session_seen_hashes.clear()
        return {
            "before_tried": before_tried,
            "after_tried": after_tried,
            "deleted_tried": deleted_tried,
            "before_ref": before_ref,
            "after_ref": after_ref,
            "deleted_ref": deleted_ref,
        }

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
