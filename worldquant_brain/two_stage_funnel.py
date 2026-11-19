"""Two-Stage High-Throughput Alpha Discovery Funnel (Field Screening -> Grid Refinement).

Stage 1 (Exploration): Systematically sweeps high-coverage non-PV dataset fields using
clean canonical probes (rank, ts_zscore, group_rank delta, winsorized ts_rank).
Stage 2 (Exploitation): Automatically promotes any Stage 1 seed with |Sharpe| >= 0.90
(auto-negating negative-Sharpe expressions) and sweeps Neutralization x Decay x Truncation
plus turnover smoothing in Multi-Simulation batches.

All progress and seeds are persisted in SQLite so restarts never lose state or repeat work.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from worldquant_brain.multi_sim_client import DEFAULT_SIM_SETTINGS

STAGE1_PROBES: tuple[tuple[str, str], ...] = (
    ("rank_bf", "rank(ts_backfill({f}, 20))"),
    ("zscore60", "ts_zscore(ts_backfill({f}, 20), 60)"),
    ("grp_delta20", "group_rank(ts_delta(ts_backfill({f}, 20), 20), subindustry)"),
    ("tsrank120", "winsorize(ts_rank(ts_backfill({f}, 20), 120), std=4)"),
)

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


def _sim_hash(expression: str, settings: Mapping[str, Any]) -> str:
    norm = {
        "expr": " ".join(str(expression).strip().split()),
        "neutralization": str(settings.get("neutralization", "SUBINDUSTRY")).upper(),
        "decay": int(settings.get("decay", 4) or 0),
        "truncation": round(float(settings.get("truncation", 0.08) or 0.08), 4),
        "universe": str(settings.get("universe", "TOP3000")).upper(),
        "delay": int(settings.get("delay", 1) or 1),
    }
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode("utf-8")).hexdigest()


def _negate_expression(expr: str) -> str:
    s = str(expr).strip()
    if s.startswith("-(") and s.endswith(")"):
        return s[2:-1].strip()
    if s.startswith("-"):
        return s[1:].strip()
    return f"-({s})"


class TwoStageAlphaFunnel:
    """SQLite-backed two-stage alpha discovery scheduler."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        seed_sharpe_threshold: float = 0.90,
        base_settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.seed_sharpe_threshold = float(seed_sharpe_threshold)
        self.base_settings = dict(DEFAULT_SIM_SETTINGS)
        if base_settings:
            self.base_settings.update(dict(base_settings))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
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
                CREATE TABLE IF NOT EXISTS funnel_seeds (
                    seed_id TEXT PRIMARY KEY,
                    expression TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    dataset_id TEXT NOT NULL DEFAULT '',
                    source_alpha_id TEXT,
                    source_sharpe REAL NOT NULL,
                    source_turnover REAL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
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
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _is_tried(self, conn: sqlite3.Connection, sim_hash: str) -> bool:
        row = conn.execute("SELECT 1 FROM funnel_tried WHERE sim_hash = ?", (sim_hash,)).fetchone()
        return row is not None

    def _mark_tried(
        self,
        conn: sqlite3.Connection,
        sim_hash: str,
        expression: str,
        settings: Mapping[str, Any],
        stage: str,
        dataset_id: str,
    ) -> None:
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

    def pending_seed_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM funnel_seeds WHERE status = 'PENDING'").fetchone()
            return int(row[0] if row else 0)

    def next_stage2_batch(self, batch_size: int = 8) -> list[dict[str, Any]]:
        """Generate a Multi-Sim refinement batch from the highest-Sharpe pending seed."""
        out: list[dict[str, Any]] = []
        with self._connect() as conn:
            seeds = conn.execute(
                "SELECT * FROM funnel_seeds WHERE status = 'PENDING' ORDER BY source_sharpe DESC LIMIT 3"
            ).fetchall()
            for seed in seeds:
                expr = str(seed["expression"])
                base_sett = dict(self.base_settings)
                try:
                    base_sett.update(json.loads(seed["settings_json"] or "{}"))
                except Exception:
                    pass
                dataset_id = str(seed["dataset_id"] or "")
                src_to = seed["source_turnover"]

                candidates: list[tuple[str, dict[str, Any]]] = []
                # 1. Neutralization sweep
                for neut in STAGE2_NEUTRALIZATIONS:
                    s = dict(base_sett, neutralization=neut)
                    candidates.append((expr, s))
                # 2. Decay & truncation sweep
                for dec in STAGE2_DECAYS:
                    s = dict(base_sett, decay=dec)
                    candidates.append((expr, s))
                for trunc in STAGE2_TRUNCATIONS:
                    s = dict(base_sett, truncation=trunc)
                    candidates.append((expr, s))
                # 3. Smoothing if turnover is high or unknown
                if src_to is None or float(src_to) > 0.35:
                    candidates.append((f"ts_decay_linear({expr}, 8)", dict(base_sett, decay=4)))
                    candidates.append((f"winsorize(ts_mean({expr}, 5), std=4)", dict(base_sett, decay=4)))

                added_for_seed = 0
                for c_expr, c_sett in candidates:
                    if len(out) >= batch_size:
                        break
                    sh = _sim_hash(c_expr, c_sett)
                    if self._is_tried(conn, sh):
                        continue
                    self._mark_tried(conn, sh, c_expr, c_sett, "STAGE2", dataset_id)
                    out.append(
                        {
                            "expression": c_expr,
                            "settings": c_sett,
                            "stage": "STAGE2",
                            "dataset_id": dataset_id,
                            "seed_id": seed["seed_id"],
                        }
                    )
                    added_for_seed += 1

                if added_for_seed == 0:
                    conn.execute("UPDATE funnel_seeds SET status = 'EXHAUSTED' WHERE seed_id = ?", (seed["seed_id"],))
                if len(out) >= batch_size:
                    break
        return out

    def next_stage1_batch(
        self,
        dataset_id: str,
        fields: Sequence[str],
        *,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Generate a Multi-Sim Stage 1 field-screening batch for a dataset."""
        out: list[dict[str, Any]] = []
        with self._connect() as conn:
            for probe_name, template in STAGE1_PROBES:
                for field in fields:
                    if len(out) >= batch_size:
                        return out
                    expr = template.format(f=field)
                    sett = dict(self.base_settings)
                    sh = _sim_hash(expr, sett)
                    if self._is_tried(conn, sh):
                        continue
                    self._mark_tried(conn, sh, expr, sett, "STAGE1", dataset_id)
                    out.append(
                        {
                            "expression": expr,
                            "settings": sett,
                            "stage": "STAGE1",
                            "dataset_id": dataset_id,
                            "probe": probe_name,
                            "field": field,
                        }
                    )
        return out

    def next_batch(
        self,
        datasets_fields: Mapping[str, Sequence[str]],
        *,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Return Stage 2 refinement batch if seeds exist; otherwise Stage 1 field screening."""
        s2 = self.next_stage2_batch(batch_size=batch_size)
        if s2:
            return s2
        for ds_id, fields in datasets_fields.items():
            s1 = self.next_stage1_batch(ds_id, fields, batch_size=batch_size)
            if s1:
                return s1
        return []

    def ingest_results(self, results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        """Record Multi-Sim results and automatically promote promising Stage 1 seeds to Stage 2."""
        promoted = 0
        passed = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for r in results:
                ds_id = str(r.get("dataset_id") or "")
                stage = str(r.get("stage") or "STAGE1")
                sharpe = r.get("sharpe")
                turnover = r.get("turnover")
                checks_passed = bool(r.get("checks_passed", False))

                if ds_id:
                    conn.execute(
                        """
                        INSERT INTO funnel_dataset_stats (dataset_id, screened_count, seeds_found, passed_count, updated_at)
                        VALUES (?, 1, 0, 0, ?)
                        ON CONFLICT(dataset_id) DO UPDATE SET
                            screened_count = screened_count + 1,
                            updated_at = excluded.updated_at
                        """,
                        (ds_id, now_iso),
                    )

                if sharpe is None:
                    continue
                sh_val = float(sharpe)
                if checks_passed and sh_val >= 1.25:
                    passed += 1
                    if ds_id:
                        conn.execute(
                            "UPDATE funnel_dataset_stats SET passed_count = passed_count + 1, updated_at = ? WHERE dataset_id = ?",
                            (now_iso, ds_id),
                        )

                if stage == "STAGE1" and abs(sh_val) >= self.seed_sharpe_threshold:
                    expr = str(r.get("expression") or "").strip()
                    if sh_val < 0:
                        expr = _negate_expression(expr)
                    sett = dict(r.get("settings") or self.base_settings)
                    seed_id = _sim_hash(expr, sett)[:20]
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO funnel_seeds (
                            seed_id, expression, settings_json, dataset_id,
                            source_alpha_id, source_sharpe, source_turnover, status, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                        """,
                        (
                            seed_id,
                            expr,
                            json.dumps(sett, sort_keys=True),
                            ds_id,
                            r.get("alpha_id"),
                            abs(sh_val),
                            float(turnover) if turnover is not None else None,
                            now_iso,
                        ),
                    )
                    if cur.rowcount > 0:
                        promoted += 1
                        if ds_id:
                            conn.execute(
                                "UPDATE funnel_dataset_stats SET seeds_found = seeds_found + 1, updated_at = ? WHERE dataset_id = ?",
                                (now_iso, ds_id),
                            )
        return {"promoted": promoted, "passed": passed}

    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            tried = int(conn.execute("SELECT COUNT(*) FROM funnel_tried").fetchone()[0])
            pending_seeds = int(conn.execute("SELECT COUNT(*) FROM funnel_seeds WHERE status = 'PENDING'").fetchone()[0])
            total_seeds = int(conn.execute("SELECT COUNT(*) FROM funnel_seeds").fetchone()[0])
            ds_rows = [dict(r) for r in conn.execute("SELECT * FROM funnel_dataset_stats ORDER BY seeds_found DESC").fetchall()]
        return {
            "tried_total": tried,
            "pending_seeds": pending_seeds,
            "total_seeds": total_seeds,
            "datasets": ds_rows,
        }
