"""Local SQLite WAL cache for ACTIVE WorldQuant BRAIN alpha daily PnL diffs (pre-submit self-correlation guard)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class ActivePnlCache:
    """Cache daily PnL diffs of submitted ACTIVE alphas for exact pre-submit self-correlation screening."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS active_alpha_pnl (
                    alpha_id TEXT PRIMARY KEY,
                    region TEXT NOT NULL,
                    daily_diff_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_cumulative_pnl(self, alpha_id: str, region: str, cum_pnl: list[float]) -> None:
        if len(cum_pnl) < 30:
            return
        arr = np.asarray(cum_pnl, dtype=float)
        diffs = np.diff(arr)
        diffs = np.where(np.isfinite(diffs), diffs, 0.0)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(
                "INSERT OR REPLACE INTO active_alpha_pnl (alpha_id, region, daily_diff_json, updated_at) VALUES (?, ?, ?, ?)",
                (str(alpha_id), str(region).upper(), json.dumps(diffs.tolist()), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def max_local_self_correlation(
        self,
        candidate_cum_pnl: list[float],
        *,
        region: str = "USA",
    ) -> tuple[float, str | None]:
        if len(candidate_cum_pnl) < 30:
            return 0.0, None
        cand_diff = np.diff(np.asarray(candidate_cum_pnl, dtype=float))
        cand_diff = np.where(np.isfinite(cand_diff), cand_diff, 0.0)
        max_corr, worst_id = 0.0, None
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            rows = conn.execute(
                "SELECT alpha_id, daily_diff_json FROM active_alpha_pnl WHERE region = ?",
                (str(region).upper(),),
            ).fetchall()
        finally:
            conn.close()
        for aid, raw_json in rows:
            ref = np.asarray(json.loads(raw_json), dtype=float)
            n = min(len(cand_diff), len(ref))
            if n < 30:
                continue
            a, b = cand_diff[-n:], ref[-n:]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(c) and abs(c) > max_corr:
                max_corr, worst_id = abs(c), str(aid)
        return round(max_corr, 4), worst_id
