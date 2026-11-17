"""Adaptive submission policy: submit every pending candidate, without local quality ranking.

The policy preserves only operational hard stops: explicit enablement, kill switch,
idempotency, cooldown, platform response, and an authenticated score-cap observation.
It intentionally does not impose a fixed number of submissions or local Sharpe/Fitness ranking.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from worldquant_brain.score_tracker import ScoreTracker


@dataclass(frozen=True)
class PolicyDecision:
    submit: bool
    reason: str


class SubmitAllAvailablePolicy:
    def __init__(
        self,
        db_path: str | Path,
        score_tracker: ScoreTracker,
        *,
        enabled: bool,
        kill_switch: Callable[[], bool] = lambda: False,
        acknowledgement_valid: Callable[[], bool] = lambda: False,
        cooldown_seconds: float = 2.0,
        order_by: str = "fifo",
    ) -> None:
        self.db_path = str(db_path)
        self.score_tracker = score_tracker
        self.enabled = bool(enabled)
        self.kill_switch = kill_switch
        self.acknowledgement_valid = acknowledgement_valid
        self.cooldown_seconds = float(cooldown_seconds)
        self.order_by = str(order_by or "fifo").lower()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS brain_adaptive_submission_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                candidates_seen INTEGER NOT NULL DEFAULT 0,
                attempted INTEGER NOT NULL DEFAULT 0,
                submitted INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stopped_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS brain_adaptive_submission_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                alpha_id TEXT NOT NULL,
                attempted_at TEXT NOT NULL,
                outcome TEXT NOT NULL,
                safe_error TEXT,
                UNIQUE(alpha_id, outcome) ON CONFLICT IGNORE
            );
            """)

    def decide(self, alpha_id: str) -> PolicyDecision:
        if not self.enabled:
            return PolicyDecision(False, "AUTO_SUBMIT_DISABLED")
        if self.kill_switch():
            return PolicyDecision(False, "KILL_SWITCH_ACTIVE")
        if not self.acknowledgement_valid():
            return PolicyDecision(False, "SUBMIT_ACK_INVALID")
        cap = self.score_tracker.cap_status()
        if cap["stop_allowed"] and cap["reached"]:
            return PolicyDecision(False, "AUTHENTICATED_DAILY_SCORE_CAP_REACHED")
        with self.connect() as conn:
            duplicate = conn.execute(
                "SELECT 1 FROM brain_adaptive_submission_events WHERE alpha_id=? AND outcome='SUBMITTED'",
                (alpha_id,),
            ).fetchone()
        if duplicate:
            return PolicyDecision(False, "ALREADY_SUBMITTED_IDEMPOTENT")
        return PolicyDecision(True, "SUBMIT_ANY_AVAILABLE")

    def pending_candidates(self) -> list[dict[str, Any]]:
        order_clause = "score DESC, created_at ASC, alpha_id ASC" if self.order_by == "score_desc" else "created_at ASC, alpha_id ASC"
        with self.connect() as conn:
            try:
                rows = conn.execute(f"""
                    SELECT alpha_id, expression, candidate_date, created_at, status
                    FROM brain_candidate_pool
                    WHERE status='PENDING'
                    ORDER BY {order_clause}
                """).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def run(self, submitter: Callable[[str], Mapping[str, Any]]) -> dict[str, Any]:
        import uuid
        run_id = uuid.uuid4().hex
        started = datetime.now(timezone.utc).isoformat()
        candidates = self.pending_candidates()
        stats = {"run_id": run_id, "seen": len(candidates), "attempted": 0, "submitted": 0, "failed": 0}
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO brain_adaptive_submission_runs(run_id,started_at,candidates_seen) VALUES(?,?,?)",
                (run_id, started, len(candidates)),
            )
        stop_reason = None
        for candidate in candidates:
            alpha_id = str(candidate["alpha_id"])
            decision = self.decide(alpha_id)
            if not decision.submit:
                if decision.reason == "AUTHENTICATED_DAILY_SCORE_CAP_REACHED":
                    stop_reason = decision.reason
                    break
                continue
            stats["attempted"] += 1
            try:
                response = dict(submitter(alpha_id))
                success = bool(response.get("success")) or response.get("status") in {"SUBMITTED", "SUCCESS"}
                outcome = "SUBMITTED" if success else "FAILED"
                safe_error = None if success else str(response.get("error") or response.get("status") or "UNKNOWN")[:500]
            except Exception as exc:
                outcome = "FAILED"
                safe_error = type(exc).__name__
            stats["submitted" if outcome == "SUBMITTED" else "failed"] += 1
            with self.connect() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO brain_adaptive_submission_events
                    (run_id,alpha_id,attempted_at,outcome,safe_error) VALUES(?,?,?,?,?)
                """, (run_id, alpha_id, datetime.now(timezone.utc).isoformat(), outcome, safe_error))
                if outcome == "SUBMITTED":
                    conn.execute(
                        "UPDATE brain_candidate_pool SET status='SUBMITTED' WHERE alpha_id=?", (alpha_id,)
                    )
            if self.cooldown_seconds > 0:
                time.sleep(self.cooldown_seconds)
        completed = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute("""
                UPDATE brain_adaptive_submission_runs SET completed_at=?, attempted=?, submitted=?,
                    failed=?, stopped_reason=? WHERE run_id=?
            """, (completed, stats["attempted"], stats["submitted"], stats["failed"], stop_reason, run_id))
        stats["stopped_reason"] = stop_reason
        return stats
