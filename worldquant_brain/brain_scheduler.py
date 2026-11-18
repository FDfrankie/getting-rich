"""Restart-safe Asia/Taipei scheduler state using SQLite claims."""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True)
class DailyClaim:
    acquired: bool
    taipei_date: str
    run_id: str | None
    reason: str


class BrainDailyScheduler:
    def __init__(self, db_path: str | Path, flush_hour: int = 22) -> None:
        if not 0 <= flush_hour <= 23:
            raise ValueError("flush_hour must be 0..23")
        self.db_path = str(db_path)
        self.flush_hour = flush_hour
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS brain_daily_runs (
                    taipei_date TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    scheduled_at TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    claimed_by TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    candidates_considered INTEGER NOT NULL DEFAULT 0,
                    candidates_submitted INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT
                )
            """)

    def next_flush_at(self, now: datetime | None = None) -> datetime:
        current = (now or datetime.now(TAIPEI)).astimezone(TAIPEI)
        today = datetime.combine(current.date(), time(self.flush_hour), TAIPEI)
        return today if current < today else today + timedelta(days=1)

    def claim_due_run(self, *, worker_id: str, now: datetime | None = None) -> DailyClaim:
        current = (now or datetime.now(TAIPEI)).astimezone(TAIPEI)
        scheduled = datetime.combine(current.date(), time(self.flush_hour), TAIPEI)
        date_key = current.date().isoformat()
        if current < scheduled:
            return DailyClaim(False, date_key, None, "NOT_DUE")
        run_id = uuid.uuid4().hex
        with self.connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("""
                    INSERT INTO brain_daily_runs (
                        taipei_date, run_id, scheduled_at, claimed_at, claimed_by, status
                    ) VALUES (?, ?, ?, ?, ?, 'CLAIMED')
                """, (date_key, run_id, scheduled.isoformat(), current.isoformat(), worker_id))
                conn.execute("COMMIT")
                return DailyClaim(True, date_key, run_id, "CLAIMED")
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK")
                return DailyClaim(False, date_key, None, "ALREADY_CLAIMED")

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        considered: int = 0,
        submitted: int = 0,
        error_code: str | None = None,
        error_message: str | None = None,
        now: datetime | None = None,
    ) -> None:
        current = (now or datetime.now(TAIPEI)).astimezone(TAIPEI)
        with self.connect() as conn:
            conn.execute("""
                UPDATE brain_daily_runs SET completed_at=?, status=?,
                    candidates_considered=?, candidates_submitted=?,
                    error_code=?, error_message=? WHERE run_id=?
            """, (current.isoformat(), status, considered, submitted, error_code, error_message, run_id))

    def status(self, now: datetime | None = None) -> dict:
        current = (now or datetime.now(TAIPEI)).astimezone(TAIPEI)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM brain_daily_runs ORDER BY taipei_date DESC LIMIT 1"
            ).fetchone()
        return {
            "timezone": "Asia/Taipei",
            "daily_flush_hour": self.flush_hour,
            "next_flush_at": self.next_flush_at(current).isoformat(),
            "last_run": dict(row) if row else None,
            "server_required": True,
        }
