"""Persistent priority ingress for manual and automatic BRAIN simulations."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from worldquant_brain.alpha_experiment_store import expression_hash, normalize_expression
from worldquant_brain.sqlite_resilience import ensure_schema_from_ddl

PRIORITY = {
    "MANUAL_HIGH": 100,
    "REVALIDATION": 80,
    "AUTO_HIGH_SCORE": 50,
    "AUTO_MINING": 20,
    "EXPLORATION": 5,
}


class ManualPriorityQueue:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
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
            CREATE TABLE IF NOT EXISTS brain_priority_jobs (
                job_id TEXT PRIMARY KEY,
                expression_hash TEXT NOT NULL,
                normalized_expression TEXT NOT NULL,
                expression TEXT NOT NULL,
                settings_json TEXT NOT NULL DEFAULT '{}',
                priority INTEGER NOT NULL,
                source TEXT NOT NULL,
                requested_by TEXT,
                manual_note TEXT,
                simulation_only INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'PENDING',
                claimed_by TEXT,
                claimed_at TEXT,
                queued_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_brain_priority_active_expression
                ON brain_priority_jobs(expression_hash)
                WHERE status IN ('PENDING','CLAIMED','RUNNING','COMPLETED');
            CREATE INDEX IF NOT EXISTS ix_brain_priority_order
                ON brain_priority_jobs(status, priority DESC, queued_at ASC);
            """)
            ensure_schema_from_ddl(conn, "\n            CREATE TABLE IF NOT EXISTS brain_priority_jobs (\n                job_id TEXT PRIMARY KEY,\n                expression_hash TEXT NOT NULL,\n                normalized_expression TEXT NOT NULL,\n                expression TEXT NOT NULL,\n                settings_json TEXT NOT NULL DEFAULT '{}',\n                priority INTEGER NOT NULL,\n                source TEXT NOT NULL,\n                requested_by TEXT,\n                manual_note TEXT,\n                simulation_only INTEGER NOT NULL DEFAULT 1,\n                status TEXT NOT NULL DEFAULT 'PENDING',\n                claimed_by TEXT,\n                claimed_at TEXT,\n                queued_at TEXT NOT NULL,\n                completed_at TEXT\n            );\n            CREATE UNIQUE INDEX IF NOT EXISTS ux_brain_priority_active_expression\n                ON brain_priority_jobs(expression_hash)\n                WHERE status IN ('PENDING','CLAIMED','RUNNING','COMPLETED');\n            CREATE INDEX IF NOT EXISTS ix_brain_priority_order\n                ON brain_priority_jobs(status, priority DESC, queued_at ASC);\n            ")

    def enqueue(
        self,
        expression: str,
        *,
        settings: Mapping[str, Any] | None = None,
        priority: int = 100,
        source: str = "MANUAL",
        requested_by: str | None = None,
        manual_note: str | None = None,
        simulation_only: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        normalized = normalize_expression(expression)
        if not normalized:
            raise ValueError("expression is required")
        ehash = expression_hash(expression)
        with self.connect() as conn:
            existing = conn.execute("""
                SELECT * FROM brain_priority_jobs
                WHERE expression_hash=? AND status IN ('PENDING','CLAIMED','RUNNING','COMPLETED')
                ORDER BY queued_at DESC LIMIT 1
            """, (ehash,)).fetchone()
            if existing:
                return dict(existing), False
            job = {
                "job_id": f"manual_{uuid.uuid4().hex}",
                "expression_hash": ehash,
                "normalized_expression": normalized,
                "expression": expression,
                "settings_json": json.dumps(dict(settings or {}), sort_keys=True),
                "priority": int(priority),
                "source": source,
                "requested_by": requested_by,
                "manual_note": manual_note,
                "simulation_only": int(bool(simulation_only)),
                "status": "PENDING",
                "queued_at": datetime.now(timezone.utc).isoformat(),
            }
            conn.execute("""
                INSERT INTO brain_priority_jobs (
                    job_id, expression_hash, normalized_expression, expression, settings_json,
                    priority, source, requested_by, manual_note, simulation_only, status, queued_at
                ) VALUES (:job_id,:expression_hash,:normalized_expression,:expression,:settings_json,
                    :priority,:source,:requested_by,:manual_note,:simulation_only,:status,:queued_at)
            """, job)
            row = conn.execute("SELECT * FROM brain_priority_jobs WHERE job_id=?", (job["job_id"],)).fetchone()
            return dict(row), True

    def claim_next(self, worker_id: str) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("""
                SELECT job_id FROM brain_priority_jobs
                WHERE status='PENDING'
                ORDER BY priority DESC, queued_at ASC LIMIT 1
            """).fetchone()
            if not row:
                conn.execute("COMMIT")
                return None
            conn.execute("""
                UPDATE brain_priority_jobs
                SET status='CLAIMED', claimed_by=?, claimed_at=?
                WHERE job_id=? AND status='PENDING'
            """, (worker_id, now, row["job_id"]))
            claimed = conn.execute(
                "SELECT * FROM brain_priority_jobs WHERE job_id=?", (row["job_id"],)
            ).fetchone()
            conn.execute("COMMIT")
            return dict(claimed) if claimed else None

    def queue_position(self, job_id: str) -> int | None:
        with self.connect() as conn:
            target = conn.execute(
                "SELECT priority, queued_at, status FROM brain_priority_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if not target or target["status"] != "PENDING":
                return None
            count = conn.execute("""
                SELECT COUNT(*) FROM brain_priority_jobs
                WHERE status='PENDING' AND (
                    priority > ? OR (priority = ? AND queued_at < ?)
                )
            """, (target["priority"], target["priority"], target["queued_at"])).fetchone()[0]
            return int(count) + 1
