#!/usr/bin/env python3
"""Idempotent Phase 9 schema migration. Does not enable BRAIN or submit Alphas."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from observability.audit_log import AuditLog
from worldquant_brain.alpha_experiment_store import AlphaExperimentStore, expression_hash
from worldquant_brain.brain_scheduler import BrainDailyScheduler
from worldquant_brain.manual_priority_queue import ManualPriorityQueue


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def add_column(conn: sqlite3.Connection, table: str, declaration: str) -> None:
    name = declaration.split()[0]
    if name not in columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {declaration}")


def migrate(db_path: Path) -> dict:
    AlphaExperimentStore(db_path)
    BrainDailyScheduler(db_path)
    ManualPriorityQueue(db_path)
    AuditLog(db_path)
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='alphas'").fetchone():
            for declaration in (
                "expression_hash TEXT",
                "brain_simulation_id TEXT",
                "brain_alpha_id TEXT",
                "is_check_status TEXT",
                "self_correlation REAL",
                "production_correlation REAL",
                "official_metrics_at TEXT",
                "submitted_at TEXT",
                "rejection_code TEXT",
                "rejection_reason TEXT",
            ):
                add_column(conn, "alphas", declaration)
            rows = conn.execute("SELECT id, expression FROM alphas WHERE expression_hash IS NULL").fetchall()
            for alpha_id, expression in rows:
                conn.execute(
                    "UPDATE alphas SET expression_hash=? WHERE id=?",
                    (expression_hash(expression), alpha_id),
                )
            conn.execute("CREATE INDEX IF NOT EXISTS ix_alphas_expression_hash ON alphas(expression_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_alphas_status ON alphas(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_alphas_simulation_id ON alphas(brain_simulation_id)")
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='brain_jobs'").fetchone():
            for declaration in (
                "priority INTEGER NOT NULL DEFAULT 20",
                "source TEXT NOT NULL DEFAULT 'AUTO_MINING'",
                "requested_by TEXT",
                "manual_note TEXT",
                "simulation_only INTEGER NOT NULL DEFAULT 1",
                "queued_at TEXT",
                "claimed_by TEXT",
                "claimed_at TEXT",
            ):
                add_column(conn, "brain_jobs", declaration)
            conn.execute("UPDATE brain_jobs SET queued_at=COALESCE(queued_at, created_at)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_brain_jobs_priority ON brain_jobs(status, priority DESC, queued_at ASC)"
            )
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    return {"db": str(db_path), "tables": sorted(tables), "status": "MIGRATED"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=Path("worldquant_brain/alpha_vault.db"))
    args = p.parse_args()
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        return 2
    print(migrate(args.db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
