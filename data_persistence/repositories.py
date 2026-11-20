"""Unified SQLite & PostgreSQL repository interfaces with transactional advisory locks (Phase 2)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from data_persistence.migrations import REQUIRED_TABLES, apply_sqlite_migrations


def advisory_lock_key(namespace: str, key: str) -> int:
    """Compute a deterministic 64-bit signed integer key for PostgreSQL pg_try_advisory_lock."""
    digest = hashlib.sha256(f"{namespace}:{key}".encode("utf-8")).digest()[:8]
    val = int.from_bytes(digest, byteorder="big", signed=False)
    return val - (1 << 63) if val >= (1 << 63) else val


class DurableRepositoryProtocol(Protocol):
    def acquire_advisory_lock(self, namespace: str, key: str, owner: str) -> bool: ...
    def release_advisory_lock(self, namespace: str, key: str, owner: str) -> None: ...
    def upsert_alpha(self, alpha: Mapping[str, Any]) -> dict[str, Any]: ...
    def record_audit_event(self, event_type: str, actor: str, entity_id: str | None, previous_state: str | None, next_state: str | None, reason: str, payload: Mapping[str, Any] | None = None) -> str: ...
    def claim_daily_scheduler_run(self, taipei_date: str, worker_id: str, scheduled_at: str) -> tuple[bool, str | None]: ...


class SQLiteRepository:
    """Local single-process SQLite repository implementing all 9 domain repository contracts."""

    _process_locks: dict[int, str] = {}
    _lock_guard = threading.Lock()

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS advisory_locks (lock_id INTEGER PRIMARY KEY, owner TEXT NOT NULL, acquired_at TEXT NOT NULL)")
            apply_sqlite_migrations(conn)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def acquire_advisory_lock(self, namespace: str, key: str, owner: str) -> bool:
        lock_id = advisory_lock_key(namespace, key)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock_guard:
            with self.connect() as conn:
                try:
                    conn.execute(
                        "INSERT INTO advisory_locks (lock_id, owner, acquired_at) VALUES (?, ?, ?)",
                        (lock_id, owner, now),
                    )
                    return True
                except sqlite3.IntegrityError:
                    return False

    def release_advisory_lock(self, namespace: str, key: str, owner: str) -> None:
        lock_id = advisory_lock_key(namespace, key)
        with self._lock_guard:
            with self.connect() as conn:
                conn.execute("DELETE FROM advisory_locks WHERE lock_id=? AND owner=?", (lock_id, owner))

    def upsert_alpha(self, alpha: Mapping[str, Any]) -> dict[str, Any]:
        expr = str(alpha.get("expression") or "").strip()
        expr_hash = str(alpha.get("expression_hash") or hashlib.sha256(expr.encode("utf-8")).hexdigest())
        alpha_id = str(alpha.get("alpha_id") or alpha.get("name") or f"alpha_{expr_hash[:12]}")
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            existing = conn.execute("SELECT * FROM alphas WHERE expression_hash=?", (expr_hash,)).fetchone()
            if existing:
                return dict(existing) | {"idempotent_duplicate": True}
            conn.execute(
                """
                INSERT INTO alphas (
                    alpha_id, name, expression, expression_hash, ast_family_hash, status,
                    brain_simulation_id, brain_alpha_id, sharpe, fitness, turnover, margin_bps,
                    drawdown, is_check_status, self_correlation, submission_utility,
                    utility_components_json, submitted_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alpha_id,
                    str(alpha.get("name") or alpha_id),
                    expr,
                    expr_hash,
                    alpha.get("ast_family_hash"),
                    str(alpha.get("status") or "GENERATED"),
                    alpha.get("brain_simulation_id"),
                    alpha.get("brain_alpha_id"),
                    alpha.get("sharpe"),
                    alpha.get("fitness"),
                    alpha.get("turnover"),
                    alpha.get("margin_bps"),
                    alpha.get("drawdown"),
                    alpha.get("is_check_status"),
                    alpha.get("self_correlation"),
                    alpha.get("submission_utility"),
                    json.dumps(alpha.get("utility_components") or {}, sort_keys=True),
                    alpha.get("submitted_at"),
                    str(alpha.get("created_at") or now),
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM alphas WHERE expression_hash=?", (expr_hash,)).fetchone()
        return dict(row) | {"idempotent_duplicate": False}

    def claim_daily_scheduler_run(self, taipei_date: str, worker_id: str, scheduled_at: str) -> tuple[bool, str | None]:
        if not self.acquire_advisory_lock("scheduler_daily", taipei_date, worker_id):
            return False, None
        run_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO brain_scheduler_runs (taipei_date, run_id, scheduled_at, claimed_at, claimed_by, status)
                    VALUES (?, ?, ?, ?, ?, 'CLAIMED')
                    """,
                    (taipei_date, run_id, scheduled_at, now, worker_id),
                )
                return True, run_id
            except sqlite3.IntegrityError:
                return False, None

    def record_audit_event(
        self,
        event_type: str,
        actor: str,
        entity_id: str | None,
        previous_state: str | None,
        next_state: str | None,
        reason: str,
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        event_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (event_id, event_type, actor, entity_id, previous_state, next_state, reason, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type,
                    actor,
                    entity_id,
                    previous_state,
                    next_state,
                    reason,
                    json.dumps(dict(payload or {}), ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
        return event_id


_repo_logger = logging.getLogger(__name__)


class PostgresRepository(SQLiteRepository):
    """PostgreSQL compatibility adapter. Currently persists to a local SQLite fallback until native psycopg driver is enabled."""

    def __init__(self, database_url: str | None = None, fallback_sqlite_path: str | Path | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        self.is_sqlite_fallback = True
        fallback = fallback_sqlite_path or (Path(__file__).resolve().parent / "postgres_compat.db")
        if self.database_url:
            _repo_logger.warning(
                "PostgresRepository: DATABASE_URL is configured, but native PostgreSQL driver is not enabled; "
                "falling back to local SQLite at %s",
                fallback,
            )
        super().__init__(fallback)

    def pg_advisory_lock_sql(self, namespace: str, key: str) -> tuple[str, int]:
        return "SELECT pg_try_advisory_lock(%s);", advisory_lock_key(namespace, key)
