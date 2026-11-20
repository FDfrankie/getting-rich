"""Unified Trial Ledger Repository (Shared by WorldQuant Alpha & Crypto Pipelines).

Enforces:
  - Idempotent trial creation on (domain, input_snapshot_hash, policy_version, evaluator_version)
  - Exactly-once event appending via (trial_id, idempotency_key)
  - Parent / Root lineage tracking
  - Test-period contamination logging (blocks promotion when modified_after_test=True)
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class UnifiedTrialLedger:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def create_or_get_trial(
        self,
        *,
        domain: str,
        stage: str,
        policy_version: str,
        code_commit_sha: str,
        config_hash: str,
        data_snapshot_hash: str,
        input_snapshot_hash: str,
        evaluator_version: str = "v1",
        parent_trial_id: str | None = None,
        root_trial_id: str | None = None,
        evidence_level: str = "UNKNOWN",
        hypothesis_id: str | None = None,
    ) -> tuple[str, bool]:
        """Idempotently create or return existing trial_id for the same input snapshot."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT trial_id FROM unified_trials
                WHERE domain = ? AND input_snapshot_hash = ? AND policy_version = ? AND evaluator_version = ?
                """,
                (domain, input_snapshot_hash, policy_version, evaluator_version),
            ).fetchone()
            if row:
                return str(row["trial_id"]), False

            tid = str(uuid.uuid4())
            eff_root = root_trial_id or parent_trial_id or tid
            conn.execute(
                """
                INSERT INTO unified_trials (
                    trial_id, domain, stage, result_status, parent_trial_id, root_trial_id,
                    hypothesis_id, policy_version, evaluator_version, schema_version,
                    code_commit_sha, config_hash, data_snapshot_hash, input_snapshot_hash,
                    evidence_level, created_at
                ) VALUES (?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, 'trial-ledger-v1', ?, ?, ?, ?, ?, ?)
                """,
                (
                    tid, domain, stage, parent_trial_id, eff_root, hypothesis_id,
                    policy_version, evaluator_version, code_commit_sha, config_hash,
                    data_snapshot_hash, input_snapshot_hash, evidence_level, now,
                ),
            )
            return tid, True

    def append_event(
        self,
        trial_id: str,
        *,
        event_type: str,
        idempotency_key: str,
        payload: Mapping[str, Any] | None = None,
        evidence_level: str = "LOCAL_REPLICATION",
    ) -> bool:
        """Append an exactly-once event; returns True if inserted, False if duplicate."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO unified_trial_events (
                    event_id, trial_id, event_type, idempotency_key, payload_json, evidence_level, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), trial_id, event_type, idempotency_key, json.dumps(dict(payload or {}), sort_keys=True), evidence_level, now),
            )
            return cur.rowcount > 0

    def record_test_access(
        self,
        trial_id: str,
        *,
        consumer_name: str,
        purpose: str,
        consumed_by_generator: bool = False,
    ) -> None:
        """Log test-period reveal and mark trial contaminated if generator consumed test results."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trial_test_access_log (
                    access_id, trial_id, test_period_revealed, result_consumed_by_generator,
                    consumer_name, purpose, accessed_at
                ) VALUES (?, ?, 1, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), trial_id, 1 if consumed_by_generator else 0, consumer_name, purpose, now),
            )
            conn.execute(
                """
                UPDATE unified_trials
                SET test_revealed_at = COALESCE(test_revealed_at, ?),
                    modified_after_test = CASE WHEN ? = 1 THEN 1 ELSE modified_after_test END
                WHERE trial_id = ?
                """,
                (now, 1 if consumed_by_generator else 0, trial_id),
            )

    def is_promotion_blocked_by_contamination(self, trial_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT modified_after_test FROM unified_trials WHERE trial_id = ?",
                (trial_id,),
            ).fetchone()
            return bool(row and row["modified_after_test"])
