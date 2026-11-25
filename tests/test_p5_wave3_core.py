"""Unit & regression tests for P5 Wave 3 Core Bundle."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from crypto_trading import model_governance as crypto_mg
import model_governance as unified_mg
from data_persistence.migrations import apply_sqlite_migrations
from observability.audit_log import AuditLog
from worldquant_brain.brain_client import BrainClient


class TestP5Wave3Core(unittest.TestCase):
    def test_brain_client_uses_distinct_thread_local_sessions_and_syncs_cookies(self) -> None:
        client = BrainClient(email="u@example.com", password="pw")
        main_sess = client.session
        main_sess.cookies.set("t", "cookie_token_123")
        with client._auth_lock:
            client._shared_cookies.update(main_sess.cookies.get_dict())
            client._auth_generation += 1

        worker_ids: list[int] = []
        worker_cookies: list[str | None] = []

        def worker() -> None:
            s = client.session
            worker_ids.append(id(s))
            worker_cookies.append(s.cookies.get("t"))

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(worker_ids), 2)
        self.assertNotIn(id(main_sess), worker_ids)
        self.assertEqual(worker_cookies, ["cookie_token_123", "cookie_token_123"])

        # Explicit injected session must still be respected for tests/mocks
        mock_sess = MagicMock()
        client_fixed = BrainClient(email="u@example.com", password="pw", session=mock_sess)
        self.assertIs(client_fixed.session, mock_sess)

    def test_audit_events_interop_between_migrations_and_audit_log_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "unified.db"
            # Run migrations twice (idempotency check)
            mconn = sqlite3.connect(str(db_path))
            try:
                apply_sqlite_migrations(mconn)
                apply_sqlite_migrations(mconn)
            finally:
                mconn.close()

            # AuditLog must write cleanly into the migrated audit_events table
            log = AuditLog(db_path)
            eid = log.write(
                component="brain_miner",
                action="submit",
                outcome="SUCCESS",
                actor="system",
                entity_id="alpha_123",
                details={"password": "secret_value", "sharpe": 1.8},
            )
            self.assertTrue(eid)

            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                "SELECT event_id, component, action, outcome, details_json FROM audit_events WHERE event_id = ?",
                (eid,),
            ).fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(row[1], "brain_miner")
            self.assertEqual(row[3], "SUCCESS")
            self.assertNotIn("secret_value", row[4])

    def test_009_correlation_evidence_null_safe_unique_index(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        sql = (repo_root / "migrations/009_correlation_evidence.sql").read_text(encoding="utf-8")
        conn = sqlite3.connect(":memory:")
        conn.executescript(sql)
        conn.execute(
            """
            INSERT INTO alpha_correlation_evidence (
                correlation_evidence_id, candidate_alpha_id, compared_alpha_id,
                correlation_type, correlation_source, pnl_period_start, pnl_period_end, checked_at
            ) VALUES ('e1', 'A1', NULL, 'SELF', 'BRAIN_API', NULL, NULL, '2026-11-21T00:00:00Z')
            """
        )
        # Second identical row with NULLs MUST raise IntegrityError thanks to COALESCE unique index
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO alpha_correlation_evidence (
                    correlation_evidence_id, candidate_alpha_id, compared_alpha_id,
                    correlation_type, correlation_source, pnl_period_start, pnl_period_end, checked_at
                ) VALUES ('e2', 'A1', NULL, 'SELF', 'BRAIN_API', NULL, NULL, '2026-11-21T00:01:00Z')
                """
            )
        conn.close()

    def test_model_governance_unified_and_taiwan_manifest_normalized(self) -> None:
        self.assertIs(crypto_mg.validate_model_manifest, unified_mg.validate_model_manifest)
        self.assertIs(crypto_mg.ManifestDecision, unified_mg.ManifestDecision)
        self.assertIs(crypto_mg.resolve_manifest_path, unified_mg.resolve_manifest_path)

        repo_root = Path(__file__).resolve().parent.parent
        tw_manifest = json.loads(
            (repo_root / "data_persistence/models/taiwan_stock_manifest.json").read_text(encoding="utf-8")
        )
        self.assertIn("feature_columns", tw_manifest)
        self.assertIn("symbols", tw_manifest)
        self.assertEqual(tw_manifest["feature_columns"], tw_manifest["feature_cols"])
        self.assertEqual(tw_manifest["symbols"], tw_manifest["universe_symbols"])


if __name__ == "__main__":
    unittest.main()
