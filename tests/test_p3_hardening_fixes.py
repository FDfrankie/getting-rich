"""Comprehensive unit & legacy-DB schema-drift regression tests for P3 Hardening Bundle."""
from __future__ import annotations

import ast
import logging
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldquant_brain.sqlite_resilience import (
    ensure_schema_from_ddl,
    ensure_table_columns,
)


class TestP3HardeningFixes(unittest.TestCase):
    def test_ensure_schema_from_ddl_migrates_legacy_tables_idempotently(self) -> None:
        """Simulate legacy SQLite tables missing newer columns (exact class of bug as submit_failed_count)."""
        conn = sqlite3.connect(":memory:")
        # Legacy table with only 2 columns and an existing row
        conn.execute("CREATE TABLE legacy_metrics (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.execute("INSERT INTO legacy_metrics (id, name) VALUES (1, 'seed_a')")
        conn.commit()

        ddl = """
        CREATE TABLE IF NOT EXISTS legacy_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            submit_failed_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            score REAL NOT NULL
        );
        """
        migrated = ensure_schema_from_ddl(conn, ddl)
        self.assertIn("legacy_metrics", migrated)
        self.assertEqual(
            set(migrated["legacy_metrics"]),
            {"submit_failed_count", "last_error", "score"},
        )
        # Verify existing row survived and new columns are queryable
        row = conn.execute(
            "SELECT id, name, submit_failed_count, last_error, score FROM legacy_metrics WHERE id = 1"
        ).fetchone()
        self.assertEqual(row, (1, "seed_a", 0, None, 0.0))

        # Second run must be a no-op (idempotent)
        migrated_again = ensure_schema_from_ddl(conn, ddl)
        self.assertEqual(migrated_again, {})
        conn.close()

    def test_ensure_table_columns_rejects_invalid_identifier(self) -> None:
        conn = sqlite3.connect(":memory:")
        with self.assertRaises(ValueError):
            ensure_table_columns(conn, "bad; DROP TABLE x;--", {"a": "TEXT"})
        conn.close()

    def test_evidence_repository_legacy_db_auto_migrates_missing_column(self) -> None:
        """Real module test: pre-create stripped legacy table, then init EvidenceRepository."""
        from worldquant_brain.evidence_repository import EvidenceRepository

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "legacy_evidence.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "CREATE TABLE research_memory_versions (policy_version TEXT PRIMARY KEY)"
            )
            conn.commit()
            conn.close()

            repo = EvidenceRepository(str(db_path))
            repo.migrate()
            conn = sqlite3.connect(str(db_path))
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(research_memory_versions)").fetchall()
            }
            conn.close()
            self.assertIn("policy_version", cols)
            self.assertIn("config_sha256", cols)
            self.assertGreater(len(cols), 1)

    def test_order_manager_legacy_db_auto_migrates_missing_column(self) -> None:
        """Real module test: pre-create stripped idempotent_orders table, then init OrderManager."""
        from crypto_trading.order_manager import OrderManager

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = str(Path(td) / "legacy_orders.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE idempotent_orders (idempotency_key TEXT PRIMARY KEY)"
            )
            conn.commit()
            conn.close()

            mgr = OrderManager(db_path=db_path)
            self.assertIsNotNone(mgr)
            conn = sqlite3.connect(db_path)
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(idempotent_orders)").fetchall()
            }
            conn.close()
            self.assertIn("state", cols)
            self.assertIn("symbol", cols)

    def test_database_migration_handles_mixed_string_and_int_versions(self) -> None:
        """Ensure DatabaseMigrationManager does not crash with TypeError when migrations.py string versions exist."""
        from data_persistence.database_migration import DatabaseMigrationManager

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = str(Path(td) / "mixed_migrations.db")
            backup_dir = str(Path(td) / "backups")
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE schema_migrations (version TEXT PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
                ("010_canonical_schema", "string version from migrations.py", "2026-01-01T00:00:00Z"),
            )
            conn.commit()
            conn.close()

            mgr = DatabaseMigrationManager(db_path=db_path, backup_dir=backup_dir)
            result = mgr.run_migrations()
            self.assertTrue(result.get("success"), f"Expected success, got: {result}")
            self.assertEqual(result.get("current_version"), 3)

    def test_postgres_repository_marks_sqlite_fallback_and_warns(self) -> None:
        from data_persistence.repositories import PostgresRepository

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            fb = Path(td) / "pg_fb.db"
            with self.assertLogs("data_persistence.repositories", level=logging.WARNING) as cm:
                repo = PostgresRepository(
                    database_url="postgresql://user:pass@localhost:5432/quant",
                    fallback_sqlite_path=fb,
                )
            self.assertTrue(repo.is_sqlite_fallback)
            self.assertTrue(any("SQLite" in msg for msg in cm.output))

    def test_hardened_hotbeds_have_zero_silent_except_pass(self) -> None:
        """Static AST check: none of the hardened modules may contain bare `except Exception: pass`."""
        repo_root = Path(__file__).resolve().parent.parent
        checked_files = [
            "worldquant_brain/sqlite_resilience.py",
            "worldquant_brain/brain_client.py",
            "worldquant_brain/brain_job_queue.py",
            "worldquant_brain/alpha_miner.py",
            "worldquant_brain/multi_sim_client.py",
            "taiwan_stock/universe_registry.py",
            "crypto_trading/testnet_client.py",
            "web_ui/server.py",
        ]
        offenders: list[str] = []
        for rel in checked_files:
            src = (repo_root / rel).read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        is_broad = node.type is None or (
                            isinstance(node.type, ast.Name) and node.type.id == "Exception"
                        )
                        if is_broad:
                            offenders.append(f"{rel}:L{node.lineno}")
        self.assertEqual(offenders, [], f"Silent except:pass still present: {offenders}")

    def test_no_hardcoded_gdrive_path_in_kaggle_generator(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        src = (repo_root / "kaggle_notebooks/generate_notebooks.py").read_text(encoding="utf-8")
        self.assertNotIn("G:\\我的雲端硬碟", src)
        self.assertIn("KAGGLE_NOTEBOOKS_OUT_DIR", src)

    def test_utc_aware_timestamps_in_patched_modules(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        self.assertNotIn(
            "datetime.utcnow()",
            (repo_root / "crypto_trading/risk_manager.py").read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "datetime.now().strftime",
            (repo_root / "worldquant_brain/alpha_vault.py").read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "datetime.now().strftime",
            (repo_root / "crypto_trading/bot_runner.py").read_text(encoding="utf-8"),
        )

    def test_headless_miner_supports_submit_immediately_env(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        src = (repo_root / "scripts/headless_brain_miner.py").read_text(encoding="utf-8")
        self.assertIn("BRAIN_SUBMIT_IMMEDIATELY", src)
        self.assertIn("[MINER_CONFIG]", src)
        self.assertIn("[FLUSH_WAITING]", src)


if __name__ == "__main__":
    unittest.main()
