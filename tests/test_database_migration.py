import os
import tempfile
import sqlite3
import pytest
from data_persistence.database_migration import DatabaseMigrationManager

def test_database_migration_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_state.db")
        backup_dir = os.path.join(tmpdir, "backups")
        mgr = DatabaseMigrationManager(db_path=db_path, backup_dir=backup_dir)

        # 1. 初次遷移
        res = mgr.run_migrations()
        assert res["success"] is True
        assert 1 in res["newly_applied"]
        assert 2 in res["newly_applied"]
        assert 3 in res["newly_applied"]
        assert res["current_version"] == 3

        # 2. 驗證 WAL PRAGMA 設定
        conn = mgr.get_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode.lower() == "wal"

        cur.execute("PRAGMA foreign_keys")
        fk = cur.fetchone()[0]
        assert fk == 1
        conn.close()

        # 3. 再次重跑 migration (idempotent / 可重跑性)
        res2 = mgr.run_migrations()
        assert res2["success"] is True
        assert len(res2["newly_applied"]) == 0
        assert res2["current_version"] == 3
