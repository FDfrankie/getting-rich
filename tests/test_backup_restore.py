import os
import tempfile

from data_persistence.database_migration import DatabaseMigrationManager


def test_backup_and_restore_cycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "primary.db")
        backup_dir = os.path.join(tmpdir, "backups")
        mgr = DatabaseMigrationManager(db_path=db_path, backup_dir=backup_dir)

        # 寫入初始資料
        conn = mgr.get_connection()
        conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, msg TEXT)")
        conn.execute("INSERT INTO test_data (msg) VALUES ('original_data')")
        conn.commit()
        conn.close()

        # 執行 backup
        backup_file = mgr.backup_database(prefix="test_backup")
        assert os.path.exists(backup_file)

        # 修改原始資料 (破壞性操作)
        conn = mgr.get_connection()
        conn.execute("UPDATE test_data SET msg = 'corrupted_data'")
        conn.commit()
        conn.close()

        # 執行 restore
        mgr.restore_database(backup_file)

        # 驗證資料已成功還原為 original_data
        conn = mgr.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT msg FROM test_data WHERE id = 1")
        restored_msg = cur.fetchone()[0]
        conn.close()

        assert restored_msg == "original_data"
