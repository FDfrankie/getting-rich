import os
import sqlite3
from datetime import datetime
from typing import Any

from data_persistence.gdrive_sync import _DB_LOCK, BACKUP_DIR, STATE_DB_PATH


class DatabaseMigrationManager:
    """
    SQLite 持久化資料庫遷移管理
    支援：
    1. 啟用 WAL (Write-Ahead Logging)、synchronous=FULL、foreign_keys=ON、busy_timeout=15000
    2. 建立完整資料表與版本控制表 schema_migrations
    3. 可重跑 migration 腳本
    4. 升級前自動執行 sqlite3.backup() 建立還原快照，並可執行 restore
    """

    def __init__(self, db_path: str = STATE_DB_PATH, backup_dir: str = BACKUP_DIR):
        self.db_path = db_path
        self.backup_dir = backup_dir
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = FULL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 15000;")
        return conn

    def backup_database(self, prefix: str = "migration") -> str:
        """升級前線上熱備份 (sqlite3.backup())"""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"{prefix}_backup_{ts}.db")
        with _DB_LOCK:
            if os.path.exists(self.db_path):
                src_conn = sqlite3.connect(self.db_path, timeout=15.0)
                dst_conn = sqlite3.connect(backup_file)
                src_conn.backup(dst_conn)
                dst_conn.close()
                src_conn.close()
            else:
                # 若主 DB 尚未存在，建立空檔案備份
                open(backup_file, "w").close()
        return backup_file

    def restore_database(self, backup_file: str) -> bool:
        """從備份檔案線上復原主資料庫 (sqlite3.backup())"""
        if not os.path.exists(backup_file):
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        with _DB_LOCK:
            src_conn = sqlite3.connect(backup_file)
            dst_conn = sqlite3.connect(self.db_path, timeout=15.0)
            src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()
        return True

    def init_migration_table(self, conn: sqlite3.Connection):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
        """)
        conn.commit()

    def get_applied_versions(self, conn: sqlite3.Connection) -> list[int]:
        self.init_migration_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT version FROM schema_migrations ORDER BY version ASC")
        return [row[0] for row in cur.fetchall()]

    def run_migrations(self) -> dict[str, Any]:
        """
        執行資料庫遷移
        每筆 migration 前先 backup，遇錯可安全還原
        """
        backup_file = self.backup_database(prefix="pre_migration")
        conn = self.get_connection()
        self.init_migration_table(conn)
        applied = set(self.get_applied_versions(conn))

        migrations = [
            (1, "Create core trading tables", self._migration_v1),
            (2, "Create idempotency and manual control tables", self._migration_v2),
            (3, "Add reconciliation indices and constraints", self._migration_v3),
        ]

        newly_applied = []
        try:
            for ver, desc, func in migrations:
                if ver not in applied:
                    func(conn)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
                        (ver, desc, datetime.utcnow().isoformat()),
                    )
                    conn.commit()
                    newly_applied.append(ver)
            conn.close()
            return {
                "success": True,
                "backup_file": backup_file,
                "newly_applied": newly_applied,
                "current_version": max(applied.union(newly_applied)) if (applied or newly_applied) else 0,
            }
        except Exception as e:
            conn.close()
            # 自動還原
            self.restore_database(backup_file)
            return {"success": False, "error": str(e), "restored_from": backup_file}

    def _migration_v1(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                message TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                cost REAL,
                fee REAL,
                status TEXT NOT NULL,
                reason TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                contracts REAL NOT NULL,
                entry_price REAL NOT NULL,
                unrealized_pnl REAL,
                leverage INTEGER DEFAULT 1,
                updated_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS account_equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_equity REAL NOT NULL,
                balance_usdt REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

    def _migration_v2(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS idempotent_orders (
                client_order_id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE NOT NULL,
                exchange_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                amount REAL NOT NULL,
                price REAL,
                state TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS manual_overrides (
                symbol TEXT PRIMARY KEY,
                is_locked INTEGER NOT NULL DEFAULT 1,
                lock_reason TEXT,
                locked_at TEXT NOT NULL,
                ttl_seconds INTEGER NOT NULL DEFAULT 3600,
                stop_loss_price REAL,
                take_profit_price REAL,
                updated_at TEXT NOT NULL
            )
        """)

    def _migration_v3(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_idempotent_orders_key ON idempotent_orders(idempotency_key)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
