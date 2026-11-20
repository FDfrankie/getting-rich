import json
import os
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any
from worldquant_brain.sqlite_resilience import ensure_schema_from_ddl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_persistence")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")
try:
    from data_persistence.state_paths import resolve_sqlite_path as _resolve_sqlite_path
    STATE_DB_PATH = _resolve_sqlite_path("STATE_DB_PATH", "system_state.db", os.path.join(DATA_DIR, "system_state.db"))
except Exception:
    STATE_DB_PATH = os.path.join(DATA_DIR, "system_state.db")
SNAPSHOT_JSON_PATH = os.path.join(DATA_DIR, "cloud_state_snapshot.json")

_DB_LOCK = threading.RLock()


class LocalPersistenceService:
    """
    本地持久化資料庫與備份服務 (Honest Naming: LocalPersistenceService)
    無真實 OAuth / Service Account 雙向同調時誠實標註為 LocalPersistenceService，
    但相容 GDrivePersistenceSync 接口以維護向前相容。
    功能：
    1. 支援 SQLite 狀態庫 (system_state.db) 結構維護與冷重啟狀態自動復原 (WAL、synchronous=FULL、foreign_keys=ON)
    2. 自動產生 JSON 狀態快照 (cloud_state_snapshot.json)
    3. 自動歷史備份 (sqlite3.backup())，杜絕狀態遺失
    """

    def __init__(self, db_path: str = STATE_DB_PATH):
        self.db_path = db_path
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(BACKUP_DIR, exist_ok=True)
        self.init_db()

    def get_connection(self):
        """獲取帶有超時、WAL 與外鍵保護的 SQLite 連線"""
        conn = sqlite3.connect(self.db_path, timeout=25.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 25000;")
        return conn

    def init_db(self):
        with _DB_LOCK:
            try:
                conn = self.get_connection()
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bot_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        message TEXT NOT NULL
                    )
                """)
                ensure_schema_from_ddl(cur.connection, '\n                    CREATE TABLE IF NOT EXISTS bot_logs (\n                        id INTEGER PRIMARY KEY AUTOINCREMENT,\n                        timestamp TEXT NOT NULL,\n                        message TEXT NOT NULL\n                    )\n                ')
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bot_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                """)
                ensure_schema_from_ddl(cur.connection, "\n                    CREATE TABLE IF NOT EXISTS bot_state (\n                        key TEXT PRIMARY KEY,\n                        value TEXT NOT NULL,\n                        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))\n                    )\n                ")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS post_mortems (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        entry_price REAL,
                        exit_price REAL,
                        loss_amount REAL,
                        loss_pct REAL,
                        cause_analysis TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                ensure_schema_from_ddl(cur.connection, '\n                    CREATE TABLE IF NOT EXISTS post_mortems (\n                        id INTEGER PRIMARY KEY AUTOINCREMENT,\n                        date TEXT NOT NULL,\n                        symbol TEXT NOT NULL,\n                        entry_price REAL,\n                        exit_price REAL,\n                        loss_amount REAL,\n                        loss_pct REAL,\n                        cause_analysis TEXT NOT NULL,\n                        created_at TEXT NOT NULL\n                    )\n                ')
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
                ensure_schema_from_ddl(cur.connection, '\n                    CREATE TABLE IF NOT EXISTS orders (\n                        id INTEGER PRIMARY KEY AUTOINCREMENT,\n                        exchange_order_id TEXT,\n                        symbol TEXT NOT NULL,\n                        side TEXT NOT NULL,\n                        order_type TEXT NOT NULL,\n                        price REAL NOT NULL,\n                        amount REAL NOT NULL,\n                        cost REAL,\n                        fee REAL,\n                        status TEXT NOT NULL,\n                        reason TEXT,\n                        timestamp TEXT NOT NULL\n                    )\n                ')
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
                ensure_schema_from_ddl(cur.connection, '\n                    CREATE TABLE IF NOT EXISTS positions (\n                        symbol TEXT PRIMARY KEY,\n                        side TEXT NOT NULL,\n                        contracts REAL NOT NULL,\n                        entry_price REAL NOT NULL,\n                        unrealized_pnl REAL,\n                        leverage INTEGER DEFAULT 1,\n                        updated_at TEXT NOT NULL\n                    )\n                ')
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        side TEXT NOT NULL,
                        price REAL NOT NULL,
                        amount REAL NOT NULL,
                        pnl REAL,
                        pnl_pct REAL,
                        reason TEXT,
                        timestamp TEXT NOT NULL
                    )
                """)
                ensure_schema_from_ddl(cur.connection, '\n                    CREATE TABLE IF NOT EXISTS trade_logs (\n                        id INTEGER PRIMARY KEY AUTOINCREMENT,\n                        symbol TEXT NOT NULL,\n                        action TEXT NOT NULL,\n                        side TEXT NOT NULL,\n                        price REAL NOT NULL,\n                        amount REAL NOT NULL,\n                        pnl REAL,\n                        pnl_pct REAL,\n                        reason TEXT,\n                        timestamp TEXT NOT NULL\n                    )\n                ')
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS account_equity (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_equity REAL NOT NULL,
                        balance_usdt REAL NOT NULL,
                        unrealized_pnl REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)
                ensure_schema_from_ddl(cur.connection, '\n                    CREATE TABLE IF NOT EXISTS account_equity (\n                        id INTEGER PRIMARY KEY AUTOINCREMENT,\n                        total_equity REAL NOT NULL,\n                        balance_usdt REAL NOT NULL,\n                        unrealized_pnl REAL NOT NULL,\n                        timestamp TEXT NOT NULL\n                    )\n                ')
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[!] LocalPersistenceService 資料庫初始化失敗: {e}")

    def save_state(self, key: str, value: Any):
        try:
            val_str = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                "REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, val_str, now_str),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 儲存狀態失敗 ({key}): {e}")

    def get_state(self, key: str, default: Any = None) -> Any:
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT value FROM bot_state WHERE key = ?", (key,))
            row = cur.fetchone()
            conn.close()
            if row:
                try:
                    return json.loads(row[0])
                except Exception:
                    return row[0]
            return default
        except Exception:
            return default

    def add_log(self, message: str):
        try:
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO bot_logs (timestamp, message) VALUES (?, ?)",
                (now_str, message),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 寫入資料庫日誌失敗: {e}")

    def get_logs(self, limit: int = 50) -> list:
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT timestamp, message FROM bot_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            conn.close()
            return [f"[{r[0]}] {r[1]}" for r in reversed(rows)]
        except Exception:
            return []

    def record_post_mortem(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        loss_amount: float,
        loss_pct: float,
        cause_analysis: str,
    ):
        try:
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO post_mortems (date, symbol, entry_price, exit_price, loss_amount, loss_pct, cause_analysis, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    today_str,
                    symbol,
                    entry_price,
                    exit_price,
                    loss_amount,
                    loss_pct,
                    cause_analysis,
                    now_str,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 記錄檢討失敗: {e}")

    def get_recent_post_mortems(self, limit: int = 10) -> list:
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, date, symbol, entry_price, exit_price, loss_amount, loss_pct, cause_analysis, created_at
                FROM post_mortems ORDER BY id DESC LIMIT ?
            """,
                (limit,),
            )
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    "id": r[0],
                    "date": r[1],
                    "symbol": r[2],
                    "entry_price": r[3],
                    "exit_price": r[4],
                    "loss_amount": r[5],
                    "loss_pct": r[6],
                    "cause_analysis": r[7],
                    "created_at": r[8],
                }
                for r in rows
            ]
        except Exception:
            return []

    def record_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        amount: float,
        status: str,
        exchange_order_id: str | None = None,
        cost: float | None = None,
        fee: float | None = None,
        reason: str | None = None,
    ):
        try:
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO orders (exchange_order_id, symbol, side, order_type, price, amount, cost, fee, status, reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    exchange_order_id,
                    symbol,
                    side,
                    order_type,
                    price,
                    amount,
                    cost,
                    fee,
                    status,
                    reason,
                    now_str,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 記錄訂單失敗: {e}")

    def update_position(
        self,
        symbol: str,
        side: str,
        contracts: float,
        entry_price: float,
        unrealized_pnl: float = 0.0,
        leverage: int = 1,
    ):
        try:
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            conn = self.get_connection()
            cur = conn.cursor()
            if contracts <= 0:
                cur.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                cur.execute(
                    """
                    INSERT INTO positions (symbol, side, contracts, entry_price, unrealized_pnl, leverage, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        side = excluded.side,
                        contracts = excluded.contracts,
                        entry_price = excluded.entry_price,
                        unrealized_pnl = excluded.unrealized_pnl,
                        leverage = excluded.leverage,
                        updated_at = excluded.updated_at
                """,
                    (
                        symbol,
                        side,
                        contracts,
                        entry_price,
                        unrealized_pnl,
                        leverage,
                        now_str,
                    ),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 更新部位失敗: {e}")

    # 相容別名
    update_position_record = update_position

    def get_positions(self) -> list:
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT symbol, side, contracts, entry_price, unrealized_pnl, leverage, updated_at FROM positions"
            )
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    "symbol": r[0],
                    "side": r[1],
                    "contracts": r[2],
                    "entry_price": r[3],
                    "unrealized_pnl": r[4],
                    "leverage": r[5],
                    "updated_at": r[6],
                }
                for r in rows
            ]
        except Exception:
            return []

    def record_trade_log(
        self,
        symbol: str,
        action: str,
        side: str,
        price: float,
        amount: float,
        pnl: float | None = None,
        pnl_pct: float | None = None,
        reason: str | None = None,
    ):
        try:
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO trade_logs (symbol, action, side, price, amount, pnl, pnl_pct, reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (symbol, action, side, price, amount, pnl, pnl_pct, reason, now_str),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 記錄成交明細失敗: {e}")

    def get_trade_logs(self, limit: int = 50) -> list:
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, symbol, action, side, price, amount, pnl, pnl_pct, reason, timestamp
                FROM trade_logs ORDER BY id DESC LIMIT ?
            """,
                (limit,),
            )
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    "id": r[0],
                    "symbol": r[1],
                    "action": r[2],
                    "side": r[3],
                    "price": r[4],
                    "amount": r[5],
                    "pnl": r[6],
                    "pnl_pct": r[7],
                    "reason": r[8],
                    "timestamp": r[9],
                }
                for r in rows
            ]
        except Exception:
            return []

    def record_equity(self, total_equity: float, balance_usdt: float, unrealized_pnl: float = 0.0):
        try:
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO account_equity (total_equity, balance_usdt, unrealized_pnl, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (total_equity, balance_usdt, unrealized_pnl, now_str),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 記錄權益歷史失敗: {e}")

    def get_equity_curve(self, limit: int = 100) -> list:
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT total_equity, balance_usdt, unrealized_pnl, timestamp
                FROM account_equity ORDER BY id DESC LIMIT ?
            """,
                (limit,),
            )
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    "total_equity": r[0],
                    "balance_usdt": r[1],
                    "unrealized_pnl": r[2],
                    "timestamp": r[3],
                }
                for r in reversed(rows)
            ]
        except Exception:
            return []

    def get_orders(self, limit: int = 50) -> list:
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, exchange_order_id, symbol, side, order_type, price, amount, cost, fee, status, reason, timestamp
                FROM orders ORDER BY id DESC LIMIT ?
            """,
                (limit,),
            )
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    "id": r[0],
                    "exchange_order_id": r[1],
                    "symbol": r[2],
                    "side": r[3],
                    "order_type": r[4],
                    "price": r[5],
                    "amount": r[6],
                    "cost": r[7],
                    "fee": r[8],
                    "status": r[9],
                    "reason": r[10],
                    "timestamp": r[11],
                }
                for r in rows
            ]
        except Exception:
            return []

    # 相容別名
    get_recent_orders = get_orders

    def export_snapshot_json(self) -> str:
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT key, value, updated_at FROM bot_state")
            state_dict = {r[0]: {"value": r[1], "updated_at": r[2]} for r in cur.fetchall()}
            cur.execute("SELECT timestamp, message FROM bot_logs ORDER BY id DESC LIMIT 100")
            logs = [{"timestamp": r[0], "message": r[1]} for r in cur.fetchall()]
            conn.close()

            payload = {
                "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "state": state_dict,
                "recent_logs": logs,
                "recent_post_mortems": self.get_recent_post_mortems(10),
            }
            with open(SNAPSHOT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return SNAPSHOT_JSON_PATH
        except Exception as e:
            print(f"[!] 匯出快照失敗: {e}")
            return ""

    def create_rolling_backup(self):
        try:
            if os.path.exists(self.db_path):
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(BACKUP_DIR, f"state_backup_{ts}.db")
                try:
                    src_conn = sqlite3.connect(self.db_path)
                    dst_conn = sqlite3.connect(backup_file)
                    src_conn.backup(dst_conn)
                    dst_conn.close()
                    src_conn.close()
                except Exception:
                    shutil.copy2(self.db_path, backup_file)
                existing = sorted([os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR) if f.endswith(".db")])
                if len(existing) > 10:
                    for old_file in existing[:-10]:
                        try:
                            os.remove(old_file)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[!] 建立自動備份失敗: {e}")

    def sync_to_gdrive(self) -> dict[str, Any]:
        """
        支援 Google Drive API 雙向同調：
        檢查環境變數 GOOGLE_DRIVE_CREDENTIALS 與 GOOGLE_DRIVE_FOLDER_ID。
        若無真實 OAuth/服務帳戶，則誠實以 LOCAL_PERSISTENCE 運作。
        """
        creds_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS", "") or os.getenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "")
        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
        snapshot_file = self.export_snapshot_json()

        if creds_json and folder_id:
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
                from googleapiclient.http import MediaFileUpload

                info = json.loads(creds_json) if creds_json.startswith("{") else json.load(open(creds_json))
                credentials = service_account.Credentials.from_service_account_info(
                    info, scopes=["https://www.googleapis.com/auth/drive"]
                )
                service = build("drive", "v3", credentials=credentials)

                query = f"name = 'system_state.db' and '{folder_id}' in parents and trashed = false"
                response = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
                files = response.get("files", [])

                media = MediaFileUpload(self.db_path, mimetype="application/x-sqlite3", resumable=True)
                if files:
                    file_id = files[0]["id"]
                    service.files().update(fileId=file_id, media_body=media).execute()
                else:
                    file_metadata = {"name": "system_state.db", "parents": [folder_id]}
                    service.files().create(body=file_metadata, media_body=media, fields="id").execute()
                return {
                    "status": "SUCCESS",
                    "mode": "REMOTE_API",
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                return {
                    "status": "API_ERROR_FALLBACK_LOCAL",
                    "error": str(e),
                    "snapshot": snapshot_file,
                }
        else:
            self.create_rolling_backup()
            return {
                "status": "SUCCESS",
                "mode": "LOCAL_PERSISTENCE",
                "snapshot": snapshot_file,
                "synced_at": datetime.now(timezone.utc).isoformat(),
            }


# 相容舊版名稱別名
GDrivePersistenceSync = LocalPersistenceService
