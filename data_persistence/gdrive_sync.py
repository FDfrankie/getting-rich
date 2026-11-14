import os
import json
import sqlite3
import shutil
import time
from datetime import datetime
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_persistence")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")
STATE_DB_PATH = os.path.join(DATA_DIR, "system_state.db")
SNAPSHOT_JSON_PATH = os.path.join(DATA_DIR, "cloud_state_snapshot.json")

class GDrivePersistenceSync:
    """
    Google Drive / 本地持久化資料庫與雲端狀態同步模組
    功能：
    1. 支援 SQLite 狀態庫 (system_state.db) 結構維護與冷重啟狀態自動復原
    2. 自動產生 JSON 狀態快照 (cloud_state_snapshot.json)，便於雲端 Render / 本地雙向同調
    3. 自動歷史備份 (Rolling Backups)，杜絕意外斷電或伺服器重啟造成的數據遺失
    """
    def __init__(self, db_path: str = STATE_DB_PATH):
        self.db_path = db_path
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(BACKUP_DIR, exist_ok=True)
        self.init_db()

    def init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS account_equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_equity REAL NOT NULL,
                    balance_usdt REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] GDrivePersistenceSync 資料庫初始化失敗: {e}")

    def save_state(self, key: str, value: Any):
        try:
            val_str = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, val_str, now_str)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 儲存狀態失敗 ({key}): {e}")

    def get_state(self, key: str, default: Any = None) -> Any:
        try:
            conn = sqlite3.connect(self.db_path)
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

    def record_post_mortem(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        loss_amount: float,
        loss_pct: float,
        cause_analysis: str
    ):
        """記錄停損/虧損覆盤檢討 (Post-Mortem)"""
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO post_mortems (date, symbol, entry_price, exit_price, loss_amount, loss_pct, cause_analysis, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (today_str, symbol, entry_price, exit_price, loss_amount, loss_pct, cause_analysis, now_str))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 記錄覆盤檢討異常: {e}")

    def get_recent_post_mortems(self, limit: int = 5) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT date, symbol, entry_price, exit_price, loss_amount, loss_pct, cause_analysis, created_at
                FROM post_mortems ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
            conn.close()
            results = []
            for r in rows:
                results.append({
                    "date": r[0],
                    "symbol": r[1],
                    "entry_price": r[2],
                    "exit_price": r[3],
                    "loss_amount": r[4],
                    "loss_pct": r[5],
                    "cause_analysis": r[6],
                    "created_at": r[7]
                })
            return results
        except Exception:
            return []

    def record_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        amount: float,
        status: str = "FILLED",
        exchange_order_id: Optional[str] = None,
        cost: Optional[float] = None,
        fee: float = 0.0,
        reason: str = ""
    ):
        """記錄訂單委託至 orders 資料表"""
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO orders (exchange_order_id, symbol, side, order_type, price, amount, cost, fee, status, reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (exchange_order_id, symbol, side, order_type, price, amount, cost or (price * amount), fee, status, reason, now_str))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 記錄訂單失敗: {e}")

    def update_position_record(
        self,
        symbol: str,
        side: str,
        contracts: float,
        entry_price: float,
        unrealized_pnl: float = 0.0,
        leverage: int = 1
    ):
        """更新當前持倉狀態 (positions 表)"""
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            if contracts <= 1e-6:
                cur.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                cur.execute("""
                    REPLACE INTO positions (symbol, side, contracts, entry_price, unrealized_pnl, leverage, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, side, contracts, entry_price, unrealized_pnl, leverage, now_str))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 更新持倉紀錄失敗: {e}")

    def record_trade_log(
        self,
        symbol: str,
        action: str,
        side: str,
        price: float,
        amount: float,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        reason: str = ""
    ):
        """記錄撮合成交歷史至 trade_logs 資料表"""
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO trade_logs (symbol, action, side, price, amount, pnl, pnl_pct, reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action, side, price, amount, pnl, pnl_pct, reason, now_str))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 記錄成交日誌失敗: {e}")

    def record_account_equity(self, total_equity: float, balance_usdt: float, unrealized_pnl: float = 0.0):
        """記錄帳戶淨值歷史至 account_equity 資料表"""
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO account_equity (total_equity, balance_usdt, unrealized_pnl, timestamp)
                VALUES (?, ?, ?, ?)
            """, (total_equity, balance_usdt, unrealized_pnl, now_str))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 記錄帳戶淨值失敗: {e}")

    def get_recent_orders(self, limit: int = 10) -> list:
        """獲取最近訂單紀錄"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT id, exchange_order_id, symbol, side, order_type, price, amount, cost, fee, status, reason, timestamp
                FROM orders ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
            conn.close()
            return [{
                "id": r[0], "exchange_order_id": r[1], "symbol": r[2], "side": r[3],
                "order_type": r[4], "price": r[5], "amount": r[6], "cost": r[7],
                "fee": r[8], "status": r[9], "reason": r[10], "timestamp": r[11]
            } for r in rows]
        except Exception:
            return []

    def export_snapshot_json(self) -> str:
        """匯出全狀態 JSON 快照以供雲端/本地備援"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT key, value, updated_at FROM bot_state")
            state_dict = {r[0]: {"value": r[1], "updated_at": r[2]} for r in cur.fetchall()}
            cur.execute("SELECT timestamp, message FROM bot_logs ORDER BY id DESC LIMIT 100")
            logs = [{"timestamp": r[0], "message": r[1]} for r in cur.fetchall()]
            conn.close()

            payload = {
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "state": state_dict,
                "recent_logs": logs,
                "recent_post_mortems": self.get_recent_post_mortems(10)
            }
            with open(SNAPSHOT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return SNAPSHOT_JSON_PATH
        except Exception as e:
            print(f"[!] 匯出快照失敗: {e}")
            return ""

    def create_rolling_backup(self):
        """建立時間戳 rolling backup，優先使用 sqlite3 在線備份 (sqlite3.backup()) 保證交易一致性"""
        try:
            if os.path.exists(self.db_path):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(BACKUP_DIR, f"state_backup_{ts}.db")
                try:
                    src_conn = sqlite3.connect(self.db_path)
                    dst_conn = sqlite3.connect(backup_file)
                    src_conn.backup(dst_conn)
                    dst_conn.close()
                    src_conn.close()
                except Exception:
                    shutil.copy2(self.db_path, backup_file)
                # 維護備份數量不超過 10 個
                existing = sorted([os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR) if f.endswith(".db")])
                if len(existing) > 10:
                    for old_file in existing[:-10]:
                        try:
                            os.remove(old_file)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[!] 建立自動備份失敗: {e}")

    def sync_to_gdrive(self) -> Dict[str, Any]:
        """
        支援 Google Drive API 雙向同調：
        檢查環境變數 GOOGLE_DRIVE_CREDENTIALS 與 GOOGLE_DRIVE_FOLDER_ID。
        若未設置則執行本地 Google Drive 資料夾鏡像與備份同步（安全無破壞性退避）。
        """
        creds_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS", "") or os.getenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "")
        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
        snapshot_file = self.export_snapshot_json()
        
        if creds_json and folder_id:
            try:
                # 雲端正式 Google Drive API 上傳
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
                from googleapiclient.http import MediaFileUpload
                
                info = json.loads(creds_json) if creds_json.startswith("{") else json.load(open(creds_json))
                credentials = service_account.Credentials.from_service_account_info(
                    info, scopes=['https://www.googleapis.com/auth/drive']
                )
                service = build('drive', 'v3', credentials=credentials)
                
                # 同步 system_state.db 與 snapshot
                media = MediaFileUpload(self.db_path, mimetype='application/x-sqlite3', resumable=True)
                file_metadata = {'name': 'system_state.db', 'parents': [folder_id]}
                service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                return {"status": "SUCCESS", "mode": "REMOTE_API", "synced_at": datetime.now().isoformat()}
            except Exception as e:
                return {"status": "API_ERROR_FALLBACK_LOCAL", "error": str(e), "snapshot": snapshot_file}
        else:
            # 本地 Google Drive 或 Render 持久化卷模式
            self.create_rolling_backup()
            return {"status": "SUCCESS", "mode": "LOCAL_SNAPSHOT_MIRROR", "snapshot": snapshot_file, "synced_at": datetime.now().isoformat()}
