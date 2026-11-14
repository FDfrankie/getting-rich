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
                    updated_at TEXT NOT NULL
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
        """建立時間戳 rolling backup"""
        try:
            if os.path.exists(self.db_path):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(BACKUP_DIR, f"state_backup_{ts}.db")
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
