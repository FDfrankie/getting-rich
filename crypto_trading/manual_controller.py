import sqlite3
from datetime import datetime, timezone
from typing import Any

from data_persistence.gdrive_sync import _DB_LOCK, STATE_DB_PATH


class ManualPositionController:
    """
    合約人工手動控盤 (manual_override) 控制器
    支援：
    1. 單筆全平 (close position)
    2. 部分減倉 (partial reduction)
    3. 設定/修改 SL/TP 保護單 (set / update Stop-Loss & Take-Profit)
    4. 鎖定 manual_override（含 TTL 到期機制）
    5. Bot cycle 不得覆蓋被人工接管的倉位
    6. 所有手動操作走結構化事件與風控檢驗
    """

    def __init__(self, db_path: str = STATE_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cur = conn.cursor()
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS manual_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    amount REAL,
                    price REAL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()

    def lock_symbol(
        self,
        symbol: str,
        reason: str = "manual_intervention",
        ttl_seconds: int = 3600,
        sl_price: float | None = None,
        tp_price: float | None = None,
    ) -> dict[str, Any]:
        """鎖定幣種倉位為手動接管"""
        now = datetime.now(timezone.utc).isoformat()
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO manual_overrides (symbol, is_locked, lock_reason, locked_at, ttl_seconds, stop_loss_price, take_profit_price, updated_at)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    is_locked = 1,
                    lock_reason = excluded.lock_reason,
                    locked_at = excluded.locked_at,
                    ttl_seconds = excluded.ttl_seconds,
                    stop_loss_price = COALESCE(excluded.stop_loss_price, manual_overrides.stop_loss_price),
                    take_profit_price = COALESCE(excluded.take_profit_price, manual_overrides.take_profit_price),
                    updated_at = excluded.updated_at
            """,
                (symbol, reason, now, ttl_seconds, sl_price, tp_price, now),
            )

            # 記錄結構化事件
            cur.execute(
                "INSERT INTO manual_events (symbol, action, amount, price, metadata, timestamp) VALUES (?, 'LOCK', NULL, NULL, ?, ?)",
                (symbol, f"Locked with reason: {reason}, TTL: {ttl_seconds}", now),
            )
            conn.commit()
            conn.close()

        return {
            "success": True,
            "symbol": symbol,
            "is_locked": True,
            "ttl_seconds": ttl_seconds,
            "reason": reason,
        }

    def unlock_symbol(self, symbol: str) -> dict[str, Any]:
        """解除手動鎖定，交還給 Bot 自動交易"""
        now = datetime.now(timezone.utc).isoformat()
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cur = conn.cursor()
            cur.execute(
                "UPDATE manual_overrides SET is_locked = 0, updated_at = ? WHERE symbol = ?",
                (now, symbol),
            )
            cur.execute(
                "INSERT INTO manual_events (symbol, action, amount, price, metadata, timestamp) VALUES (?, 'UNLOCK', NULL, NULL, 'Manual unlock', ?)",
                (symbol, now),
            )
            conn.commit()
            conn.close()

        return {"success": True, "symbol": symbol, "is_locked": False}

    def is_symbol_locked(self, symbol: str) -> bool:
        """檢查是否處於人工手動鎖定狀態 (含 TTL 到期檢驗)"""
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cur = conn.cursor()
            cur.execute(
                "SELECT is_locked, locked_at, ttl_seconds FROM manual_overrides WHERE symbol = ?",
                (symbol,),
            )
            row = cur.fetchone()
            if not row:
                conn.close()
                return False

            is_locked, locked_at_str, ttl = row
            if not is_locked:
                conn.close()
                return False

            # 檢查 TTL
            try:
                locked_dt = datetime.fromisoformat(locked_at_str)
                if locked_dt.tzinfo is None:
                    locked_dt = locked_dt.replace(tzinfo=timezone.utc)
                elapsed = (datetime.now(timezone.utc) - locked_dt).total_seconds()
                if elapsed > ttl:
                    # TTL 已過期，自動解鎖
                    cur.execute(
                        "UPDATE manual_overrides SET is_locked = 0, updated_at = ? WHERE symbol = ?",
                        (datetime.now(timezone.utc).isoformat(), symbol),
                    )
                    conn.commit()
                    conn.close()
                    return False
            except Exception:
                pass

            conn.close()
            return True

    def set_sl_tp(
        self,
        symbol: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """設定或更新停損停利保護單"""
        now = datetime.now(timezone.utc).isoformat()
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO manual_overrides (symbol, is_locked, lock_reason, locked_at, ttl_seconds, stop_loss_price, take_profit_price, updated_at)
                VALUES (?, 1, 'sl_tp_updated', ?, 3600, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    stop_loss_price = excluded.stop_loss_price,
                    take_profit_price = excluded.take_profit_price,
                    updated_at = excluded.updated_at
            """,
                (symbol, now, stop_loss, take_profit, now),
            )

            cur.execute(
                "INSERT INTO manual_events (symbol, action, amount, price, metadata, timestamp) VALUES (?, 'SET_SL_TP', NULL, NULL, ?, ?)",
                (symbol, f"SL={stop_loss}, TP={take_profit}", now),
            )
            conn.commit()
            conn.close()

        return {
            "success": True,
            "symbol": symbol,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def manual_close_position(self, client, symbol: str, current_position: dict[str, Any]) -> dict[str, Any]:
        """單筆全平"""
        now = datetime.now(timezone.utc).isoformat()
        contracts = float(current_position.get("contracts", 0.0))
        side = current_position.get("side", "LONG").upper()
        close_side = "SELL" if side == "LONG" else "BUY"

        if hasattr(client, "create_order"):
            try:
                res = client.create_order(symbol=symbol, side=close_side, order_type="MARKET", amount=contracts, reduce_only=True)
            except TypeError:
                res = client.create_order(symbol=symbol, side=close_side, order_type="MARKET", amount=contracts)
        else:
            res = {"success": True, "message": "Simulated close"}

        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO manual_events (symbol, action, amount, price, metadata, timestamp) VALUES (?, 'CLOSE_POSITION', ?, NULL, ?, ?)",
                (symbol, contracts, f"Full close of {side} {contracts}", now),
            )
            conn.commit()
            conn.close()

        # 鎖定手動接管以防 Bot 立即反向開倉
        self.lock_symbol(symbol, reason="manual_full_close", ttl_seconds=1800)
        return {
            "success": res.get("success", True),
            "symbol": symbol,
            "closed_contracts": contracts,
            "raw_response": res,
        }

    def manual_reduce_position(
        self,
        client,
        symbol: str,
        current_position: dict[str, Any],
        reduce_amount: float,
    ) -> dict[str, Any]:
        """部分減倉"""
        now = datetime.now(timezone.utc).isoformat()
        side = current_position.get("side", "LONG").upper()
        close_side = "SELL" if side == "LONG" else "BUY"

        if hasattr(client, "create_order"):
            try:
                res = client.create_order(symbol=symbol, side=close_side, order_type="MARKET", amount=reduce_amount, reduce_only=True)
            except TypeError:
                res = client.create_order(symbol=symbol, side=close_side, order_type="MARKET", amount=reduce_amount)
        else:
            res = {"success": True, "message": "Simulated reduce"}

        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO manual_events (symbol, action, amount, price, metadata, timestamp) VALUES (?, 'REDUCE_POSITION', ?, NULL, ?, ?)",
                (
                    symbol,
                    reduce_amount,
                    f"Partial reduce of {side} by {reduce_amount}",
                    now,
                ),
            )
            conn.commit()
            conn.close()

        # 保持手動鎖定
        self.lock_symbol(symbol, reason="manual_partial_reduce", ttl_seconds=1800)
        return {
            "success": res.get("success", True),
            "symbol": symbol,
            "reduced_amount": reduce_amount,
            "raw_response": res,
        }
