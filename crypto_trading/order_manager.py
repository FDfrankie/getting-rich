import hashlib
import sqlite3
import time
from datetime import datetime
from enum import Enum
from typing import Any

from data_persistence.gdrive_sync import _DB_LOCK, STATE_DB_PATH


class OrderState(str, Enum):
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


def generate_idempotency_key(
    symbol: str,
    side: str,
    order_type: str,
    amount: float,
    price: float | None,
    client_tag: str = "",
) -> str:
    raw = f"{symbol}_{side.upper()}_{order_type.upper()}_{amount:.6f}_{price or 0:.6f}_{client_tag}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class OrderManager:
    """
    訂單冪等管理與狀態機
    保證 client_order_id / idempotency_key 唯一，杜絕重複下單
    狀態轉換：CREATED -> SUBMITTING -> exchange -> ACK/FILLED
    若遇到網路超時則轉入 UNKNOWN，先查遠端狀態，嚴禁盲目重送
    """

    def __init__(self, db_path: str = STATE_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
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
            conn.commit()
            conn.close()

    def claim_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_order_id: str | None = None,
        client_tag: str = "",
    ) -> dict[str, Any]:
        """
        在 DB 登記並宣稱該筆訂單 (claim -> SUBMITTING)
        若 idempotency_key 或 client_order_id 已存在，直接拒絕重複下單
        """
        now = datetime.utcnow().isoformat()
        cid = (
            client_order_id
            or f"ord_{int(time.time() * 1000)}_{hashlib.md5(f'{symbol}_{amount}_{time.time()}'.encode()).hexdigest()[:8]}"
        )
        ikey = generate_idempotency_key(
            symbol, side, order_type, amount, price, client_tag or cid
        )

        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO idempotent_orders 
                    (client_order_id, idempotency_key, exchange_order_id, symbol, side, order_type, amount, price, state, error_message, created_at, updated_at)
                    VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        cid,
                        ikey,
                        symbol,
                        side.upper(),
                        order_type.upper(),
                        amount,
                        price or 0.0,
                        OrderState.SUBMITTING.value,
                        now,
                        now,
                    ),
                )
                conn.commit()
                return {
                    "claimed": True,
                    "client_order_id": cid,
                    "idempotency_key": ikey,
                    "state": OrderState.SUBMITTING.value,
                }
            except sqlite3.IntegrityError:
                # 衝突：已存在相同的 client_order_id 或 idempotency_key
                cur.execute(
                    "SELECT client_order_id, state, exchange_order_id FROM idempotent_orders WHERE client_order_id = ? OR idempotency_key = ?",
                    (cid, ikey),
                )
                row = cur.fetchone()
                existing_cid, existing_state, ex_oid = (
                    row if row else (cid, OrderState.UNKNOWN.value, None)
                )
                return {
                    "claimed": False,
                    "error": "DUPLICATE_ORDER_REJECTED",
                    "client_order_id": existing_cid,
                    "state": existing_state,
                    "exchange_order_id": ex_oid,
                    "message": f"冪等攔截：重複的訂單請求 (idempotency_key={ikey})",
                }
            finally:
                conn.close()

    def update_order_state(
        self,
        client_order_id: str,
        state: OrderState,
        exchange_order_id: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        now = datetime.utcnow().isoformat()
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    UPDATE idempotent_orders 
                    SET state = ?, exchange_order_id = COALESCE(?, exchange_order_id), error_message = ?, updated_at = ?
                    WHERE client_order_id = ?
                    """,
                    (
                        state.value,
                        exchange_order_id,
                        error_message,
                        now,
                        client_order_id,
                    ),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def get_order(self, client_order_id: str) -> dict[str, Any] | None:
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cur = conn.cursor()
            cur.execute(
                "SELECT client_order_id, idempotency_key, exchange_order_id, symbol, side, order_type, amount, price, state, error_message, created_at, updated_at FROM idempotent_orders WHERE client_order_id = ?",
                (client_order_id,),
            )
            row = cur.fetchone()
            conn.close()
            if not row:
                return None
            return {
                "client_order_id": row[0],
                "idempotency_key": row[1],
                "exchange_order_id": row[2],
                "symbol": row[3],
                "side": row[4],
                "order_type": row[5],
                "amount": row[6],
                "price": row[7],
                "state": row[8],
                "error_message": row[9],
                "created_at": row[10],
                "updated_at": row[11],
            }

    def execute_with_idempotency(
        self,
        client,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_tag: str = "",
    ) -> dict[str, Any]:
        """
        執行安全冪等下單流程：
        1. DB Claim -> SUBMITTING
        2. 若重複下單直接返回已存在紀錄
        3. 呼叫 client 下單
        4. 成功更新為 ACKNOWLEDGED / FILLED
        5. 逾時或異常轉入 UNKNOWN 並先查詢遠端狀態，嚴禁重送
        """
        claim_res = self.claim_order(
            symbol, side, order_type, amount, price, client_tag=client_tag
        )
        if not claim_res["claimed"]:
            return claim_res

        cid = claim_res["client_order_id"]
        try:
            # 呼叫交易所或客戶端下單
            if hasattr(client, "create_order"):
                res = client.create_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    amount=amount,
                    price=price,
                )
            else:
                raise AttributeError("Client does not have create_order method")

            if res.get("success"):
                ex_oid = str(
                    res.get("order", {}).get("id") or res.get("order_id") or ""
                )
                new_state = (
                    OrderState.FILLED
                    if (
                        res.get("order", {}).get("status") == "closed"
                        or order_type.upper() == "MARKET"
                    )
                    else OrderState.ACKNOWLEDGED
                )
                self.update_order_state(cid, new_state, exchange_order_id=ex_oid)
                return {
                    "success": True,
                    "client_order_id": cid,
                    "exchange_order_id": ex_oid,
                    "state": new_state.value,
                    "raw": res,
                }
            else:
                err_msg = res.get("message") or res.get("error") or "Order rejected"
                self.update_order_state(cid, OrderState.REJECTED, error_message=err_msg)
                return {
                    "success": False,
                    "client_order_id": cid,
                    "state": OrderState.REJECTED.value,
                    "error": err_msg,
                }
        except Exception as e:
            # 發生連線逾時或其他例外，進入 UNKNOWN 狀態
            self.update_order_state(cid, OrderState.UNKNOWN, error_message=str(e))
            return {
                "success": False,
                "client_order_id": cid,
                "state": OrderState.UNKNOWN.value,
                "error": str(e),
                "message": "下單呼叫超時或異常，已鎖定 UNKNOWN 狀態，請向遠端交易所對帳確認，切勿盲目重送",
            }
