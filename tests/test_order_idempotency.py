import os
import tempfile
import pytest
from crypto_trading.order_manager import OrderManager, OrderState, generate_idempotency_key

class MockFailingClient:
    def __init__(self, mode="success"):
        self.mode = mode

    def create_order(self, symbol, side, order_type, amount, price=None):
        if self.mode == "timeout":
            raise TimeoutError("Exchange gateway timed out")
        elif self.mode == "reject":
            return {"success": False, "message": "Insufficient balance"}
        return {
            "success": True,
            "order": {"id": "ex_999888", "status": "closed", "price": 50000.0}
        }

def test_order_idempotency_key_generation():
    k1 = generate_idempotency_key("BTC/USDT", "BUY", "LIMIT", 0.1, 60000.0, "tag1")
    k2 = generate_idempotency_key("BTC/USDT", "BUY", "LIMIT", 0.1, 60000.0, "tag1")
    k3 = generate_idempotency_key("BTC/USDT", "BUY", "LIMIT", 0.1, 60000.0, "tag2")
    assert k1 == k2
    assert k1 != k3

def test_order_idempotency_duplicate_claim_prevention():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        om = OrderManager(db_path=db_path)
        # 第一次宣稱訂單 -> 成功
        c1 = om.claim_order("BTC/USDT", "BUY", "MARKET", 0.5, client_tag="req_123")
        assert c1["claimed"] is True
        assert c1["state"] == OrderState.SUBMITTING.value

        # 重複送出相同 client_tag 的請求 -> 冪等拒絕
        c2 = om.claim_order("BTC/USDT", "BUY", "MARKET", 0.5, client_tag="req_123")
        assert c2["claimed"] is False
        assert c2["error"] == "DUPLICATE_ORDER_REJECTED"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def test_order_idempotency_timeout_moves_to_unknown_state():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        om = OrderManager(db_path=db_path)
        client = MockFailingClient(mode="timeout")

        # 遇到 timeout，必須轉入 UNKNOWN 狀態，嚴禁重送
        res = om.execute_with_idempotency(client, "ETH/USDC", "SELL", "LIMIT", 1.0, 3000.0, client_tag="order_timeout_test")
        assert res["success"] is False
        assert res["state"] == OrderState.UNKNOWN.value

        # 查詢 DB
        ord_rec = om.get_order(res["client_order_id"])
        assert ord_rec["state"] == OrderState.UNKNOWN.value
        assert "Exchange gateway timed out" in ord_rec["error_message"]
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def test_order_idempotency_successful_fill():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        om = OrderManager(db_path=db_path)
        client = MockFailingClient(mode="success")

        res = om.execute_with_idempotency(client, "SOL/USDT", "BUY", "MARKET", 10.0, client_tag="order_success_test")
        assert res["success"] is True
        assert res["state"] == OrderState.FILLED.value
        assert res["exchange_order_id"] == "ex_999888"

        ord_rec = om.get_order(res["client_order_id"])
        assert ord_rec["state"] == OrderState.FILLED.value
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
