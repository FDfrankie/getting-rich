import os
import tempfile

from crypto_trading.manual_controller import ManualPositionController


class MockExecutionClient:
    def __init__(self):
        self.orders = []

    def create_order(self, symbol, side, order_type, amount, price=None):
        self.orders.append(
            {"symbol": symbol, "side": side, "type": order_type, "amount": amount}
        )
        return {"success": True, "order_id": f"mock_ex_{len(self.orders)}"}


def test_manual_lock_and_ttl_expiration():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        ctrl = ManualPositionController(db_path=db_path)
        # 鎖定 1 秒
        ctrl.lock_symbol("BTC/USDT", reason="manual_hedge", ttl_seconds=1)
        assert ctrl.is_symbol_locked("BTC/USDT") is True

        # 等待過期
        import time

        time.sleep(1.2)
        assert ctrl.is_symbol_locked("BTC/USDT") is False
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_manual_sl_tp_setting():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        ctrl = ManualPositionController(db_path=db_path)
        res = ctrl.set_sl_tp("ETH/USDC", stop_loss=3100.0, take_profit=3500.0)
        assert res["success"] is True
        assert res["stop_loss"] == 3100.0
        assert res["take_profit"] == 3500.0
        assert ctrl.is_symbol_locked("ETH/USDC") is True
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_manual_close_and_reduce_position():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        ctrl = ManualPositionController(db_path=db_path)
        client = MockExecutionClient()

        # 模擬持倉 LONG 0.5 BTC
        curr_pos = {"contracts": 0.5, "side": "LONG"}
        # 1. 部分減倉 0.2
        red_res = ctrl.manual_reduce_position(
            client, "BTC/USDT", curr_pos, reduce_amount=0.2
        )
        assert red_res["success"] is True
        assert len(client.orders) == 1
        assert client.orders[0]["side"] == "SELL"
        assert client.orders[0]["amount"] == 0.2

        # 2. 全平剩餘 0.3
        close_res = ctrl.manual_close_position(
            client, "BTC/USDT", {"contracts": 0.3, "side": "LONG"}
        )
        assert close_res["success"] is True
        assert len(client.orders) == 2
        assert client.orders[1]["side"] == "SELL"
        assert client.orders[1]["amount"] == 0.3
        assert ctrl.is_symbol_locked("BTC/USDT") is True
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
