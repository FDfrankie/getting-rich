from crypto_trading.order_manager import OrderState
from crypto_trading.reconciliation_service import ReconciliationService


def test_reconciliation_perfect_match():
    svc = ReconciliationService()
    local_pos = {"BTC/USDT": 0.5, "ETH/USDT": -1.0}
    remote_pos = [
        {"symbol": "BTC/USDT", "side": "LONG", "contracts": 0.5},
        {"symbol": "ETH/USDT", "side": "SHORT", "contracts": 1.0},
    ]
    local_orders = []
    remote_orders = []
    local_bal = 10000.0
    remote_bal = 10000.0

    res = svc.reconcile(
        local_pos, remote_pos, local_orders, remote_orders, local_bal, remote_bal
    )
    assert res["is_reconciled"] is True
    assert res["discrepancy_count"] == 0
    assert res["allowed_actions"] == "ALL"


def test_reconciliation_position_mismatch_locks_reduce_only():
    svc = ReconciliationService()
    local_pos = {"BTC/USDT": 0.5}
    # 遠端倉位已變成 0.0 (可能已被強制平倉或手動干預)
    remote_pos = []
    local_orders = []
    remote_orders = []
    local_bal = 10000.0
    remote_bal = 10000.0

    res = svc.reconcile(
        local_pos, remote_pos, local_orders, remote_orders, local_bal, remote_bal
    )
    assert res["is_reconciled"] is False
    assert res["discrepancy_count"] > 0
    assert res["allowed_actions"] == "REDUCE_ONLY"
    assert "BTC/USDT" in res["position_diffs"]


def test_reconciliation_inflight_order_resolution():
    svc = ReconciliationService()
    local_pos = {}
    remote_pos = []
    local_orders = [
        {
            "client_order_id": "cid_1",
            "exchange_order_id": "ex_1",
            "state": OrderState.SUBMITTING.value,
        }
    ]
    # 遠端交易所回報該訂單已經 FILLED
    remote_orders = [{"orderId": "ex_1", "status": "FILLED"}]
    local_bal = 5000.0
    remote_bal = 5000.0

    res = svc.reconcile(
        local_pos, remote_pos, local_orders, remote_orders, local_bal, remote_bal
    )
    assert len(res["order_resolutions"]) == 1
    assert res["order_resolutions"][0]["resolved_state"] == "FILLED"
