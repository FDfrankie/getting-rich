from datetime import datetime
from typing import Any

from crypto_trading.order_manager import OrderManager, OrderState


class ReconciliationService:
    """
    對帳生命週期服務
    比對 orders、fills、positions、balances、partial fills、pending cancels。
    Restart 後先執行對帳，對帳未完成前只允許 reduce-only 操作。
    """

    def __init__(self, order_manager: OrderManager | None = None):
        self.order_manager = order_manager or OrderManager()
        self.is_reconciled = False
        self.last_reconciled_time: str | None = None
        self.reconciliation_report: dict[str, Any] = {}

    def is_ready_for_trading(self) -> bool:
        """重啟後是否已完成對帳"""
        return self.is_reconciled

    def reconcile(
        self,
        local_positions: dict[str, float],
        remote_positions: list[dict[str, Any]],
        local_orders: list[dict[str, Any]],
        remote_orders: list[dict[str, Any]],
        local_balance: float,
        remote_balance: float,
    ) -> dict[str, Any]:
        """
        執行全方位對帳：
        1. 持倉對帳 (Positions)
        2. 未完成訂單對帳 (Open/Pending Orders, Partial fills)
        3. 資產餘額對帳 (Balances)
        """
        discrepancies = []

        # 1. 持倉對帳
        remote_pos_map = {}
        for rp in remote_positions:
            sym = rp.get("symbol", "")
            amt = float(rp.get("contracts", 0.0))
            side = rp.get("side", "LONG").upper()
            signed_amt = amt if side == "LONG" else -amt
            remote_pos_map[sym] = signed_amt

        all_symbols = set(local_positions.keys()).union(set(remote_pos_map.keys()))
        pos_diffs = {}
        for s in all_symbols:
            loc = local_positions.get(s, 0.0)
            rem = remote_pos_map.get(s, 0.0)
            diff = round(abs(loc - rem), 6)
            if diff > 1e-4:
                pos_diffs[s] = {"local": loc, "remote": rem, "discrepancy": diff}
                discrepancies.append(
                    f"Position mismatch on {s}: local={loc}, remote={rem}"
                )

        # 2. 訂單狀態與未結清掛單比對
        remote_order_ids = {
            str(ro.get("id") or ro.get("orderId")): ro for ro in remote_orders
        }
        order_diffs = []
        for lo in local_orders:
            ex_id = str(lo.get("exchange_order_id") or "")
            state = lo.get("state")
            if state in [OrderState.SUBMITTING.value, OrderState.UNKNOWN.value]:
                if ex_id and ex_id in remote_order_ids:
                    # 遠端已確認訂單
                    ro = remote_order_ids[ex_id]
                    order_diffs.append(
                        {
                            "client_order_id": lo.get("client_order_id"),
                            "resolved_state": ro.get("status"),
                        }
                    )
                else:
                    discrepancies.append(
                        f"In-flight order {lo.get('client_order_id')} not confirmed by exchange"
                    )

        # 3. 餘額比對
        bal_diff = round(abs(local_balance - remote_balance), 2)
        if bal_diff > 1.0:
            discrepancies.append(
                f"Balance mismatch: local={local_balance}, remote={remote_balance}"
            )

        self.is_reconciled = len(discrepancies) == 0
        self.last_reconciled_time = datetime.utcnow().isoformat()
        self.reconciliation_report = {
            "is_reconciled": self.is_reconciled,
            "timestamp": self.last_reconciled_time,
            "discrepancy_count": len(discrepancies),
            "discrepancies": discrepancies,
            "position_diffs": pos_diffs,
            "order_resolutions": order_diffs,
            "balance_discrepancy": bal_diff,
            "allowed_actions": "ALL" if self.is_reconciled else "REDUCE_ONLY",
        }
        return self.reconciliation_report
