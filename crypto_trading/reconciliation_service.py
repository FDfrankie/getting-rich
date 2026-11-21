from datetime import datetime, timezone
from typing import Any

from crypto_trading.order_execution_service import is_exchange_risk_event_order
from crypto_trading.order_manager import OrderManager, OrderState


def _normalize_recon_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper().replace(":USDT", "").replace(":USDC", "").replace("_", "/")
    if "/" not in s and s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/USDT"
    if "/" not in s and s.endswith("USDC") and len(s) > 4:
        return f"{s[:-4]}/USDC"
    return s


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

    def get_latest_result(self) -> dict[str, Any] | None:
        if not self.reconciliation_report:
            return None
        return {
            "in_sync": bool(self.is_reconciled),
            "discrepancy_count": int(self.reconciliation_report.get("discrepancy_count", 0)),
            "completed_at": self.last_reconciled_time,
            "unknown_order_count": int(self.reconciliation_report.get("unknown_order_count", 0)),
            "exchange_risk_event_count": int(self.reconciliation_report.get("exchange_risk_event_count", 0)),
        }

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

        # 1. 持倉對帳 (正規化幣對格式，如 BTCUSDT -> BTC/USDT)
        norm_local_positions: dict[str, float] = {}
        for k, v in (local_positions or {}).items():
            nk = _normalize_recon_symbol(str(k))
            norm_local_positions[nk] = norm_local_positions.get(nk, 0.0) + float(v or 0.0)

        remote_pos_map: dict[str, float] = {}
        for rp in remote_positions:
            sym = _normalize_recon_symbol(str(rp.get("symbol", "")))
            amt = abs(float(rp.get("contracts", 0.0) or 0.0))
            side = str(rp.get("side", "LONG")).upper()
            signed_amt = amt if side in {"LONG", "BUY"} else -amt
            if sym:
                remote_pos_map[sym] = remote_pos_map.get(sym, 0.0) + signed_amt

        all_symbols = set(norm_local_positions.keys()).union(set(remote_pos_map.keys()))
        pos_diffs = {}
        for s in all_symbols:
            loc = norm_local_positions.get(s, 0.0)
            rem = remote_pos_map.get(s, 0.0)
            diff = round(abs(loc - rem), 6)
            if diff > 1e-4:
                pos_diffs[s] = {"local": loc, "remote": rem, "discrepancy": diff}
                discrepancies.append(f"Position mismatch on {s}: local={loc}, remote={rem}")

        # 2. 訂單狀態與未結清掛單比對
        remote_order_ids = {str(ro.get("id") or ro.get("orderId")): ro for ro in remote_orders}
        order_diffs = []
        for lo in local_orders:
            ex_id = str(lo.get("exchange_order_id") or "")
            state = lo.get("state")
            if state in [OrderState.SUBMITTING.value, OrderState.UNKNOWN.value]:
                if ex_id and ex_id in remote_order_ids:
                    ro = remote_order_ids[ex_id]
                    r_status = str(ro.get("status") or "").lower()
                    if self.order_manager is not None and lo.get("client_order_id"):
                        target_st = (
                            OrderState.FILLED if r_status in ("closed", "filled")
                            else (OrderState.CANCELLED if r_status in ("canceled", "cancelled") else OrderState.ACKNOWLEDGED)
                        )
                        try:
                            self.order_manager.update_order_state(str(lo["client_order_id"]), target_st, exchange_order_id=ex_id)
                        except Exception:
                            pass
                    order_diffs.append(
                        {
                            "client_order_id": lo.get("client_order_id"),
                            "resolved_state": ro.get("status"),
                        }
                    )
                else:
                    discrepancies.append(f"In-flight order {lo.get('client_order_id')} not confirmed by exchange")

        # 2b. 偵測交易所強制平倉 / ADL / 結算風險事件單 (Binance autoclose-/adl_autoclose)
        exchange_risk_events: list[dict[str, Any]] = []
        for ro in (remote_orders or []):
            cid = str(ro.get("clientOrderId") or ro.get("client_order_id") or "")
            if is_exchange_risk_event_order(cid):
                sym_ev = _normalize_recon_symbol(str(ro.get("symbol") or ""))
                st_ev = str(ro.get("status") or "UNKNOWN")
                exchange_risk_events.append(
                    {
                        "client_order_id": cid,
                        "exchange_order_id": str(ro.get("id") or ro.get("orderId") or ""),
                        "symbol": sym_ev,
                        "status": st_ev,
                    }
                )
                discrepancies.append(f"Exchange risk event order detected: {cid} ({sym_ev} {st_ev})")

        # 3. 餘額比對
        bal_diff = round(abs(local_balance - remote_balance), 2)
        if bal_diff > 1.0:
            discrepancies.append(f"Balance mismatch: local={local_balance}, remote={remote_balance}")

        self.is_reconciled = len(discrepancies) == 0
        self.last_reconciled_time = datetime.now(timezone.utc).isoformat()
        self.reconciliation_report = {
            "is_reconciled": self.is_reconciled,
            "timestamp": self.last_reconciled_time,
            "discrepancy_count": len(discrepancies),
            "unknown_order_count": sum(1 for lo in local_orders if lo.get("state") in (OrderState.UNKNOWN.value, OrderState.SUBMITTING.value) and str(lo.get("exchange_order_id") or "") not in remote_order_ids),
            "discrepancies": discrepancies,
            "position_diffs": pos_diffs,
            "order_resolutions": order_diffs,
            "exchange_risk_events": exchange_risk_events,
            "exchange_risk_event_count": len(exchange_risk_events),
            "balance_discrepancy": bal_diff,
            "allowed_actions": "ALL" if self.is_reconciled else "REDUCE_ONLY",
        }
        return self.reconciliation_report
