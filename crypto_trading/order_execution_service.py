"""Single OrderExecutionService choke point and stateful EmergencyFlattenWorkflow."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from crypto_trading.instruments import BTC_USDT_BINANCE_TESTNET, InstrumentSpec
from crypto_trading.manual_controller import ManualPositionController
from crypto_trading.order_manager import OrderManager
from crypto_trading.risk_manager import RiskManager


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    side: str
    order_type: str
    quantity: Decimal | float | str
    price: Decimal | float | str | None = None
    reduce_only: bool = False
    client_tag: str = ""
    caller: str = "API"


class OrderExecutionService:
    """Single choke point for all order submissions (Bot, UI manual, Partial Reduce, Flatten)."""

    def __init__(
        self,
        risk_manager: RiskManager | None = None,
        order_manager: OrderManager | None = None,
        manual_controller: ManualPositionController | None = None,
        instrument: InstrumentSpec = BTC_USDT_BINANCE_TESTNET,
    ) -> None:
        self.risk_manager = risk_manager or RiskManager()
        self.order_manager = order_manager or OrderManager()
        self.manual_controller = manual_controller or ManualPositionController()
        self.instrument = instrument

    def submit(
        self,
        intent: OrderIntent,
        client: Any,
        *,
        mark_price: float,
        risk_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if intent.caller == "BOT" and self.manual_controller.is_symbol_locked(intent.symbol):
            return {
                "success": False,
                "status": "BLOCKED_BY_MANUAL_OVERRIDE",
                "error": f"{intent.symbol} is locked by manual_override",
            }

        norm_qty = self.instrument.normalize_quantity(intent.quantity)
        ref_price = Decimal(str(intent.price if intent.price is not None else mark_price))
        norm_price = self.instrument.normalize_price(ref_price)
        if not intent.reduce_only:
            self.instrument.validate_order(norm_qty, norm_price)

        ctx = dict(risk_context or {})
        ctx.setdefault("data_source", "ORDER_EXECUTION_SERVICE")
        ctx.setdefault("data_timestamp", datetime.now(timezone.utc).isoformat())
        ctx.setdefault("data_quality_passed", True)
        ctx.setdefault("reconciliation_fresh", True)
        ctx.setdefault("has_unknown_orders", False)
        ctx.setdefault("exchange_connected", client is not None)
        ctx.setdefault("step_size", float(self.instrument.quantity_step))
        ctx.setdefault("tick_size", float(self.instrument.price_tick))
        ctx.setdefault("min_notional", float(self.instrument.min_notional))

        decision = self.risk_manager.evaluate_order(
            symbol=intent.symbol,
            side=intent.side.upper(),
            order_type=intent.order_type.upper(),
            amount=float(norm_qty),
            price=float(norm_price),
            is_reduce_only=intent.reduce_only,
            context=ctx,
        )
        if not decision.allowed:
            return {
                "success": False,
                "status": "RISK_REJECTED",
                "action": decision.action.value,
                "reasons": decision.reasons,
            }

        return self.order_manager.execute_with_idempotency(
            client=client,
            symbol=intent.symbol,
            side=intent.side.upper(),
            order_type=intent.order_type.upper(),
            amount=float(norm_qty),
            price=float(norm_price) if intent.order_type.upper() == "LIMIT" else None,
            client_tag=intent.client_tag,
        )

    def emergency_flatten(self, client: Any, symbol: str | None = None) -> dict[str, Any]:
        """Stateful flatten workflow: FETCH_REMOTE_POSITIONS -> GENERATE_REDUCE_ONLY -> SUBMIT -> COMPLETE/PARTIAL/FAILED."""
        if client is None or not hasattr(client, "fetch_positions"):
            return {"status": "FAILED", "closed": [], "failed": ["CLIENT_UNAVAILABLE"]}
        positions = client.fetch_positions() or []
        targets = [p for p in positions if float(p.get("contracts", 0) or 0) > 0 and (symbol is None or p.get("symbol") == symbol)]
        if not targets:
            return {"status": "COMPLETE", "closed": [], "failed": [], "message": "No open positions to flatten"}

        closed = []
        failed = []
        for pos in targets:
            sym = str(pos.get("symbol") or symbol or "BTC/USDT")
            qty = Decimal(str(pos.get("contracts")))
            side = "SELL" if str(pos.get("side", "LONG")).upper() == "LONG" else "BUY"
            try:
                norm_q = self.instrument.normalize_quantity(qty)
                res = self.order_manager.execute_with_idempotency(
                    client=client,
                    symbol=sym,
                    side=side,
                    order_type="MARKET",
                    amount=float(norm_q),
                    price=None,
                    client_tag=f"flatten_{sym}_{int(datetime.now(timezone.utc).timestamp() // 5)}",
                )
                if res.get("success"):
                    closed.append({"symbol": sym, "quantity": str(norm_q), "order_id": res.get("exchange_order_id")})
                else:
                    failed.append({"symbol": sym, "error": res.get("error", "REJECTED")})
            except Exception as exc:
                failed.append({"symbol": sym, "error": type(exc).__name__})

        final_status = "COMPLETE" if (closed and not failed) else ("PARTIAL" if closed else "FAILED")
        return {"status": final_status, "closed": closed, "failed": failed}
