"""Dedicated MarketDataService, PaperBroker, and sign-safe PnL calculation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from crypto_trading.trading_snapshot import (
    AccountSnapshot,
    PositionSnapshot,
    ReconciliationSnapshot,
)


def calculate_pnl(side: str, quantity: Decimal | str | float, entry_price: Decimal | str | float, mark_price: Decimal | str | float) -> Decimal:
    """Sign-safe PnL: always uses abs(quantity) so negative short quantities never invert profit."""
    abs_qty = abs(Decimal(str(quantity)))
    entry_d = Decimal(str(entry_price))
    mark_d = Decimal(str(mark_price))
    norm_side = str(side).upper()
    if norm_side == "LONG":
        return (mark_d - entry_d) * abs_qty
    if norm_side == "SHORT":
        return (entry_d - mark_d) * abs_qty
    return Decimal("0")


@dataclass(frozen=True, slots=True)
class MarketTick:
    symbol: str
    price: Decimal | None
    source: str
    venue: str
    event_time: datetime | None
    receive_time: datetime
    timestamp_source: str


class MarketDataService:
    """Fetches public market ticks without coupling Local Paper to authenticated Testnet account state."""

    def __init__(self, public_adapter: Any | None = None) -> None:
        self.public_adapter = public_adapter

    def get_latest_tick(self, symbol: str, now: datetime | None = None) -> MarketTick:
        recv_now = now or datetime.now(timezone.utc)
        if self.public_adapter is None:
            return MarketTick(symbol, None, "UNAVAILABLE", "NONE", None, recv_now, "UNAVAILABLE")
        try:
            is_mock = bool(getattr(self.public_adapter, "mock_mode", False))
            if is_mock:
                return MarketTick(symbol, None, "UNAVAILABLE", "MOCK_REJECTED", None, recv_now, "UNAVAILABLE")
            raw = self.public_adapter.fetch_ticker(symbol)
            last_val = raw.get("last") if isinstance(raw, dict) else None
            if last_val is None or float(last_val) <= 0:
                return MarketTick(symbol, None, "UNAVAILABLE", "BINANCE", None, recv_now, "UNAVAILABLE")
            ts_ms = raw.get("timestamp")
            if ts_ms is not None:
                ev_time = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
                ts_src = "EXCHANGE_EVENT"
            else:
                ev_time = recv_now
                ts_src = "RECEIVE_TIME"
            return MarketTick(
                symbol=symbol,
                price=Decimal(str(last_val)),
                source="EXCHANGE_PUBLIC_REST",
                venue="BINANCE",
                event_time=ev_time,
                receive_time=recv_now,
                timestamp_source=ts_src,
            )
        except Exception:
            return MarketTick(symbol, None, "UNAVAILABLE", "BINANCE", None, recv_now, "UNAVAILABLE")


class PaperBroker:
    """Provides truthful positions, open orders, mark-to-market account, and reconciliation for LOCAL_PAPER."""

    def __init__(self, bot: Any) -> None:
        self.bot = bot
        self._last_reconciled_at: datetime | None = None

    def fetch_positions(self, mark_price: Decimal | None, now: datetime | None = None) -> tuple[PositionSnapshot, ...]:
        ts = now or datetime.now(timezone.utc)
        raw_qty = Decimal(str(getattr(self.bot, "position", 0.0) or 0.0))
        if raw_qty == 0:
            return ()
        entry_raw = Decimal(str(getattr(self.bot, "entry_price", 0.0) or 0.0))
        entry_p = entry_raw if entry_raw > 0 else None
        side = str(getattr(self.bot, "position_side", "LONG" if raw_qty > 0 else "SHORT")).upper()
        if side not in {"LONG", "SHORT"}:
            side = "LONG" if raw_qty > 0 else "SHORT"
        upnl = calculate_pnl(side, raw_qty, entry_p, mark_price) if (entry_p is not None and mark_price is not None) else None
        symbol = str(getattr(self.bot, "symbol", "BTC/USDT"))
        return (
            PositionSnapshot(
                canonical_symbol=symbol,
                venue_symbol=symbol,
                side=side,
                position_mode="ONE_WAY",
                quantity=abs(raw_qty),
                contract_size=Decimal("1"),
                entry_price=entry_p,
                mark_price=mark_price,
                leverage=Decimal("1"),
                liquidation_price=None,
                unrealized_pnl=upnl,
                realized_pnl=Decimal("0"),
                fees=Decimal("0"),
                funding=Decimal("0"),
                source="LOCAL_PAPER_BROKER",
                timestamp=ts,
            ),
        )

    def fetch_open_orders(self) -> tuple[dict[str, Any], ...]:
        orders = getattr(self.bot, "open_orders", ()) or ()
        return tuple(orders)

    def account_snapshot(self, positions: tuple[PositionSnapshot, ...], now: datetime | None = None) -> AccountSnapshot:
        ts = now or datetime.now(timezone.utc)
        wallet = Decimal(str(getattr(self.bot, "balance_usdt", 10000.0)))
        if any(p.unrealized_pnl is None for p in positions):
            return AccountSnapshot(wallet, None, None, wallet, None, None, None, "USDT", source="LOCAL_PAPER_BROKER", timestamp=ts)
        upnl = sum((p.unrealized_pnl for p in positions if p.unrealized_pnl is not None), Decimal("0"))
        eq = wallet + upnl
        return AccountSnapshot(wallet, upnl, eq, wallet, None, None, None, "USDT", source="LOCAL_PAPER_BROKER", timestamp=ts)

    def reconcile(self, now: datetime | None = None) -> ReconciliationSnapshot:
        ts = now or datetime.now(timezone.utc)
        self._last_reconciled_at = ts
        return ReconciliationSnapshot("OK", 0, ts, ())
