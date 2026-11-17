"""Strict Binance/CCXT balance and position adapters (never guess missing fields)."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from crypto_trading.trading_snapshot import AccountSnapshot, PositionSnapshot


def _dec_or_none(val: Any) -> Decimal | None:
    if val is None or val == "":
        return None
    try:
        return Decimal(str(val))
    except Exception:
        return None


def normalize_binance_balance(raw: dict[str, Any], default_currency: str = "USDT", now: datetime | None = None) -> AccountSnapshot:
    ts = now or datetime.now(timezone.utc)
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    currency = str(raw.get("settlement_currency") or default_currency)

    wallet = _dec_or_none(info.get("totalWalletBalance"))
    upnl = _dec_or_none(info.get("totalUnrealizedProfit"))
    equity = _dec_or_none(info.get("totalMarginBalance"))
    avail = _dec_or_none(info.get("availableBalance"))
    init_m = _dec_or_none(info.get("totalInitialMargin"))
    maint_m = _dec_or_none(info.get("totalMaintMargin"))

    if equity is None:
        equity = _dec_or_none(raw.get("total_equity"))
    if avail is None:
        avail = _dec_or_none(raw.get("free_balance"))
    if wallet is None and equity is not None and upnl is not None:
        wallet = equity - upnl

    return AccountSnapshot(
        wallet_balance=wallet,
        unrealized_pnl=upnl,
        equity=equity,
        available_balance=avail,
        initial_margin=init_m,
        maintenance_margin=maint_m,
        margin_ratio=None,
        currency=currency,
        source="BINANCE_TESTNET",
        timestamp=ts,
    )


def normalize_remote_positions(
    raw_positions: list[dict[str, Any]] | None,
    canonical_symbol: str,
    source: str,
    now: datetime | None = None,
) -> tuple[PositionSnapshot, ...]:
    ts = now or datetime.now(timezone.utc)
    out: list[PositionSnapshot] = []
    for p in raw_positions or []:
        qty = _dec_or_none(p.get("contracts") if p.get("contracts") is not None else p.get("positionAmt"))
        if qty is None or qty == 0:
            continue
        info = p.get("info") if isinstance(p.get("info"), dict) else {}
        pos_side = p.get("positionSide") or info.get("positionSide")
        position_mode = "HEDGE" if pos_side in {"LONG", "SHORT"} else "ONE_WAY"
        side = str(pos_side or p.get("side") or ("LONG" if qty > 0 else "SHORT")).upper()
        out.append(
            PositionSnapshot(
                canonical_symbol=canonical_symbol,
                venue_symbol=str(p.get("symbol") or canonical_symbol),
                side=side,
                position_mode=position_mode,
                quantity=abs(qty),
                contract_size=_dec_or_none(p.get("contractSize")) or Decimal("1"),
                entry_price=_dec_or_none(p.get("entryPrice") if p.get("entryPrice") is not None else p.get("entry_price")),
                mark_price=_dec_or_none(p.get("markPrice") if p.get("markPrice") is not None else p.get("mark_price")),
                leverage=_dec_or_none(p.get("leverage")),
                liquidation_price=_dec_or_none(p.get("liquidationPrice")),
                unrealized_pnl=_dec_or_none(p.get("unrealizedPnl") if p.get("unrealizedPnl") is not None else p.get("unrealized_pnl")),
                realized_pnl=_dec_or_none(p.get("realizedPnl")),
                fees=_dec_or_none(p.get("fees")),
                funding=_dec_or_none(p.get("funding")),
                source=source,
                timestamp=ts,
            )
        )
    return tuple(out)
