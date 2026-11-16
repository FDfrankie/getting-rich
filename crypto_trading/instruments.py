"""Canonical instrument identity and Decimal-safe normalization."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    canonical_id: str
    display_symbol: str
    venue_symbol: str
    venue: str
    market_type: str
    base_asset: str
    quote_asset: str
    settlement_asset: str
    contract_size: Decimal
    price_tick: Decimal
    quantity_step: Decimal
    min_quantity: Decimal
    min_notional: Decimal

    def normalize_price(self, value: Decimal | str | float) -> Decimal:
        price = Decimal(str(value))
        if price <= 0:
            raise ValueError("price must be positive")
        return (price / self.price_tick).to_integral_value(rounding=ROUND_DOWN) * self.price_tick

    def normalize_quantity(self, value: Decimal | str | float) -> Decimal:
        quantity = Decimal(str(value))
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        return (quantity / self.quantity_step).to_integral_value(rounding=ROUND_DOWN) * self.quantity_step

    def validate_order(self, quantity: Decimal, price: Decimal) -> None:
        if quantity < self.min_quantity:
            raise ValueError(f"quantity {quantity} is below min_quantity {self.min_quantity}")
        notional = quantity * price * self.contract_size
        if notional < self.min_notional:
            raise ValueError(f"notional {notional} is below min_notional {self.min_notional}")


BTC_USDT_BINANCE_TESTNET = InstrumentSpec(
    canonical_id="BINANCE_TESTNET:BTCUSDT:PERPETUAL",
    display_symbol="BTC/USDT",
    venue_symbol="BTC/USDT:USDT",
    venue="BINANCE_TESTNET",
    market_type="PERPETUAL",
    base_asset="BTC",
    quote_asset="USDT",
    settlement_asset="USDT",
    contract_size=Decimal("1"),
    price_tick=Decimal("0.1"),
    quantity_step=Decimal("0.001"),
    min_quantity=Decimal("0.001"),
    min_notional=Decimal("5"),
)
