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

ETH_USDT_BINANCE_TESTNET = InstrumentSpec(
    canonical_id="BINANCE_TESTNET:ETHUSDT:PERPETUAL",
    display_symbol="ETH/USDT",
    venue_symbol="ETH/USDT:USDT",
    venue="BINANCE_TESTNET",
    market_type="PERPETUAL",
    base_asset="ETH",
    quote_asset="USDT",
    settlement_asset="USDT",
    contract_size=Decimal("1"),
    price_tick=Decimal("0.01"),
    quantity_step=Decimal("0.001"),
    min_quantity=Decimal("0.001"),
    min_notional=Decimal("5"),
)

SOL_USDT_BINANCE_TESTNET = InstrumentSpec(
    canonical_id="BINANCE_TESTNET:SOLUSDT:PERPETUAL",
    display_symbol="SOL/USDT",
    venue_symbol="SOL/USDT:USDT",
    venue="BINANCE_TESTNET",
    market_type="PERPETUAL",
    base_asset="SOL",
    quote_asset="USDT",
    settlement_asset="USDT",
    contract_size=Decimal("1"),
    price_tick=Decimal("0.01"),
    quantity_step=Decimal("0.01"),
    min_quantity=Decimal("0.01"),
    min_notional=Decimal("5"),
)

BNB_USDT_BINANCE_TESTNET = InstrumentSpec(
    canonical_id="BINANCE_TESTNET:BNBUSDT:PERPETUAL",
    display_symbol="BNB/USDT",
    venue_symbol="BNB/USDT:USDT",
    venue="BINANCE_TESTNET",
    market_type="PERPETUAL",
    base_asset="BNB",
    quote_asset="USDT",
    settlement_asset="USDT",
    contract_size=Decimal("1"),
    price_tick=Decimal("0.01"),
    quantity_step=Decimal("0.01"),
    min_quantity=Decimal("0.01"),
    min_notional=Decimal("5"),
)

INSTRUMENT_SPECS: dict[str, InstrumentSpec] = {
    "BTC/USDT": BTC_USDT_BINANCE_TESTNET,
    "BTCUSDT": BTC_USDT_BINANCE_TESTNET,
    "ETH/USDT": ETH_USDT_BINANCE_TESTNET,
    "ETHUSDT": ETH_USDT_BINANCE_TESTNET,
    "SOL/USDT": SOL_USDT_BINANCE_TESTNET,
    "SOLUSDT": SOL_USDT_BINANCE_TESTNET,
    "BNB/USDT": BNB_USDT_BINANCE_TESTNET,
    "BNBUSDT": BNB_USDT_BINANCE_TESTNET,
}


def get_instrument_spec(symbol: str, venue: str = "BINANCE_TESTNET") -> InstrumentSpec:
    """Resolve an InstrumentSpec by display or venue symbol, with safe generic fallback."""
    key = (symbol or "BTC/USDT").strip().upper().replace(":USDT", "").replace(":USDC", "")
    if key in INSTRUMENT_SPECS:
        return INSTRUMENT_SPECS[key]
    norm = key.replace("_", "/")
    if "/" not in norm and norm.endswith("USDT") and len(norm) > 4:
        norm = f"{norm[:-4]}/USDT"
    elif "/" not in norm and norm.endswith("USDC") and len(norm) > 4:
        norm = f"{norm[:-4]}/USDC"
    if norm in INSTRUMENT_SPECS:
        return INSTRUMENT_SPECS[norm]
    base, quote = (norm.split("/", 1) + ["USDT"])[:2]
    return InstrumentSpec(
        canonical_id=f"{venue.upper()}:{base}{quote}:PERPETUAL",
        display_symbol=f"{base}/{quote}",
        venue_symbol=f"{base}/{quote}:{quote}",
        venue=venue.upper(),
        market_type="PERPETUAL",
        base_asset=base,
        quote_asset=quote,
        settlement_asset=quote,
        contract_size=Decimal("1"),
        price_tick=Decimal("0.01"),
        quantity_step=Decimal("0.01"),
        min_quantity=Decimal("0.01"),
        min_notional=Decimal("5"),
    )
