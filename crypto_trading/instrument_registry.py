"""
Instrument and InstrumentRegistry module.
Provides canonical instrument identity, standardizing symbols across venues
and avoiding string-based guesswork like '"BTC" in close_df'.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN


@dataclass(frozen=True)
class Instrument:
    symbol: str  # e.g., "BTC/USDT"
    base_asset: str  # e.g., "BTC"
    quote_asset: str  # e.g., "USDT"
    settlement_asset: str  # e.g., "USDT" or "USDC"
    venue: str = "binance"
    is_perpetual: bool = True
    min_order_qty: float = 0.001
    price_precision: int = 2
    qty_precision: int = 3

    @property
    def canonical_id(self) -> str:
        return f"{self.venue.upper()}:{self.base_asset}/{self.quote_asset}:{self.settlement_asset}"


class InstrumentRegistry:
    """Registry maintaining active and known trading instruments."""

    def __init__(self):
        self._instruments: dict[str, Instrument] = {}
        self._register_defaults()

    def _register_defaults(self):
        defaults = [
            Instrument("BTC/USDT", "BTC", "USDT", "USDT"),
            Instrument("BTC/USDC", "BTC", "USDC", "USDC"),
            Instrument("ETH/USDT", "ETH", "USDT", "USDT"),
            Instrument("ETH/USDC", "ETH", "USDC", "USDC"),
            Instrument("SOL/USDT", "SOL", "USDT", "USDT"),
            Instrument("SOL/USDC", "SOL", "USDC", "USDC"),
            Instrument("BNB/USDT", "BNB", "USDT", "USDT"),
        ]
        for inst in defaults:
            self.register(inst)

    def register(self, instrument: Instrument) -> None:
        self._instruments[instrument.symbol.upper()] = instrument
        # Also index by base_asset if not present
        if instrument.base_asset.upper() not in self._instruments:
            self._instruments[instrument.base_asset.upper()] = instrument

    def get(self, key: str) -> Instrument | None:
        return self._instruments.get(key.upper().strip())

    def resolve_base_asset(self, key: str) -> str:
        inst = self.get(key)
        if inst:
            return inst.base_asset
        # Fallback parsing
        clean = key.split("/")[0].split(":")[0].upper()
        for suffix in ("USDT", "USDC", "USD"):
            if clean.endswith(suffix) and len(clean) > len(suffix):
                return clean[: -len(suffix)]
        return clean

    def has_instrument(self, key: str) -> bool:
        return key.upper().strip() in self._instruments

    def list_symbols(self) -> set[str]:
        return {inst.symbol for inst in self._instruments.values()}


_DEFAULT_REGISTRY: InstrumentRegistry | None = None


def get_instrument_registry() -> InstrumentRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = InstrumentRegistry()
    return _DEFAULT_REGISTRY


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
