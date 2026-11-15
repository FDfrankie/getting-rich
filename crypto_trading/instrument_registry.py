"""
Instrument and InstrumentRegistry module.
Provides canonical instrument identity, standardizing symbols across venues
and avoiding string-based guesswork like '"BTC" in close_df'.
"""

from __future__ import annotations

from dataclasses import dataclass


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
