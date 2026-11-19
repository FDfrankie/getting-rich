"""Multi-symbol bot manager with PortfolioRiskAllocator integration."""
from __future__ import annotations

import os
from typing import Any, Callable

from crypto_trading.bot_runner import CryptoTradingBot
from crypto_trading.instruments import get_instrument_spec
from crypto_trading.strategies.momentum_breakout import BaseStrategy
from portfolio.risk_allocator import PortfolioAllocationDecision, PortfolioRiskAllocator


DEFAULT_CRYPTO_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT")


def parse_configured_symbols(raw: str | None = None) -> list[str]:
    text = raw if raw is not None else os.getenv("CRYPTO_BOT_SYMBOLS", "")
    if not text.strip():
        return list(DEFAULT_CRYPTO_SYMBOLS)
    seen: set[str] = set()
    out: list[str] = []
    for part in text.split(","):
        sym = part.strip().upper()
        if not sym:
            continue
        if "/" not in sym and sym.endswith("USDT") and len(sym) > 4:
            sym = f"{sym[:-4]}/USDT"
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out or list(DEFAULT_CRYPTO_SYMBOLS)


def attach_bot_safety_context(bot: CryptoTradingBot, reconciliation_service: Any | None = None) -> CryptoTradingBot:
    """Ensure reconciliation_service and per-symbol InstrumentSpec are attached to a CryptoTradingBot."""
    if reconciliation_service is not None:
        bot.reconciliation_service = reconciliation_service
    spec = get_instrument_spec(getattr(bot, "symbol", "BTC/USDT"))
    bot.instrument_spec = spec
    return bot


class MultiSymbolBotManager:
    """Coordinates independent CryptoTradingBot instances with portfolio-level risk allocation."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        *,
        strategy_factory: Callable[[str], BaseStrategy] | None = None,
        exchange_adapter: Any | None = None,
        reconciliation_service: Any | None = None,
        exec_timeframe: str | None = None,
        portfolio_allocator: PortfolioRiskAllocator | None = None,
    ) -> None:
        self.symbols = parse_configured_symbols(",".join(symbols) if symbols else None)
        self.exchange_adapter = exchange_adapter
        self.reconciliation_service = reconciliation_service
        self.portfolio_allocator = portfolio_allocator or PortfolioRiskAllocator()
        self.bots: dict[str, CryptoTradingBot] = {}
        for sym in self.symbols:
            strat = strategy_factory(sym) if callable(strategy_factory) else None
            bot = CryptoTradingBot(
                symbol=sym,
                strategy=strat,
                exchange_adapter=exchange_adapter,
                exec_timeframe=exec_timeframe,
            )
            attach_bot_safety_context(bot, reconciliation_service)
            self.bots[sym] = bot

    @property
    def primary_symbol(self) -> str:
        return self.symbols[0]

    @property
    def primary_bot(self) -> CryptoTradingBot:
        return self.bots[self.primary_symbol]

    def get_bot(self, symbol: str) -> CryptoTradingBot | None:
        norm = symbol.strip().upper()
        if norm in self.bots:
            return self.bots[norm]
        spec = get_instrument_spec(norm)
        return self.bots.get(spec.display_symbol)

    def active_positions_notional(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for sym, bot in self.bots.items():
            pos = float(getattr(bot, "position", 0.0) or 0.0)
            entry = float(getattr(bot, "entry_price", 0.0) or 0.0)
            if abs(pos) > 0 and entry > 0:
                out[sym] = abs(pos * entry)
        return out

    def evaluate_portfolio_gate(
        self,
        symbol: str,
        requested_notional_usdt: float,
        *,
        is_reduce_only: bool = False,
    ) -> PortfolioAllocationDecision:
        total_eq = sum(float(getattr(b, "balance_usdt", 0.0) or 0.0) for b in self.bots.values()) / max(1, len(self.bots))
        self.portfolio_allocator.total_equity_usdt = max(self.portfolio_allocator.total_equity_usdt, total_eq)
        return self.portfolio_allocator.evaluate_allocation(
            symbol=symbol,
            requested_notional_usdt=requested_notional_usdt,
            active_positions_notional=self.active_positions_notional(),
            is_reduce_only=is_reduce_only,
        )

    def set_exchange_adapter(self, adapter: Any | None) -> None:
        self.exchange_adapter = adapter
        for bot in self.bots.values():
            bot.exchange_adapter = adapter

    def switch_strategy_all(self, strategy_factory: Callable[[str], BaseStrategy]) -> None:
        for sym, bot in self.bots.items():
            bot.strategy = strategy_factory(sym)

    def step_all(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for sym, bot in self.bots.items():
            try:
                is_flat = getattr(bot, "position_side", "FLAT") == "FLAT"
                if is_flat:
                    gate = self.evaluate_portfolio_gate(sym, requested_notional_usdt=1.0, is_reduce_only=False)
                    if not gate.allowed:
                        results[sym] = {
                            "ok": True,
                            "skipped_by_portfolio_gate": True,
                            "reason": gate.reason,
                            "symbol": sym,
                            "position": float(bot.position),
                            "position_side": bot.position_side,
                            "balance_usdt": float(bot.balance_usdt),
                        }
                        continue
                bot.step()
                results[sym] = {
                    "ok": True,
                    "symbol": sym,
                    "position": float(bot.position),
                    "position_side": bot.position_side,
                    "balance_usdt": float(bot.balance_usdt),
                }
            except Exception as exc:
                results[sym] = {
                    "ok": False,
                    "symbol": sym,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return results

    def status_summary(self) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for sym, bot in self.bots.items():
            spec = get_instrument_spec(sym)
            summary.append({
                "symbol": sym,
                "venue_symbol": spec.venue_symbol,
                "quantity_step": str(spec.quantity_step),
                "price_tick": str(spec.price_tick),
                "position": float(bot.position),
                "position_side": bot.position_side,
                "balance_usdt": float(bot.balance_usdt),
                "strategy": type(bot.strategy).__name__,
            })
        return summary
