"""Portfolio-level risk allocator across multiple crypto symbols (BTC, ETH, SOL, etc.)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class PortfolioAllocationDecision:
    allowed: bool
    symbol: str
    approved_notional_usdt: float
    reason: str


class PortfolioRiskAllocator:
    """Enforces cross-symbol position count, single-asset cap, and total portfolio exposure cap."""

    def __init__(
        self,
        *,
        total_equity_usdt: float = 10000.0,
        max_open_positions: int = 3,
        max_total_exposure_pct: float = 1.50,
        max_single_symbol_pct: float = 0.50,
    ) -> None:
        self.total_equity_usdt = max(0.0, float(total_equity_usdt))
        self.max_open_positions = max(1, int(max_open_positions))
        self.max_total_exposure_pct = max(0.1, float(max_total_exposure_pct))
        self.max_single_symbol_pct = max(0.05, float(max_single_symbol_pct))

    def update_equity(self, new_equity_usdt: float) -> None:
        val = float(new_equity_usdt)
        if not math.isfinite(val):
            return
        self.total_equity_usdt = max(0.0, val)

    def evaluate_allocation(
        self,
        symbol: str,
        requested_notional_usdt: float,
        active_positions_notional: Mapping[str, float],
        *,
        is_reduce_only: bool = False,
        min_notional_usdt: float = 0.0,
    ) -> PortfolioAllocationDecision:
        if not math.isfinite(float(requested_notional_usdt)):
            return PortfolioAllocationDecision(False, symbol, 0.0, "NON_FINITE_INPUT")
        if is_reduce_only:
            return PortfolioAllocationDecision(
                allowed=True,
                symbol=symbol,
                approved_notional_usdt=max(0.0, float(requested_notional_usdt)),
                reason="REDUCE_ONLY_EXEMPT",
            )

        if self.total_equity_usdt <= 0:
            return PortfolioAllocationDecision(False, symbol, 0.0, "ZERO_EQUITY")

        if not math.isfinite(float(requested_notional_usdt)) or not math.isfinite(self.total_equity_usdt):
            return PortfolioAllocationDecision(False, symbol, 0.0, "NON_FINITE_INPUT")
        req = max(0.0, float(requested_notional_usdt))
        open_syms = {k for k, v in active_positions_notional.items() if abs(float(v)) > 0}
        if symbol not in open_syms and len(open_syms) >= self.max_open_positions:
            return PortfolioAllocationDecision(
                allowed=False,
                symbol=symbol,
                approved_notional_usdt=0.0,
                reason=f"MAX_OPEN_POSITIONS_REACHED:{len(open_syms)}/{self.max_open_positions}",
            )

        current_total = sum(abs(float(v)) for v in active_positions_notional.values())
        current_sym = abs(float(active_positions_notional.get(symbol, 0.0)))
        max_total_cap = self.total_equity_usdt * self.max_total_exposure_pct
        max_sym_cap = self.total_equity_usdt * self.max_single_symbol_pct

        remaining_total = max(0.0, max_total_cap - current_total)
        remaining_sym = max(0.0, max_sym_cap - current_sym)
        approved = min(req, remaining_total, remaining_sym)

        if approved <= 0.0:
            return PortfolioAllocationDecision(
                allowed=False,
                symbol=symbol,
                approved_notional_usdt=0.0,
                reason="EXPOSURE_CAP_EXHAUSTED",
            )
        if min_notional_usdt > 0.0 and approved < float(min_notional_usdt):
            return PortfolioAllocationDecision(
                allowed=False,
                symbol=symbol,
                approved_notional_usdt=0.0,
                reason=f"BELOW_MIN_NOTIONAL:{approved:.2f}<{float(min_notional_usdt):.2f}",
            )

        return PortfolioAllocationDecision(
            allowed=True,
            symbol=symbol,
            approved_notional_usdt=round(approved, 2),
            reason="APPROVED" if approved >= req - 1e-6 else "SCALED_TO_PORTFOLIO_CAP",
        )
