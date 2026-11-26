"""Crypto Cost-Aware Decision Filter, Central Rate-Limit Budget, and Timeout-Safe Order FSM.

Implements three P0 execution safeguards:
  1. Cost-Aware No-Trade Barrier (`CostEstimate`, `evaluate_cost_barrier`):
     Separates model forecast from trade decision so gross-positive but
     cost-negative signals are rejected before hitting the exchange.
  2. Timeout-Safe Order FSM (`OrderState`, `OrderStateMachine`):
     Enforces `UNKNOWN -> RECONCILING` on `-1007 TIMEOUT` / network timeout,
     strictly preventing duplicate `create_order` retries.
  3. Central Rate-Limit Budget (`CentralRateLimitBudget`):
     Shared per-exchange weight tracker honoring HTTP 429 Retry-After and
     HTTP 418 IP-ban windows across all workers.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# ---------------------------------------------------------------------------
# 1. Unified Cost Model & No-Trade Barrier
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostEstimate:
    gross_edge: float
    explicit_fee: float
    spread_cost: float
    slippage: float
    market_impact: float
    funding: float
    borrow: float
    uncertainty_buffer: float

    @property
    def estimated_total_cost(self) -> float:
        return (
            abs(self.explicit_fee)
            + abs(self.spread_cost)
            + abs(self.slippage)
            + abs(self.market_impact)
            + max(0.0, self.funding)
            + abs(self.borrow)
        )

    @property
    def net_edge(self) -> float:
        return self.gross_edge - self.estimated_total_cost - abs(self.uncertainty_buffer)


@dataclass(frozen=True)
class TradeGateDecision:
    trade_allowed: bool
    cost_estimate: CostEstimate
    minimum_edge_multiple: float
    required_net_edge: float
    reason: str


def evaluate_cost_barrier(
    *,
    expected_return: float,
    explicit_fee: float = 0.0012,
    spread_cost: float = 0.0001,
    slippage: float = 0.0002,
    market_impact: float = 0.0,
    funding: float = 0.0001,
    borrow: float = 0.0,
    uncertainty_buffer: float = 0.0002,
    minimum_edge_multiple: float = 1.25,
) -> TradeGateDecision:
    """Only allow trade if expected_net_edge exceeds `minimum_edge_multiple * total_cost`."""
    est = CostEstimate(
        gross_edge=float(expected_return),
        explicit_fee=float(explicit_fee),
        spread_cost=float(spread_cost),
        slippage=float(slippage),
        market_impact=float(market_impact),
        funding=float(funding),
        borrow=float(borrow),
        uncertainty_buffer=float(uncertainty_buffer),
    )
    req = float(minimum_edge_multiple) * est.estimated_total_cost
    allowed = est.net_edge > req
    reason = "EDGE_CLEARS_COST_BARRIER" if allowed else f"NO_TRADE_COST_BARRIER(net={est.net_edge:.5f}<=req={req:.5f})"
    return TradeGateDecision(
        trade_allowed=allowed,
        cost_estimate=est,
        minimum_edge_multiple=float(minimum_edge_multiple),
        required_net_edge=req,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 2. Timeout-Safe Order State Machine (UNKNOWN / RECONCILING)
# ---------------------------------------------------------------------------
class OrderState(StrEnum):
    CREATED = "CREATED"
    SENT = "SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"
    RECONCILING = "RECONCILING"


_ALLOWED_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset({OrderState.SENT, OrderState.REJECTED, OrderState.CANCELLED}),
    OrderState.SENT: frozenset({OrderState.ACKNOWLEDGED, OrderState.FILLED, OrderState.REJECTED, OrderState.UNKNOWN}),
    OrderState.ACKNOWLEDGED: frozenset({OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCEL_PENDING, OrderState.CANCELLED, OrderState.UNKNOWN}),
    OrderState.PARTIALLY_FILLED: frozenset({OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCEL_PENDING, OrderState.CANCELLED, OrderState.UNKNOWN}),
    OrderState.CANCEL_PENDING: frozenset({OrderState.CANCELLED, OrderState.FILLED, OrderState.UNKNOWN}),
    OrderState.UNKNOWN: frozenset({OrderState.RECONCILING}),
    OrderState.RECONCILING: frozenset({OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED}),
    OrderState.EXPIRED: frozenset(),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
}


@dataclass
class OrderRecord:
    client_order_id: str
    symbol: str
    side: str
    qty: float
    state: OrderState = OrderState.CREATED
    exchange_order_id: str | None = None
    history: list[str] = field(default_factory=list)

    def transition(self, target: OrderState, note: str = "") -> None:
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"Illegal order transition {self.state.value} -> {target.value} ({note})")
        self.history.append(f"{self.state.value}->{target.value}:{note}")
        self.state = target

    def on_timeout(self, error_code: str = "-1007") -> None:
        """Handle Binance -1007 TIMEOUT / network timeout safely without retrying create_order."""
        self.transition(OrderState.UNKNOWN, f"timeout:{error_code}")

    def begin_reconciliation(self) -> None:
        self.transition(OrderState.RECONCILING, "query_order_or_user_stream")

    @property
    def can_create_replacement_order(self) -> bool:
        """Never allow creating a replacement order while original is UNKNOWN or RECONCILING."""
        return self.state in (OrderState.REJECTED, OrderState.CANCELLED)


# ---------------------------------------------------------------------------
# 3. Central Rate-Limit Budget (Shared Across Workers)
# ---------------------------------------------------------------------------
class CentralRateLimitBudget:
    """Thread-safe shared API weight & 429/418 backoff budget per exchange."""

    def __init__(self, exchange: str, weight_limit_per_min: int = 1200) -> None:
        self.exchange = exchange.upper()
        self.weight_limit = max(1, int(weight_limit_per_min))
        self._lock = threading.Lock()
        self.used_weight: int = 0
        self.retry_after_until: float = 0.0
        self.banned_until: float = 0.0

    def can_spend(self, weight: int = 1, *, now: float | None = None) -> tuple[bool, float]:
        t = time.monotonic() if now is None else float(now)
        with self._lock:
            if t < self.banned_until:
                return False, self.banned_until - t
            if t < self.retry_after_until:
                return False, self.retry_after_until - t
            if self.used_weight + weight > self.weight_limit:
                return False, 5.0
            return True, 0.0

    def record_response(
        self,
        *,
        status_code: int,
        used_weight_header: int | None = None,
        retry_after_seconds: float | None = None,
        now: float | None = None,
    ) -> None:
        t = time.monotonic() if now is None else float(now)
        with self._lock:
            if used_weight_header is not None:
                self.used_weight = int(used_weight_header)
            if status_code == 429:
                wait = max(1.0, float(retry_after_seconds or 30.0))
                self.retry_after_until = max(self.retry_after_until, t + wait)
            elif status_code == 418:
                wait = max(120.0, float(retry_after_seconds or 300.0))
                self.banned_until = max(self.banned_until, t + wait)


BITUNIX_STATUS_MAP: dict[str, OrderState] = {
    "INIT": OrderState.CREATED,
    "NEW": OrderState.ACKNOWLEDGED,
    "PART_FILLED": OrderState.PARTIALLY_FILLED,
    "PART_FILLED_CANCELED": OrderState.CANCELLED,
    "CANCELED": OrderState.CANCELLED,
    "FILLED": OrderState.FILLED,
    "EXPIRED": OrderState.EXPIRED,
}


def map_bitunix_order_status(raw_status: str) -> OrderState:
    return BITUNIX_STATUS_MAP.get(str(raw_status or "").upper().strip(), OrderState.UNKNOWN)


# ---------------------------------------------------------------------------
# 4. Binance USDT-M Futures Fee-Aware Barrier Helpers (P50 Upgrade)
# ---------------------------------------------------------------------------
USDT_M_MAKER_FEE_ONE_WAY = 0.0002       # 2 bps (VIP 0 Maker)
USDT_M_TAKER_FEE_ONE_WAY = 0.0005       # 5 bps (VIP 0 Taker)
USDT_M_BNB_DISCOUNT_MULT = 0.90         # 10% fee discount when paying in BNB


def estimate_usdt_roundtrip_fee(
    *,
    entry_mode: str = "maker",
    exit_mode: str = "maker",
    use_bnb_discount: bool = False,
) -> float:
    """Return total round-trip explicit fee ratio for Binance USDT-M perpetuals."""
    e_fee = USDT_M_MAKER_FEE_ONE_WAY if str(entry_mode).lower() == "maker" else USDT_M_TAKER_FEE_ONE_WAY
    x_fee = USDT_M_MAKER_FEE_ONE_WAY if str(exit_mode).lower() == "maker" else USDT_M_TAKER_FEE_ONE_WAY
    total = e_fee + x_fee
    if use_bnb_discount:
        total *= USDT_M_BNB_DISCOUNT_MULT
    return round(total, 8)


def evaluate_usdt_cost_barrier(
    *,
    expected_return: float,
    entry_mode: str = "maker",
    exit_mode: str = "maker",
    use_bnb_discount: bool = False,
    spread_cost: float | None = None,
    slippage: float | None = None,
    funding: float = 0.0001,
    uncertainty_buffer: float = 0.00015,
    minimum_edge_multiple: float = 1.25,
) -> TradeGateDecision:
    """USDT-M fee-calibrated no-trade barrier (Maker 4 bps RT vs Taker 10 bps RT)."""
    rt_fee = estimate_usdt_roundtrip_fee(
        entry_mode=entry_mode,
        exit_mode=exit_mode,
        use_bnb_discount=use_bnb_discount,
    )
    is_maker_entry = str(entry_mode).lower() == "maker"
    eff_spread = 0.00005 if (spread_cost is None and is_maker_entry) else (0.0001 if spread_cost is None else float(spread_cost))
    eff_slip = 0.00005 if (slippage is None and is_maker_entry) else (0.00025 if slippage is None else float(slippage))
    return evaluate_cost_barrier(
        expected_return=expected_return,
        explicit_fee=rt_fee,
        spread_cost=eff_spread,
        slippage=eff_slip,
        funding=funding,
        uncertainty_buffer=uncertainty_buffer,
        minimum_edge_multiple=minimum_edge_multiple,
    )
