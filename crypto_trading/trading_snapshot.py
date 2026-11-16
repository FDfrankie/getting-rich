"""Atomic trading snapshot models and strict fail-closed consistency checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from crypto_trading.runtime_context import RuntimeTradingContext


class SnapshotStatus(str, Enum):
    HEALTHY = "HEALTHY"
    STALE = "STALE"
    INCONSISTENT = "INCONSISTENT"
    AUTH_FAILED = "AUTH_FAILED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ConnectionSnapshot:
    exchange: str
    public_connected: bool
    authenticated: bool
    latency_ms: int | None
    last_success_at: datetime | None
    error_code: str | None = None
    safe_error: str | None = None


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    canonical_symbol: str
    venue_symbol: str
    instrument_type: str
    settlement_currency: str
    mark_price: Decimal | None
    last_price: Decimal | None
    data_source: str
    timestamp: datetime | None
    stale_seconds: float | None
    timestamp_source: str = "EXCHANGE_EVENT"


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    wallet_balance: Decimal | None
    unrealized_pnl: Decimal | None
    equity: Decimal | None
    available_balance: Decimal | None
    initial_margin: Decimal | None
    maintenance_margin: Decimal | None
    margin_ratio: Decimal | None
    currency: str
    source: str = "UNKNOWN"
    timestamp: datetime | None = None

    def consistency_error(self, tolerance: Decimal = Decimal("0.01")) -> str | None:
        if self.wallet_balance is None or self.unrealized_pnl is None or self.equity is None:
            return "ACCOUNT_FIELDS_MISSING"
        expected = self.wallet_balance + self.unrealized_pnl
        if abs(self.equity - expected) > tolerance:
            return f"EQUITY_MISMATCH expected={expected} actual={self.equity}"
        return None


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    canonical_symbol: str
    venue_symbol: str
    side: str
    position_mode: str
    quantity: Decimal
    contract_size: Decimal
    entry_price: Decimal | None
    mark_price: Decimal | None
    leverage: Decimal | None
    liquidation_price: Decimal | None
    unrealized_pnl: Decimal | None
    realized_pnl: Decimal | None
    fees: Decimal | None
    funding: Decimal | None
    source: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class ReconciliationSnapshot:
    status: str
    discrepancy_count: int
    last_completed_at: datetime | None
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    trading_enabled: bool
    allowed_actions: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TradingSnapshot:
    context: RuntimeTradingContext
    connection: ConnectionSnapshot
    market: MarketSnapshot
    account: AccountSnapshot
    positions: tuple[PositionSnapshot, ...]
    open_orders: tuple[dict[str, Any], ...]
    reconciliation: ReconciliationSnapshot
    risk: RiskSnapshot
    configured_strategy: str
    effective_strategy: str
    model_id: str | None
    status: SnapshotStatus = SnapshotStatus.HEALTHY
    errors: tuple[str, ...] = ()
    snapshot_id: str = field(default_factory=lambda: uuid4().hex)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def validated(self) -> "TradingSnapshot":
        errors = list(self.errors)
        account_error = self.account.consistency_error()
        if account_error:
            errors.append(account_error)

        pos_missing_price = any(p.mark_price is None or p.unrealized_pnl is None for p in self.positions)
        if pos_missing_price:
            errors.append("POSITION_VALUATION_MISSING")
        account_stale = self.account.timestamp is None or (self.generated_at - self.account.timestamp).total_seconds() > 15
        position_stale = any((self.generated_at - p.timestamp).total_seconds() > 15 for p in self.positions)
        reconciliation_stale = self.reconciliation.last_completed_at is None or (self.generated_at - self.reconciliation.last_completed_at).total_seconds() > 30
        unknown_orders = any(
            str(o.get("status", "")).upper() in {"UNKNOWN", "SUBMITTING_TIMEOUT", "REMOTE_ONLY", "LOCAL_ONLY"}
            for o in self.open_orders
        )
        if account_stale:
            errors.append("ACCOUNT_DATA_STALE_OR_MISSING_TIMESTAMP")
        if position_stale:
            errors.append("POSITION_DATA_STALE")
        if reconciliation_stale:
            errors.append("RECONCILIATION_STALE_OR_NEVER_RUN")
        if unknown_orders:
            errors.append("UNKNOWN_ORDERS_PRESENT")

        status = self.status
        if not self.connection.authenticated and self.context.effective_mode.value != "LOCAL_PAPER":
            status = SnapshotStatus.AUTH_FAILED
            errors.append("REMOTE_ACCOUNT_NOT_AUTHENTICATED")
        elif self.context.degraded:
            status = SnapshotStatus.AUTH_FAILED if self.connection.error_code == "AUTH_FAILED" else SnapshotStatus.UNAVAILABLE
            if self.context.degrade_reason:
                errors.append(self.context.degrade_reason)
        elif self.market.mark_price is None or self.market.mark_price <= 0:
            status = SnapshotStatus.UNAVAILABLE
            errors.append("MARK_PRICE_UNAVAILABLE")
        elif self.market.timestamp is None:
            status = SnapshotStatus.UNAVAILABLE
            errors.append("MARKET_TIMESTAMP_MISSING")
        elif account_error or pos_missing_price or account_stale or position_stale or reconciliation_stale or unknown_orders:
            status = SnapshotStatus.INCONSISTENT
        elif self.market.stale_seconds is None or self.market.stale_seconds > 5:
            status = SnapshotStatus.STALE
            errors.append("MARKET_DATA_STALE")
        elif self.reconciliation.status != "OK":
            status = SnapshotStatus.INCONSISTENT
            errors.append(f"RECONCILIATION_{self.reconciliation.status}")

        if status != SnapshotStatus.HEALTHY:
            safe_actions = tuple(a for a in self.risk.allowed_actions if a in {"VIEW", "CANCEL_ENTRY_ORDERS", "REDUCE_ONLY", "CLOSE_ALL"})
            risk = RiskSnapshot(False, safe_actions, tuple(dict.fromkeys((*self.risk.reasons, *errors))))
        else:
            risk = self.risk

        return TradingSnapshot(
            context=self.context,
            connection=self.connection,
            market=self.market,
            account=self.account,
            positions=self.positions,
            open_orders=self.open_orders,
            reconciliation=self.reconciliation,
            risk=risk,
            configured_strategy=self.configured_strategy,
            effective_strategy=self.effective_strategy,
            model_id=self.model_id,
            status=status,
            errors=tuple(dict.fromkeys(errors)),
            snapshot_id=self.snapshot_id,
            generated_at=self.generated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, Decimal):
                return str(value)
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return [encode(v) for v in value]
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return {k: encode(v) for k, v in asdict(value).items()}
            return value
        return encode(self.validated())
