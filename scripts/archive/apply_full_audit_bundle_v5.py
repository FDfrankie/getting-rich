"""
Transactional, conflict-aware migration & integration script (v5) for Antigravity Quant.
Default mode (no args) is PLAN / DRY-RUN ONLY.

Usage:
    python scripts/apply_full_audit_bundle_v5.py                        # Plan / Dry-run only
    python scripts/apply_full_audit_bundle_v5.py --apply                # Stage, compile, backup, apply, verify (auto-rollback on test failure)
    python scripts/apply_full_audit_bundle_v5.py --rollback <RUN_ID>    # Restore files & manifest from .patch_backups/<RUN_ID>
    python scripts/apply_full_audit_bundle_v5.py --recover-interrupted  # Rollback any incomplete APPLYING run
    python scripts/apply_full_audit_bundle_v5.py --cleanup              # Dry-run cleanup list
    python scripts/apply_full_audit_bundle_v5.py --execute-cleanup [--force]
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / ".patch_backup_manifest.json"
BACKUPS_DIR = ROOT / ".patch_backups"


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read(rel_path: str, base_root: Path | None = None) -> str | None:
    p = (base_root or ROOT) / rel_path
    return p.read_text(encoding="utf-8") if p.exists() else None


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.patchtmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_manifest(manifest_path: Path | None = None) -> dict:
    mp = manifest_path or MANIFEST_PATH
    if not mp.exists():
        return {"files": {}, "runs": {}}
    return json.loads(mp.read_text(encoding="utf-8"))


def _save_manifest(manifest: dict, manifest_path: Path | None = None) -> None:
    mp = manifest_path or MANIFEST_PATH
    _atomic_write_text(mp, json.dumps(manifest, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# 1. Core Bundle Modules
# ---------------------------------------------------------------------------

RUNTIME_CONTEXT_PY = '''"""Immutable runtime context for one trading execution session."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping
from uuid import uuid4

from crypto_trading.settings import TradingMode


class ConnectionState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    CONNECTING = "CONNECTING"
    AUTHENTICATED = "AUTHENTICATED"
    AUTH_FAILED = "AUTH_FAILED"
    UNREACHABLE = "UNREACHABLE"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True, slots=True)
class RuntimeTradingContext:
    requested_mode: TradingMode
    effective_mode: TradingMode
    environment_id: str
    account_id: str
    venue: str
    connection_state: ConnectionState = ConnectionState.UNINITIALIZED
    degraded: bool = False
    degrade_reason: str | None = None
    runtime_session_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.degraded and not self.degrade_reason:
            raise ValueError("degrade_reason is required when degraded=True")
        if self.requested_mode == TradingMode.BITUNIX_LIVE and self.effective_mode != TradingMode.BITUNIX_LIVE:
            if not self.degraded:
                raise ValueError("A live-mode mismatch must be explicitly degraded")
        if self.effective_mode == TradingMode.BITUNIX_LIVE and self.connection_state != ConnectionState.AUTHENTICATED:
            raise ValueError("BITUNIX_LIVE requires an authenticated connection")

    @property
    def remote_account_claim_allowed(self) -> bool:
        return (
            not self.degraded
            and self.connection_state == ConnectionState.AUTHENTICATED
            and self.effective_mode in {TradingMode.BINANCE_TESTNET, TradingMode.BITUNIX_LIVE}
        )

    @classmethod
    def local_paper(cls, *, account_id: str = "paper-default") -> "RuntimeTradingContext":
        return cls(
            requested_mode=TradingMode.LOCAL_PAPER,
            effective_mode=TradingMode.LOCAL_PAPER,
            environment_id="local",
            account_id=account_id,
            venue="LOCAL",
            connection_state=ConnectionState.AUTHENTICATED,
        )

    @classmethod
    def degraded_from_remote(
        cls,
        requested_mode: TradingMode,
        *,
        account_id: str,
        venue: str,
        reason: str,
        connection_state: ConnectionState,
    ) -> "RuntimeTradingContext":
        if requested_mode == TradingMode.LOCAL_PAPER:
            raise ValueError("LOCAL_PAPER cannot degrade from a remote mode")
        return cls(
            requested_mode=requested_mode,
            effective_mode=TradingMode.LOCAL_PAPER,
            environment_id="degraded-local",
            account_id=account_id,
            venue=venue,
            connection_state=connection_state,
            degraded=True,
            degrade_reason=reason,
        )
'''

INSTRUMENTS_PY = '''"""Canonical instrument identity and Decimal-safe normalization."""
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
'''

TRADING_SNAPSHOT_PY = '''"""Atomic trading snapshot models and strict fail-closed consistency checks."""
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
'''

EXCHANGE_NORMALIZER_PY = '''"""Strict Binance/CCXT balance and position adapters (never guess missing fields)."""
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
'''

PAPER_BROKER_PY = '''"""Dedicated MarketDataService, PaperBroker, and sign-safe PnL calculation."""
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
'''

BRAIN_CONNECTIVITY_SERVICE_PY = '''"""Cached WorldQuant BRAIN connectivity probe and auth failure classifier."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class BrainConnectivity:
    service_available: bool
    configured: bool
    reachable: bool
    authenticated: bool
    biometric_required: bool
    terms_required: bool
    waf_blocked: bool
    rate_limited: bool
    simulation_enabled: bool
    auto_submit_enabled: bool
    local_alpha_worker_alive: bool
    brain_worker_alive: bool
    queue_depth: int
    last_http_status: int | None
    latency_ms: int | None
    checked_at: str
    safe_error_code: str | None


class BrainAuthFailureClassifier:
    """Classifies HTTP status, headers, and sanitized body fingerprints without leaking tokens."""

    @staticmethod
    def classify(status_code: int | None, headers: dict[str, Any] | None, body: str | None) -> str | None:
        if status_code in (200, 201):
            return None
        hdrs = {str(k).lower(): str(v).lower() for k, v in (headers or {}).items()}
        text = (body or "").lower()
        auth_hdr = hdrs.get("www-authenticate", "")
        ctype = hdrs.get("content-type", "")

        if status_code == 429:
            return "RATE_LIMITED"
        if status_code in (502, 503, 504):
            return "UPSTREAM_UNAVAILABLE"
        if status_code in (401, 403):
            if "persona" in auth_hdr or "biometric" in text:
                return "BRAIN_BIOMETRIC_REQUIRED"
            if "terms" in text or "agreement" in text:
                return "TERMS_REQUIRED"
            if "cloudflare" in text or "cf-ray" in hdrs or "waf" in text or "captcha" in text:
                return "BRAIN_WAF_BLOCKED"
            if "expired" in text:
                return "SESSION_EXPIRED"
            if "text/html" in ctype:
                return "HTML_RESPONSE_UNKNOWN"
            return "AUTH_FAILED"
        return f"HTTP_{status_code}" if status_code else "UNREACHABLE"


class BrainConnectivityService:
    """Maintains cached BRAIN connectivity state so 3-second UI refreshes never spam /authentication."""

    def __init__(self, brain_client: Any, job_queue: Any, settings_getter: Any) -> None:
        self.brain_client = brain_client
        self.job_queue = job_queue
        self.settings_getter = settings_getter
        self._last_http_status: int | None = None
        self._last_latency_ms: int | None = None
        self._last_error_code: str | None = None
        self._last_reachable: bool = False
        self._last_checked_at: datetime = datetime.now(timezone.utc)

    def record_probe_outcome(
        self,
        *,
        reachable: bool,
        http_status: int | None,
        latency_ms: int | None,
        error_code: str | None = None,
    ) -> None:
        self._last_reachable = reachable
        self._last_http_status = http_status
        self._last_latency_ms = latency_ms
        self._last_error_code = error_code
        self._last_checked_at = datetime.now(timezone.utc)

    def get_status(self, *, local_alpha_worker_alive: bool = False) -> dict[str, Any]:
        offline = bool(getattr(self.brain_client, "offline_mode", True))
        auth_attr = getattr(self.brain_client, "is_authenticated", False)
        authed = bool(auth_attr() if callable(auth_attr) else auth_attr) and not offline
        settings = self.settings_getter() if callable(self.settings_getter) else None
        sim_enabled = bool(getattr(settings, "brain_simulation_enabled", False)) if settings else False
        auto_sub = bool(getattr(settings, "brain_auto_submit_enabled", False)) if settings else False

        q_fn = getattr(self.job_queue, "queue_depth", None)
        q_depth = int(q_fn()) if callable(q_fn) else 0
        bw_fn = getattr(self.job_queue, "worker_alive", None)
        bw_alive = bool(bw_fn()) if callable(bw_fn) else False

        err_code = "BRAIN_CREDENTIALS_MISSING" if offline else self._last_error_code
        conn = BrainConnectivity(
            service_available=True,
            configured=not offline,
            reachable=authed or self._last_reachable,
            authenticated=authed,
            biometric_required=(err_code == "BRAIN_BIOMETRIC_REQUIRED"),
            terms_required=(err_code == "TERMS_REQUIRED"),
            waf_blocked=(err_code == "BRAIN_WAF_BLOCKED"),
            rate_limited=(err_code == "RATE_LIMITED"),
            simulation_enabled=sim_enabled,
            auto_submit_enabled=auto_sub,
            local_alpha_worker_alive=local_alpha_worker_alive,
            brain_worker_alive=bw_alive,
            queue_depth=q_depth,
            last_http_status=self._last_http_status,
            latency_ms=self._last_latency_ms,
            checked_at=self._last_checked_at.isoformat(),
            safe_error_code=err_code,
        )
        return asdict(conn)
'''

TWSE_CALENDAR_PY = '''"""Official TWSE trading calendar (2025-2027 holiday schedule + optional exchange_calendars XTAI)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    TAIPEI_TZ = ZoneInfo("Asia/Taipei")
except Exception:
    TAIPEI_TZ = timezone(timedelta(hours=8))

try:
    import exchange_calendars as xcals
    _XTAI_CAL = xcals.get_calendar("XTAI")
except Exception:
    _XTAI_CAL = None

# Verified TWSE market closures (national holidays & lunar new year closures 2025-2027)
TWSE_CLOSED_DATES: set[date] = {
    # 2025
    date(2025, 1, 1), date(2025, 1, 23), date(2025, 1, 24), date(2025, 1, 27),
    date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30), date(2025, 1, 31),
    date(2025, 2, 28), date(2025, 4, 3), date(2025, 4, 4), date(2025, 5, 1),
    date(2025, 5, 30), date(2025, 10, 6), date(2025, 10, 10),
    # 2026
    date(2026, 1, 1), date(2026, 2, 12), date(2026, 2, 13), date(2026, 2, 16),
    date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20),
    date(2026, 2, 27), date(2026, 4, 3), date(2026, 4, 6), date(2026, 5, 1),
    date(2026, 6, 19), date(2026, 9, 25), date(2026, 10, 9),
    # 2027
    date(2027, 1, 1), date(2027, 2, 3), date(2027, 2, 4), date(2027, 2, 5),
    date(2027, 2, 8), date(2027, 2, 9), date(2027, 2, 10), date(2027, 3, 1),
    date(2027, 4, 5), date(2027, 4, 30), date(2027, 6, 9), date(2027, 9, 15),
    date(2027, 10, 11),
}

CALENDAR_MIN_YEAR = 2025
CALENDAR_MAX_YEAR = 2027


def is_twse_trading_day(dt: date | datetime | None = None) -> bool:
    """Returns True iff the given Taipei date is a valid TWSE trading session. Fails closed outside bounds."""
    if dt is None:
        target = datetime.now(TAIPEI_TZ).date()
    elif isinstance(dt, datetime):
        target = dt.astimezone(TAIPEI_TZ).date() if dt.tzinfo else dt.date()
    else:
        target = dt

    if _XTAI_CAL is not None:
        try:
            return bool(_XTAI_CAL.is_session(target.isoformat()))
        except Exception:
            pass

    if not (CALENDAR_MIN_YEAR <= target.year <= CALENDAR_MAX_YEAR):
        # Fail closed when outside verified static calendar bounds
        return False
    if target.weekday() >= 5:
        return False
    return target not in TWSE_CLOSED_DATES
'''

TRADING_SNAPSHOT_API_PY = '''"""Framework-neutral handler helpers for GET /api/trading/snapshot."""
from __future__ import annotations

import json
from typing import Callable

from crypto_trading.trading_snapshot import TradingSnapshot

SnapshotProvider = Callable[[], TradingSnapshot]


def get_trading_snapshot_response(provider: SnapshotProvider) -> tuple[int, dict[str, str], bytes]:
    try:
        snapshot = provider().validated()
        payload = snapshot.to_dict()
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return 200, {"Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store"}, body
    except Exception as exc:
        payload = {
            "success": False,
            "status": "UNAVAILABLE",
            "error_code": "SNAPSHOT_PROVIDER_FAILED",
            "safe_error": type(exc).__name__,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return 503, {"Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store"}, body
'''

STATIC_SNAPSHOT_PATCH_JS = '''// Truthful trading snapshot renderer (single in-flight-guarded scheduler, zero innerHTML XSS, binds both canonical & legacy DOM IDs).
async function fetchJsonStrict(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  const text = await response.text();
  let payload = null;
  if (text.trim()) {
    try { payload = JSON.parse(text); }
    catch (_) {
      throw new Error(`NON_JSON_RESPONSE http=${response.status} endpoint=${url}`);
    }
  }
  if (!payload) throw new Error(`EMPTY_RESPONSE http=${response.status} endpoint=${url}`);
  if (!response.ok) {
    throw new Error(payload.safe_error || payload.error || `HTTP_${response.status}`);
  }
  return payload;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? 'N/A';
}

function renderTradingSnapshot(s) {
  const account = s.account || {};
  const market = s.market || {};
  const positions = Array.isArray(s.positions) ? s.positions : [];
  const curr = account.currency || market.settlement_currency || 'USDT';

  const eqText = account.equity == null ? 'N/A' : `${account.equity} ${curr}`;
  const walText = account.wallet_balance == null ? 'N/A' : `${account.wallet_balance} ${curr}`;
  const pnlText = account.unrealized_pnl == null ? 'N/A' : `${account.unrealized_pnl} ${curr}`;
  const posText = positions.length > 0 ? `${positions.length} 筆合約持倉` : 'FLAT (0)';
  const priceText = market.mark_price == null ? 'N/A' : `${market.mark_price} ${curr}`;
  const modeText = `${s.context?.requested_mode || 'UNKNOWN'} -> ${s.context?.effective_mode || 'UNKNOWN'}`;

  // Update canonical IDs
  setText('equityValue', eqText);
  setText('walletBalanceValue', walText);
  setText('unrealizedPnlValue', pnlText);
  setText('positionCountValue', String(positions.length));
  setText('markPriceValue', priceText);
  setText('tradingModeValue', modeText);
  setText('snapshotStatusValue', s.status || 'UNAVAILABLE');
  setText('snapshotAgeValue', market.stale_seconds == null ? 'N/A' : `${Number(market.stale_seconds).toFixed(1)}s`);

  // Also update existing index.html card IDs so legacy cards never show 15,000 / +112.50
  setText('cryptoEquity', eqText);
  setText('cryptoPrice', priceText);
  setText('cryptoPosition', posText);
  setText('cryptoPnL', pnlText);
  setText('cryptoModeBadge', s.context?.degraded ? `DEGRADED (${modeText})` : modeText);

  const safeForNewPositions = s.status === 'HEALTHY' && s.risk?.trading_enabled === true;
  document.querySelectorAll('[data-opens-position="true"]').forEach((button) => {
    button.disabled = !safeForNewPositions;
    button.title = safeForNewPositions ? '' : `Trading blocked: ${(s.errors || s.risk?.reasons || []).join(', ')}`;
  });
}

function updateSubsystemStatusBar(s, brainConn, sysHealth) {
  const headerEl = document.getElementById('headerStatus');
  const status = s?.status || 'UNAVAILABLE';
  const degraded = Boolean(s?.context?.degraded);
  if (headerEl) {
    if (status === 'HEALTHY' && !degraded) {
      headerEl.textContent = '系統健康 (HEALTHY)';
    } else if (degraded) {
      headerEl.textContent = `降級本地 (${s?.context?.degrade_reason || 'DEGRADED'})`;
    } else {
      headerEl.textContent = `警戒 (${status})`;
    }
  }
  let bar = document.getElementById('subsystemStatusBar');
  if (!bar) {
    const header = document.querySelector('header');
    if (header && header.parentNode) {
      bar = document.createElement('div');
      bar.id = 'subsystemStatusBar';
      bar.className = 'max-w-7xl mx-auto w-full px-4 pt-2 flex flex-wrap gap-2 text-xs';
      header.parentNode.insertBefore(bar, header.nextSibling);
    }
  }
  if (bar) {
    bar.replaceChildren();
    const pills = [
      ['Bot', sysHealth?.watchdog?.trading_bot || 'UNKNOWN', sysHealth?.watchdog?.trading_bot === 'OK'],
      ['Mode', `${s?.context?.requested_mode || '?'} -> ${s?.context?.effective_mode || '?'}`, !degraded],
      ['Market', s?.market?.data_source || 'N/A', status === 'HEALTHY'],
      ['Account', s?.connection?.authenticated ? 'AUTH' : (s?.connection?.error_code || 'LOCAL'), Boolean(s?.connection?.authenticated)],
      ['Reconcile', s?.reconciliation?.status || 'UNKNOWN', s?.reconciliation?.status === 'OK'],
      ['BRAIN Auth', brainConn?.authenticated ? 'OK' : (brainConn?.safe_error_code || 'OFFLINE'), Boolean(brainConn?.authenticated)],
      ['BRAIN Queue', String(brainConn?.queue_depth ?? 0), Boolean(brainConn?.brain_worker_alive)],
    ];
    pills.forEach(([k, v, ok]) => {
      const span = document.createElement('span');
      span.className = ok
        ? 'px-2 py-0.5 rounded border bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
        : 'px-2 py-0.5 rounded border bg-amber-500/10 border-amber-500/30 text-amber-300';
      const strong = document.createElement('strong');
      strong.textContent = `${k}: `;
      span.appendChild(strong);
      span.appendChild(document.createTextNode(String(v)));
      bar.appendChild(span);
    });
  }
}

let truthRefreshInFlight = false;
async function refreshAllTruthStatus() {
  if (truthRefreshInFlight) return;
  truthRefreshInFlight = true;
  let snap = null;
  let brainConn = null;
  let sysHealth = null;
  try {
    snap = await fetchJsonStrict('/api/trading/snapshot');
    renderTradingSnapshot(snap);
  } catch (error) {
    ['equityValue', 'walletBalanceValue', 'unrealizedPnlValue', 'positionCountValue', 'markPriceValue', 'cryptoEquity', 'cryptoPrice', 'cryptoPosition', 'cryptoPnL'].forEach((id) => setText(id, 'N/A'));
    setText('snapshotStatusValue', 'UNAVAILABLE');
    document.querySelectorAll('[data-opens-position="true"]').forEach((button) => { button.disabled = true; });
    console.error('[TRADING_SNAPSHOT]', error);
  }
  try { brainConn = await fetchJsonStrict('/api/brain/connectivity'); } catch (_) {}
  try { sysHealth = await fetchJsonStrict('/api/status'); } catch (_) {}
  updateSubsystemStatusBar(snap, brainConn, sysHealth);
  truthRefreshInFlight = false;
}

setInterval(refreshAllTruthStatus, 3000);
refreshAllTruthStatus();
'''


# ---------------------------------------------------------------------------
# 2. Pure Transform Functions
# ---------------------------------------------------------------------------

def _transform_settings_dotenv(txt: str) -> tuple[str, str]:
    if "load_dotenv(" in txt:
        return "ALREADY_APPLIED", txt
    anchor = "from pathlib import Path\n"
    if anchor in txt:
        block = (
            anchor
            + "try:\n"
            + "    from dotenv import load_dotenv\n"
            + '    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)\n'
            + "except ImportError:\n"
            + "    pass\n"
        )
        return "PATCHED", txt.replace(anchor, block, 1)
    m = re.search(r"^(?:from __future__[^\n]*\n)+", txt, flags=re.MULTILINE)
    insert_pos = m.end() if m else 0
    block = (
        "from pathlib import Path\n"
        "try:\n"
        "    from dotenv import load_dotenv\n"
        '    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)\n'
        "except ImportError:\n"
        "    pass\n"
    )
    return "PATCHED", txt[:insert_pos] + block + txt[insert_pos:]


def _transform_env_example(txt: str) -> tuple[str, str]:
    new_txt = txt
    if "BRAIN_PASSWORD=" not in new_txt:
        if "BRAIN_API_KEY=" in new_txt:
            new_txt = new_txt.replace(
                "BRAIN_API_KEY=",
                "BRAIN_PASSWORD=\n# BRAIN_API_KEY=  # (Deprecated alias; BrainClient uses BRAIN_EMAIL + BRAIN_PASSWORD)",
            )
        else:
            new_txt = new_txt.rstrip() + "\nBRAIN_EMAIL=\nBRAIN_PASSWORD=\n"
    if "BINANCE_TESTNET_API_KEY=" not in new_txt:
        new_txt = new_txt.rstrip() + "\nBINANCE_TESTNET_API_KEY=\nBINANCE_TESTNET_API_SECRET=\n"
    return ("ALREADY_APPLIED" if new_txt == txt else "PATCHED"), new_txt


def _transform_pyproject(txt: str) -> tuple[str, str]:
    if "apscheduler" in txt.lower():
        return "ALREADY_APPLIED", txt
    target = '"requests>=2.31.0"'
    if target not in txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", txt.replace(target, '"requests>=2.31.0",\n    "apscheduler>=3.10.0"', 1)


def _transform_bot_runner(txt: str) -> tuple[str, str]:
    if "PAPER_TRADING" not in txt:
        return "ALREADY_APPLIED", txt
    new_txt = re.sub(
        r"from\s+crypto_trading\.config\s+import\s*(?:\(\s*DEFAULT_SYMBOL\s*,\s*INITIAL_BALANCE_USDT\s*,\s*PAPER_TRADING\s*,?\s*\)|DEFAULT_SYMBOL\s*,\s*INITIAL_BALANCE_USDT\s*,\s*PAPER_TRADING)",
        "from crypto_trading.config import DEFAULT_SYMBOL, INITIAL_BALANCE_USDT\nfrom crypto_trading.settings import TradingMode, get_settings",
        txt,
        flags=re.MULTILINE,
    )
    new_txt = new_txt.replace(
        "self.is_paper = PAPER_TRADING",
        "self.is_paper = get_settings().trading_mode == TradingMode.LOCAL_PAPER",
    )
    if "PAPER_TRADING" in new_txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", new_txt


def _transform_feature_engineer(txt: str) -> tuple[str, str]:
    if "pct_change(fill_method=None)" in txt and "ETH_corr_BTC_24_valid" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = re.sub(r"\.pct_change\(\s*\)", ".pct_change(fill_method=None)", txt)
    new_txt = re.sub(r"(\.corr\([^)]+\))\.fillna\(\s*0(?:\.0)?\s*\)", r"\1", new_txt)
    if "ETH_corr_BTC_24_valid" not in new_txt and 'cross_df["ETH_corr_BTC_24"]' in new_txt:
        new_txt = re.sub(
            r'(cross_df\["ETH_corr_BTC_24"\]\s*=[^\n]+\n)',
            r'\1            cross_df["ETH_corr_BTC_24_valid"] = cross_df["ETH_corr_BTC_24"].notna().astype(int)\n',
            new_txt,
            count=1,
        )
    if new_txt == txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", new_txt


def _transform_brain_client(txt: str) -> tuple[str, str]:
    if "BrainBiometricRequired" in txt:
        return "ALREADY_APPLIED", txt
    cls_def = (
        "class BrainBiometricRequired(RuntimeError):\n"
        '    error_code = "BRAIN_BIOMETRIC_REQUIRED"\n\n\n'
        "class BrainWAFBlocked(RuntimeError):\n"
        '    error_code = "BRAIN_WAF_BLOCKED"\n\n\n'
        "def _classify_auth_challenge(response) -> str | None:\n"
        '    authenticate = response.headers.get("WWW-Authenticate", "").lower()\n'
        '    body = (response.text or "").lower()\n'
        "    if response.status_code in {401, 403}:\n"
        '        if "persona" in authenticate or "biometric" in body:\n'
        '            return "BRAIN_BIOMETRIC_REQUIRED"\n'
        '        if "cloudflare" in body or "waf" in body or "captcha" in body:\n'
        '            return "BRAIN_WAF_BLOCKED"\n'
        "    return None\n\n\n"
    )
    new_txt = txt.replace("class BrainClient:", cls_def + "class BrainClient:", 1)
    m = re.search(r"^([ \t]*)r\.raise_for_status\(\)", new_txt, flags=re.MULTILINE)
    if not m:
        return "TARGET_NOT_FOUND", txt
    indent = m.group(1)
    replacement = (
        f"{indent}challenge = _classify_auth_challenge(r)\n"
        f'{indent}if challenge == "BRAIN_BIOMETRIC_REQUIRED":\n'
        f"{indent}    raise BrainBiometricRequired(\n"
        f'{indent}        "WorldQuant BRAIN requires interactive verification. Complete verification in the official browser session and retry."\n'
        f"{indent}    )\n"
        f'{indent}if challenge == "BRAIN_WAF_BLOCKED":\n'
        f'{indent}    raise BrainWAFBlocked("WorldQuant BRAIN request blocked by WAF/anti-bot challenge.")\n'
        f"{indent}r.raise_for_status()"
    )
    return "PATCHED", new_txt[:m.start()] + replacement + new_txt[m.end():]


def _transform_brain_job_queue(txt: str) -> tuple[str, str]:
    if "def queue_depth(self)" in txt and "autostart: bool = True" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt.replace(
        "def __init__(self, brain_client: Any, default_poll_interval: float = 2.0) -> None:",
        "def __init__(self, brain_client: Any, default_poll_interval: float = 2.0, autostart: bool = True) -> None:",
        1,
    )
    new_txt = new_txt.replace(
        "        self._running: bool = True\n\n        self._start_worker()",
        "        self._running: bool = False\n        if autostart:\n            self.start()",
        1,
    )
    anchor = "    def get_job(self, job_id: str)"
    if anchor not in new_txt:
        return "TARGET_NOT_FOUND", txt
    methods = (
        "    def start(self) -> None:\n"
        "        with self._lock:\n"
        "            if self._worker_thread and self._worker_thread.is_alive():\n"
        "                return\n"
        "            self._running = True\n"
        "            self._start_worker()\n\n"
        "    def stop(self) -> None:\n"
        "        self._running = False\n\n"
        "    def join(self, timeout: float = 2.0) -> None:\n"
        "        t = self._worker_thread\n"
        "        if t and t.is_alive():\n"
        "            t.join(timeout=timeout)\n\n"
        "    def queue_depth(self) -> int:\n"
        "        with self._lock:\n"
        "            return len(self._pending_queue)\n\n"
        "    def worker_alive(self) -> bool:\n"
        "        return bool(self._running and self._worker_thread and self._worker_thread.is_alive())\n\n"
    )
    if "def queue_depth(self)" not in new_txt:
        new_txt = new_txt.replace(anchor, methods + anchor, 1)
    return "PATCHED", new_txt


def _transform_alpha_miner(txt: str) -> tuple[str, str]:
    if "LOCAL_CANDIDATE_ONLY" in txt:
        return "ALREADY_APPLIED", txt
    new_txt = txt.replace('"status": "VERIFIED_QUALIFIED"', '"status": "LOCAL_SCREENED"')
    new_txt = new_txt.replace(
        '"category": "Price-Volume / Continuous Mined"',
        '"category": "LOCAL_CANDIDATE_ONLY / Unverified by BRAIN"',
    )
    if new_txt == txt:
        return "TARGET_NOT_FOUND", txt
    return "PATCHED", new_txt


def _transform_scheduler(txt: str) -> tuple[str, str]:
    if "is_twse_trading_day" in txt and "APScheduler is required for timezone-safe scheduling" in txt:
        return "ALREADY_APPLIED", txt
    header_replacement = (
        "from datetime import datetime\n"
        "from taiwan_stock.twse_calendar import TAIPEI_TZ as _TAIPEI_TZ, is_twse_trading_day\n"
        "try:\n"
        "    from apscheduler.schedulers.blocking import BlockingScheduler\n"
        "    from apscheduler.triggers.cron import CronTrigger\n"
        "except ImportError:\n"
        "    BlockingScheduler = None\n"
        "    CronTrigger = None\n"
    )
    new_txt = txt.replace("from datetime import datetime\n", header_replacement, 1)
    new_txt = new_txt.replace("datetime.now().strftime", "datetime.now(_TAIPEI_TZ).strftime")
    sched_pattern = re.compile(
        r"    schedule\.every\(\)\.day\.at\(AFTERNOON_SCHEDULE_TIME\)\.do\(.*?time\.sleep\(30\)",
        re.DOTALL,
    )
    new_main_sched = (
        '    if BlockingScheduler is None or CronTrigger is None:\n'
        '        raise RuntimeError("APScheduler is required for timezone-safe scheduling (Asia/Taipei)")\n'
        '    sched = BlockingScheduler(timezone=_TAIPEI_TZ)\n'
        '    sched.add_job(lambda: is_twse_trading_day() and scheduled_job("下午盤後報告"), CronTrigger(hour=16, minute=30, timezone=_TAIPEI_TZ))\n'
        '    sched.add_job(lambda: is_twse_trading_day() and scheduled_job("早上盤前晨報"), CronTrigger(hour=8, minute=0, timezone=_TAIPEI_TZ))\n'
        '    sched.start()'
    )
    if not sched_pattern.search(new_txt):
        return "TARGET_NOT_FOUND", txt
    new_txt = sched_pattern.sub(lambda _: new_main_sched, new_txt, count=1)
    return "PATCHED", new_txt


def _transform_report_generator(txt: str) -> tuple[str, str]:
    if "def collect_report_snapshot(" in txt and "html.escape" in txt:
        return "ALREADY_APPLIED", txt
    tz_helper = (
        "import html\n"
        "from dataclasses import dataclass\n"
        "from datetime import datetime, timedelta, timezone\n"
        "try:\n"
        "    from zoneinfo import ZoneInfo\n"
        '    _TAIPEI_TZ = ZoneInfo("Asia/Taipei")\n'
        "except Exception:\n"
        "    _TAIPEI_TZ = timezone(timedelta(hours=8))\n\n"
        "@dataclass(frozen=True)\n"
        "class ReportSnapshot:\n"
        "    status: str\n"
        "    session_type: str\n"
        "    generated_at: str\n"
        "    last_valid_trade_date: str | None\n"
        "    market_info: dict\n"
        "    screener_res: dict\n"
    )
    new_txt = txt.replace("from datetime import datetime\n", tz_helper, 1)
    new_txt = new_txt.replace("datetime.now().strftime", "datetime.now(_TAIPEI_TZ).strftime")
    new_txt = new_txt.replace('market_info.get("date", today_str)', '(market_info.get("date") or "DATA_UNAVAILABLE")')
    new_txt = new_txt.replace(
        '"> ✅ 最近期無觸發停損或重大虧損事件，系統執行紀律率 100%。"',
        '"> ℹ️ 尚無足夠平倉 / Post-Mortem 紀錄，無法計算執行紀律率（Insufficient data）。"',
    )
    new_txt = new_txt.replace("{e!s}", "{html.escape(str(e), quote=True)}")

    # Add collect_report_snapshot() and guard both generate() and generate_html_report()
    init_anchor = "        self.sync = GDrivePersistenceSync()\n"
    collector_code = (
        init_anchor
        + "        self._last_snapshot: ReportSnapshot | None = None\n\n"
        + "    def collect_report_snapshot(self, session_type: str = '盤前晨報', cached_market_info: dict = None) -> ReportSnapshot:\n"
        + "        now_str = datetime.now(_TAIPEI_TZ).strftime('%Y-%m-%d %H:%M:%S')\n"
        + "        m_info = cached_market_info if cached_market_info else self.market_analyzer.analyze()\n"
        + "        is_valid = m_info.get('status') == '正常' and m_info.get('latest_close') not in (None, 'N/A')\n"
        + "        s_res = self.stock_screener.run_screener_detailed() if is_valid else {'scanned_count': 0, 'qualified_count': 0, 'picks': [], 'near_picks': [], 'defense_hedges': []}\n"
        + "        snap = ReportSnapshot(\n"
        + "            status='VALID' if is_valid else 'DATA_UNAVAILABLE',\n"
        + "            session_type=session_type,\n"
        + "            generated_at=now_str,\n"
        + "            last_valid_trade_date=m_info.get('date') if is_valid else None,\n"
        + "            market_info=m_info,\n"
        + "            screener_res=s_res,\n"
        + "        )\n"
        + "        self._last_snapshot = snap\n"
        + "        return snap\n"
    )
    if init_anchor not in new_txt:
        return "TARGET_NOT_FOUND", txt
    new_txt = new_txt.replace(init_anchor, collector_code, 1)

    gen_pat = re.compile(
        r'        print\("\[\*\] 正在進行大盤多空診斷\.\.\."\)\s+market_info = \(?\s*cached_market_info if cached_market_info else self\.market_analyzer\.analyze\(\)\s*\)?\s+print\("\[\*\] 正在掃描潛力強勢標的\.\.\."\)\s+screener_res = self\.stock_screener\.run_screener_detailed\(\)',
        re.DOTALL,
    )
    gen_new = (
        "        snap = self.collect_report_snapshot(session_type=session_type, cached_market_info=cached_market_info)\n"
        "        if snap.status != 'VALID':\n"
        '            return f"# 📊 台股量化決策報告【{session_type}】({today_str})\\n\\nREPORT_STATUS: DATA_UNAVAILABLE\\n\\n> ⚠️ 無法取得有效大盤收盤資料，已中止計算多空分數與選股建議。"\n'
        "        market_info = snap.market_info\n"
        "        screener_res = snap.screener_res"
    )
    if not gen_pat.search(new_txt):
        return "TARGET_NOT_FOUND", txt
    new_txt = gen_pat.sub(lambda _: gen_new, new_txt, count=1)

    html_pat = re.compile(
        r'        market_info = \(?\s*cached_market_info if cached_market_info else self\.market_analyzer\.analyze\(\)\s*\)?\s+screener_res = self\.stock_screener\.run_screener_detailed\(\)',
        re.DOTALL,
    )
    html_new = (
        "        snap = self._last_snapshot if (self._last_snapshot and cached_market_info is not None) else self.collect_report_snapshot(session_type=session_type, cached_market_info=cached_market_info)\n"
        "        if snap.status != 'VALID':\n"
        '            return "<div class=\'p-4 rounded-xl bg-rose-950/40 border border-rose-500/30 text-rose-300\'><strong>REPORT_STATUS: DATA_UNAVAILABLE</strong> — 無法取得有效大盤收盤資料，已中止計算多空分數與選股建議。</div>"\n'
        "        market_info = snap.market_info\n"
        "        screener_res = snap.screener_res"
    )
    if not html_pat.search(new_txt):
        return "TARGET_NOT_FOUND", txt
    new_txt = html_pat.sub(lambda _: html_new, new_txt, count=1)
    return "PATCHED", new_txt


def _transform_server(txt: str) -> tuple[str, str]:
    if "BrainConnectivityService" in txt and "PaperBroker" in txt and "allow_reuse_address = True" in txt:
        return "ALREADY_APPLIED", txt

    new_txt = txt
    if "def _now_taipei()" not in new_txt:
        helper = (
            "from datetime import datetime, timedelta, timezone\n"
            "from decimal import Decimal\n"
            "try:\n"
            "    from zoneinfo import ZoneInfo\n"
            '    _TAIPEI_TZ = ZoneInfo("Asia/Taipei")\n'
            "except Exception:\n"
            "    _TAIPEI_TZ = timezone(timedelta(hours=8))\n\n"
            "def _now_taipei() -> datetime:\n"
            "    return datetime.now(_TAIPEI_TZ)\n"
        )
        new_txt = re.sub(r"from datetime import datetime[^\n]*\n", helper, new_txt, count=1)

    if "get_trading_snapshot_response" not in new_txt:
        snap_import = (
            "from crypto_trading.exchange_normalizer import normalize_binance_balance, normalize_remote_positions\n"
            "from crypto_trading.paper_broker import MarketDataService, PaperBroker\n"
            "from crypto_trading.runtime_context import ConnectionState, RuntimeTradingContext\n"
            "from crypto_trading.settings import TradingMode, get_settings\n"
            "from crypto_trading.trading_snapshot import (\n"
            "    AccountSnapshot, ConnectionSnapshot, MarketSnapshot, PositionSnapshot,\n"
            "    ReconciliationSnapshot, RiskSnapshot, TradingSnapshot,\n"
            ")\n"
            "from web_ui.trading_snapshot_api import get_trading_snapshot_response\n"
            "from worldquant_brain.connectivity_service import BrainConnectivityService\n"
        )
        new_txt = new_txt.replace("from crypto_trading.config import (", snap_import + "from crypto_trading.config import (", 1)

    for old_expr, new_expr in [
        ('ts = datetime.now().strftime("%H:%M:%S")', 'ts = _now_taipei().strftime("%H:%M:%S")'),
        ('now_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")', 'now_dt = _now_taipei().strftime("%Y-%m-%d %H:%M:%S")'),
        ('"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")', '"time": _now_taipei().strftime("%Y-%m-%d %H:%M:%S")'),
        ("datetime.now().strftime('%H:%M:%S')", "_now_taipei().strftime('%H:%M:%S')"),
    ]:
        new_txt = new_txt.replace(old_expr, new_expr)

    if "def build_trading_snapshot(" not in new_txt:
        method_anchor = "    def add_log(self, msg: str):"
        if method_anchor not in new_txt:
            return "TARGET_NOT_FOUND", txt
        method_code = '''    def _parse_recon_snapshot(self, now: datetime) -> ReconciliationSnapshot:
        recon_svc = getattr(self, "reconciliation_service", None)
        if recon_svc is None or not callable(getattr(recon_svc, "get_latest_result", None)):
            return ReconciliationSnapshot("NEVER_RUN", 0, None)
        latest_rec = recon_svc.get_latest_result()
        if not isinstance(latest_rec, dict):
            return ReconciliationSnapshot("NEVER_RUN", 0, None)
        raw_ts = latest_rec.get("completed_at")
        comp_at: datetime | None = None
        if isinstance(raw_ts, datetime):
            comp_at = raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
        elif isinstance(raw_ts, str):
            try:
                parsed = datetime.fromisoformat(raw_ts)
                comp_at = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except Exception:
                comp_at = None
        if comp_at is None:
            return ReconciliationSnapshot("NEVER_RUN", 0, None)
        age = (now - comp_at).total_seconds()
        disc = int(latest_rec.get("discrepancy_count", 0))
        if age > 30:
            st = "STALE"
        elif int(latest_rec.get("unknown_order_count", 0)) > 0:
            st = "UNKNOWN_ORDERS"
        elif not latest_rec.get("in_sync", False):
            st = "DRIFT"
        else:
            st = "OK"
        return ReconciliationSnapshot(st, disc, comp_at)

    def build_trading_snapshot(self) -> TradingSnapshot:
        now = datetime.now(timezone.utc)
        raw_mode = getattr(self, "execution_mode", TradingMode.LOCAL_PAPER)
        if isinstance(raw_mode, TradingMode):
            req_mode = raw_mode
        elif str(raw_mode) in {m.value for m in TradingMode}:
            req_mode = TradingMode(str(raw_mode))
        else:
            req_mode = TradingMode.LOCAL_PAPER

        recon_snap = self._parse_recon_snapshot(now)

        if req_mode == TradingMode.BINANCE_TESTNET:
            t0 = time.perf_counter()
            probe_fn = getattr(self.testnet_client, "probe_authenticated_connection", None)
            is_connected = bool(getattr(self.testnet_client, "is_connected", False)) and not bool(getattr(self.testnet_client, "mock_mode", True))
            if callable(probe_fn):
                probe = probe_fn()
                is_connected = bool(getattr(probe, "authenticated", False))
            if not is_connected:
                ctx = RuntimeTradingContext.degraded_from_remote(
                    TradingMode.BINANCE_TESTNET,
                    account_id="testnet-degraded",
                    venue="BINANCE_TESTNET",
                    reason="TESTNET_AUTH_OR_CONNECTION_UNAVAILABLE",
                    connection_state=ConnectionState.AUTH_FAILED,
                )
                return TradingSnapshot(
                    context=ctx,
                    connection=ConnectionSnapshot("BINANCE_TESTNET", False, False, None, None, "AUTH_FAILED", "Remote testnet unavailable; no fake values shown"),
                    market=MarketSnapshot(self.active_bot.symbol, self.active_bot.symbol, "PERPETUAL", "USDT", None, None, "UNAVAILABLE", None, None),
                    account=AccountSnapshot(None, None, None, None, None, None, None, "USDT", source="UNAVAILABLE", timestamp=None),
                    positions=(),
                    open_orders=(),
                    reconciliation=recon_snap,
                    risk=RiskSnapshot(False, ("VIEW", "CANCEL_ENTRY_ORDERS", "REDUCE_ONLY", "CLOSE_ALL"), ("TESTNET_UNAVAILABLE",)),
                    configured_strategy=self.bot_strategy_name,
                    effective_strategy=self.bot_strategy_name,
                    model_id=None,
                ).validated()

            t = self.testnet_client.fetch_ticker(self.active_bot.symbol)
            b = self.testnet_client.fetch_balance()
            ps = self.testnet_client.fetch_positions()
            oo_fn = getattr(self.testnet_client, "fetch_open_orders", None)
            open_orders = tuple(oo_fn(self.active_bot.symbol) or ()) if callable(oo_fn) else ()
            latency_ms = round((time.perf_counter() - t0) * 1000)

            last_p = t.get("last")
            price = Decimal(str(last_p)) if last_p is not None else None
            ts_ms = t.get("timestamp")
            if ts_ms is not None:
                mkt_ts = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
                stale_s = max(0.0, (now - mkt_ts).total_seconds())
                ts_src = "EXCHANGE_EVENT"
            else:
                mkt_ts = now
                stale_s = 0.0
                ts_src = "RECEIVE_TIME"

            acc = normalize_binance_balance(b, default_currency=str(b.get("settlement_currency", "USDT")), now=now)
            pos_snaps = normalize_remote_positions(ps, self.active_bot.symbol, "BINANCE_TESTNET", now)
            ctx = RuntimeTradingContext(
                requested_mode=TradingMode.BINANCE_TESTNET,
                effective_mode=TradingMode.BINANCE_TESTNET,
                environment_id="testnet",
                account_id="binance-testnet",
                venue="BINANCE_TESTNET",
                connection_state=ConnectionState.AUTHENTICATED,
            )
            return TradingSnapshot(
                context=ctx,
                connection=ConnectionSnapshot("BINANCE_TESTNET", True, True, latency_ms, now),
                market=MarketSnapshot(self.active_bot.symbol, self.active_bot.symbol, "PERPETUAL", acc.currency, price, price, "EXCHANGE_REST", mkt_ts, stale_s, ts_src),
                account=acc,
                positions=pos_snaps,
                open_orders=open_orders,
                reconciliation=recon_snap,
                risk=RiskSnapshot(True, ("VIEW", "OPEN_LONG", "OPEN_SHORT", "REDUCE_ONLY", "CLOSE_ALL"), ()),
                configured_strategy=self.bot_strategy_name,
                effective_strategy=self.bot_strategy_name,
                model_id=None,
            ).validated()

        # LOCAL_PAPER: dedicated MarketDataService + PaperBroker (never uses mock testnet balance)
        ctx = RuntimeTradingContext.local_paper()
        mds = getattr(self, "market_data_service", None) or MarketDataService(getattr(self, "public_market_adapter", None))
        broker = getattr(self, "paper_broker", None) or PaperBroker(self.active_bot)
        tick = mds.get_latest_tick(self.active_bot.symbol, now=now)
        stale_s = max(0.0, (now - tick.event_time).total_seconds()) if tick.event_time else None
        pos_snaps = broker.fetch_positions(tick.price, now=now)
        open_orders = broker.fetch_open_orders()
        acc_snap = broker.account_snapshot(pos_snaps, now=now)
        paper_recon = broker.reconcile(now=now)
        return TradingSnapshot(
            context=ctx,
            connection=ConnectionSnapshot("LOCAL", True, True, 0, now),
            market=MarketSnapshot(self.active_bot.symbol, self.active_bot.symbol, "PERPETUAL", "USDT", tick.price, tick.price, tick.source, tick.event_time, stale_s, tick.timestamp_source),
            account=acc_snap,
            positions=pos_snaps,
            open_orders=open_orders,
            reconciliation=paper_recon,
            risk=RiskSnapshot(True, ("VIEW", "OPEN_LONG", "OPEN_SHORT", "REDUCE_ONLY", "CLOSE_ALL"), ()),
            configured_strategy=self.bot_strategy_name,
            effective_strategy=self.bot_strategy_name,
            model_id=None,
        ).validated()

'''
        new_txt = new_txt.replace(method_anchor, method_code + method_anchor, 1)

    route_anchor = '        if path == "/api/status" or path == "/health":'
    if '"/api/trading/snapshot"' not in new_txt:
        if route_anchor not in new_txt:
            return "TARGET_NOT_FOUND", txt
        new_routes = (
            '        if path == "/api/trading/snapshot":\n'
            '            status_code, headers, body = get_trading_snapshot_response(APP_STATE.build_trading_snapshot)\n'
            '            self.send_response(status_code)\n'
            '            for hk, hv in headers.items():\n'
            '                self.send_header(hk, hv)\n'
            '            self.send_header("Content-Length", str(len(body)))\n'
            '            self.end_headers()\n'
            '            self.wfile.write(body)\n'
            '            return\n\n'
            '        if path == "/health/alpha-miner" or path == "/api/brain/status" or path == "/api/brain/connectivity":\n'
            '            if not hasattr(APP_STATE, "brain_connectivity_service"):\n'
            '                APP_STATE.brain_connectivity_service = BrainConnectivityService(APP_STATE.brain_client, APP_STATE.brain_job_queue, get_settings)\n'
            '            local_alive = bool(APP_STATE.miner_thread and APP_STATE.miner_thread.is_alive())\n'
            '            payload = APP_STATE.brain_connectivity_service.get_status(local_alpha_worker_alive=local_alive)\n'
            '            payload["time_taipei"] = _now_taipei().strftime("%Y-%m-%d %H:%M:%S")\n'
            '            return self._send_json_response(payload)\n\n'
        )
        new_txt = new_txt.replace(route_anchor, new_routes + route_anchor, 1)

    if "allow_reuse_address = True" not in new_txt:
        if "    httpd = ThreadingHTTPServer(" not in new_txt:
            return "TARGET_NOT_FOUND", txt
        new_txt = new_txt.replace(
            "    httpd = ThreadingHTTPServer(server_address, AppAPIHandler)",
            "    ThreadingHTTPServer.allow_reuse_address = True\n    ThreadingHTTPServer.daemon_threads = True\n    httpd = ThreadingHTTPServer(server_address, AppAPIHandler)",
            1,
        )
    return "PATCHED", new_txt


def _transform_index_html(txt: str) -> tuple[str, str]:
    new_txt = txt
    # 1. Remove hardcoded initial financial numbers
    new_txt = new_txt.replace(
        '<p id="cryptoEquity" class="text-lg md:text-2xl font-black text-white mt-1">10,000.00 <span class="text-xs font-normal text-slate-400">USDT</span></p>',
        '<p id="cryptoEquity" class="text-lg md:text-2xl font-black text-white mt-1">N/A</p>',
    )
    new_txt = new_txt.replace(
        '<p id="cryptoPnL" class="text-lg md:text-2xl font-black text-slate-400 mt-1">+0.00 <span class="text-xs font-normal text-slate-400">USDT</span></p>',
        '<p id="cryptoPnL" class="text-lg md:text-2xl font-black text-slate-400 mt-1">N/A</p>',
    )
    # 2. Mark position-opening buttons with data-opens-position="true" (idempotent)
    if 'data-opens-position="true"' not in new_txt:
        new_txt = new_txt.replace(
            'onclick="executeOrder(\'BUY\')"',
            'data-opens-position="true" onclick="executeOrder(\'BUY\')"',
        )
        new_txt = new_txt.replace(
            'onclick="executeOrder(\'SELL\')"',
            'data-opens-position="true" onclick="executeOrder(\'SELL\')"',
        )
    # 3. Initialize MA5/20/60 legend on chart load
    anchor = "if (d.overlay.bb_lower) bbLowerLine.setData(d.overlay.bb_lower);\n          }"
    if "initLegendFromLatestCandle" not in new_txt and anchor in new_txt:
        snippet = (
            anchor
            + "\n\n          // initLegendFromLatestCandle\n"
            + "          const lastC = d.candles[d.candles.length - 1];\n"
            + "          if (lastC) {\n"
            + "            const setTxt = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };\n"
            + "            setTxt('legendOpen', Number(lastC.open).toFixed(2));\n"
            + "            setTxt('legendHigh', Number(lastC.high).toFixed(2));\n"
            + "            setTxt('legendLow', Number(lastC.low).toFixed(2));\n"
            + "            setTxt('legendClose', Number(lastC.close).toFixed(2));\n"
            + "            setTxt('legendVol', Number(lastC.volume || 0).toLocaleString(undefined, { maximumFractionDigits: 2 }));\n"
            + "            if (d.overlay && d.overlay.ma5 && d.overlay.ma5.length) setTxt('legendMA5', Number(d.overlay.ma5[d.overlay.ma5.length - 1].value).toFixed(2));\n"
            + "            if (d.overlay && d.overlay.ma20 && d.overlay.ma20.length) setTxt('legendMA20', Number(d.overlay.ma20[d.overlay.ma20.length - 1].value).toFixed(2));\n"
            + "            if (d.overlay && d.overlay.ma60 && d.overlay.ma60.length) setTxt('legendMA60', Number(d.overlay.ma60[d.overlay.ma60.length - 1].value).toFixed(2));\n"
            + "          }"
        )
        new_txt = new_txt.replace(anchor, snippet, 1)
    if "static_snapshot_patch.js" not in new_txt and "</body>" in new_txt:
        new_txt = new_txt.replace("</body>", '  <script src="/static/static_snapshot_patch.js"></script>\n</body>', 1)
    return ("ALREADY_APPLIED" if new_txt == txt else "PATCHED"), new_txt


# ---------------------------------------------------------------------------
# 3. Comprehensive Test Suites (Runtime Truth + Behavioral + Migration Engine)
# ---------------------------------------------------------------------------

TEST_RUNTIME_TRUTH_PY = '''from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest

from crypto_trading.paper_broker import calculate_pnl
from crypto_trading.runtime_context import ConnectionState, RuntimeTradingContext, TradingMode
from crypto_trading.trading_snapshot import (
    AccountSnapshot,
    ConnectionSnapshot,
    MarketSnapshot,
    ReconciliationSnapshot,
    RiskSnapshot,
    SnapshotStatus,
    TradingSnapshot,
)


class TestRuntimeTruth(unittest.TestCase):
    def test_remote_degradation_is_explicit_and_cannot_claim_remote_account(self):
        ctx = RuntimeTradingContext.degraded_from_remote(
            TradingMode.BINANCE_TESTNET,
            account_id="testnet-account",
            venue="BINANCE",
            reason="AUTH_FAILED",
            connection_state=ConnectionState.AUTH_FAILED,
        )
        self.assertIs(ctx.requested_mode, TradingMode.BINANCE_TESTNET)
        self.assertIs(ctx.effective_mode, TradingMode.LOCAL_PAPER)
        self.assertTrue(ctx.degraded)
        self.assertFalse(ctx.remote_account_claim_allowed)

    def test_live_mode_requires_authentication(self):
        with self.assertRaises(ValueError):
            RuntimeTradingContext(
                requested_mode=TradingMode.BITUNIX_LIVE,
                effective_mode=TradingMode.BITUNIX_LIVE,
                environment_id="prod",
                account_id="live",
                venue="BITUNIX",
                connection_state=ConnectionState.UNREACHABLE,
            )

    def test_local_paper_short_profit_when_price_falls(self):
        pnl = calculate_pnl("SHORT", Decimal("-0.01"), Decimal("65000"), Decimal("64000"))
        self.assertEqual(pnl, Decimal("10.00"))
        pnl_long = calculate_pnl("LONG", Decimal("0.01"), Decimal("64000"), Decimal("65000"))
        self.assertEqual(pnl_long, Decimal("10.00"))

    def test_equity_mismatch_fails_closed(self):
        now = datetime.now(timezone.utc)
        snapshot = TradingSnapshot(
            context=RuntimeTradingContext.local_paper(),
            connection=ConnectionSnapshot("LOCAL", True, True, 0, now),
            market=MarketSnapshot("BTC", "BTC/USDT", "PERPETUAL", "USDT", Decimal("100"), Decimal("100"), "LOCAL", now, 0.1),
            account=AccountSnapshot(Decimal("1000"), Decimal("25"), Decimal("1100"), Decimal("900"), None, None, None, "USDT", source="TEST", timestamp=now),
            positions=(),
            open_orders=(),
            reconciliation=ReconciliationSnapshot("OK", 0, now),
            risk=RiskSnapshot(True, ("VIEW", "OPEN_LONG", "REDUCE_ONLY"), ()),
            configured_strategy="PPO",
            effective_strategy="MOMENTUM_FALLBACK",
            model_id=None,
        ).validated()
        self.assertIs(snapshot.status, SnapshotStatus.INCONSISTENT)
        self.assertFalse(snapshot.risk.trading_enabled)
        self.assertNotIn("OPEN_LONG", snapshot.risk.allowed_actions)
        self.assertIn("REDUCE_ONLY", snapshot.risk.allowed_actions)

    def test_stale_account_or_position_or_unknown_orders_block_entries(self):
        now = datetime.now(timezone.utc)
        old_ts = now - timedelta(seconds=20)
        # 1. Stale account (>15s)
        s1 = TradingSnapshot(
            context=RuntimeTradingContext.local_paper(),
            connection=ConnectionSnapshot("LOCAL", True, True, 0, now),
            market=MarketSnapshot("BTC", "BTC/USDT", "PERPETUAL", "USDT", Decimal("100"), Decimal("100"), "LOCAL", now, 0.1),
            account=AccountSnapshot(Decimal("1000"), Decimal("0"), Decimal("1000"), Decimal("1000"), None, None, None, "USDT", source="TEST", timestamp=old_ts),
            positions=(),
            open_orders=(),
            reconciliation=ReconciliationSnapshot("OK", 0, now),
            risk=RiskSnapshot(True, ("VIEW", "OPEN_LONG", "REDUCE_ONLY"), ()),
            configured_strategy="MOMENTUM",
            effective_strategy="MOMENTUM",
            model_id=None,
            generated_at=now,
        ).validated()
        self.assertIs(s1.status, SnapshotStatus.INCONSISTENT)
        self.assertIn("ACCOUNT_DATA_STALE_OR_MISSING_TIMESTAMP", s1.errors)

        # 2. Unknown orders present
        s2 = TradingSnapshot(
            context=RuntimeTradingContext.local_paper(),
            connection=ConnectionSnapshot("LOCAL", True, True, 0, now),
            market=MarketSnapshot("BTC", "BTC/USDT", "PERPETUAL", "USDT", Decimal("100"), Decimal("100"), "LOCAL", now, 0.1),
            account=AccountSnapshot(Decimal("1000"), Decimal("0"), Decimal("1000"), Decimal("1000"), None, None, None, "USDT", source="TEST", timestamp=now),
            positions=(),
            open_orders=({"client_order_id": "ord_1", "status": "UNKNOWN"},),
            reconciliation=ReconciliationSnapshot("OK", 0, now),
            risk=RiskSnapshot(True, ("VIEW", "OPEN_LONG", "REDUCE_ONLY"), ()),
            configured_strategy="MOMENTUM",
            effective_strategy="MOMENTUM",
            model_id=None,
            generated_at=now,
        ).validated()
        self.assertIs(s2.status, SnapshotStatus.INCONSISTENT)
        self.assertIn("UNKNOWN_ORDERS_PRESENT", s2.errors)
        self.assertFalse(s2.risk.trading_enabled)
'''

BEHAVIOR_TESTS_PY = '''"""Behavioral offline tests for PaperBroker, BrainConnectivityService, TWSE calendar, and ReportSnapshot."""

from datetime import date, datetime, timezone
from decimal import Decimal
import unittest

import numpy as np
import pandas as pd


class TestAuditBehaviors(unittest.TestCase):
    def test_single_trading_mode_enum(self):
        from crypto_trading.runtime_context import TradingMode as RtMode
        from crypto_trading.settings import TradingMode as SetMode
        self.assertIs(RtMode, SetMode)

    def test_paper_broker_dynamic_short_pnl_and_reconcile(self):
        from crypto_trading.paper_broker import PaperBroker

        class DummyBot:
            symbol = "BTC/USDT"
            position = -0.02
            position_side = "SHORT"
            entry_price = 65000.0
            balance_usdt = 10000.0
            open_orders = ()

        broker = PaperBroker(DummyBot())
        now = datetime.now(timezone.utc)
        positions = broker.fetch_positions(Decimal("64000"), now=now)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].unrealized_pnl, Decimal("20.00"))
        acc = broker.account_snapshot(positions, now=now)
        self.assertEqual(acc.equity, Decimal("10020.00"))
        self.assertIsNone(acc.consistency_error())
        rec = broker.reconcile(now=now)
        self.assertEqual(rec.status, "OK")
        self.assertEqual(rec.last_completed_at, now)

    def test_brain_connectivity_service_and_queue_lifecycle(self):
        from worldquant_brain.brain_client import BrainClient
        from worldquant_brain.brain_job_queue import BrainJobQueue
        from worldquant_brain.connectivity_service import BrainAuthFailureClassifier, BrainConnectivityService

        self.assertEqual(
            BrainAuthFailureClassifier.classify(401, {"WWW-Authenticate": "persona"}, "Biometric"),
            "BRAIN_BIOMETRIC_REQUIRED",
        )
        self.assertEqual(
            BrainAuthFailureClassifier.classify(403, {"Content-Type": "text/html"}, "Cloudflare WAF"),
            "BRAIN_WAF_BLOCKED",
        )
        self.assertEqual(BrainAuthFailureClassifier.classify(429, {}, ""), "RATE_LIMITED")

        client = BrainClient(email="", password="")
        client.email = None
        client.password = None
        q = BrainJobQueue(client, autostart=False)
        self.assertFalse(q.worker_alive())
        q.start()
        self.assertTrue(q.worker_alive())
        svc = BrainConnectivityService(client, q, lambda: None)
        st = svc.get_status(local_alpha_worker_alive=True)
        self.assertTrue(st["service_available"])
        self.assertFalse(st["configured"])
        self.assertEqual(st["safe_error_code"], "BRAIN_CREDENTIALS_MISSING")
        q.stop()
        q.join(timeout=1.0)
        self.assertFalse(q.worker_alive())

    def test_twse_trading_calendar_holidays_and_bounds(self):
        from taiwan_stock.twse_calendar import is_twse_trading_day

        # 2026-02-17 is Lunar New Year (Tuesday) -> must be False!
        self.assertFalse(is_twse_trading_day(date(2026, 2, 17)))
        # 2026-11-16 is a normal Monday -> True
        self.assertTrue(is_twse_trading_day(date(2026, 11, 16)))
        # Outside verified calendar bounds -> fails closed False
        self.assertFalse(is_twse_trading_day(date(2035, 5, 2)))

    def test_feature_engineer_preserves_nan_correlation(self):
        from crypto_trading.feature_engineer import CryptoFeatureEngineer

        idx = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
        btc = pd.DataFrame({"Open": np.linspace(100, 110, 10), "High": np.linspace(101, 111, 10), "Low": np.linspace(99, 109, 10), "Close": np.linspace(100, 110, 10), "Volume": [100.0] * 10}, index=idx)
        eth = pd.DataFrame({"Open": [np.nan] * 10, "High": [np.nan] * 10, "Low": [np.nan] * 10, "Close": [np.nan] * 10, "Volume": [np.nan] * 10}, index=idx)
        cross = CryptoFeatureEngineer().compute_multi_asset_cross_features({"BTC/USDT": btc, "ETH/USDT": eth})
        if "ETH_corr_BTC_24" in cross.columns:
            self.assertTrue(cross["ETH_corr_BTC_24"].isna().all())


if __name__ == "__main__":
    unittest.main()
'''

MIGRATION_ENGINE_TESTS_PY = '"""Unit tests for the v5 transactional migration engine (dry-run, conflict, rollback, manifest recovery)."""\n\nimport json\nimport tempfile\nimport unittest\nfrom pathlib import Path\n\nimport scripts.apply_full_audit_bundle_v5 as mig\n\n\nclass TestMigrationEngine(unittest.TestCase):\n    def setUp(self):\n        self.tmp_dir = tempfile.TemporaryDirectory()\n        self.root = Path(self.tmp_dir.name)\n        self.manifest_path = self.root / ".patch_backup_manifest.json"\n        self.backups_dir = self.root / ".patch_backups"\n\n    def tearDown(self):\n        self.tmp_dir.cleanup()\n\n    def test_unknown_existing_file_causes_conflict(self):\n        target = self.root / "crypto_trading" / "runtime_context.py"\n        target.parent.mkdir(parents=True, exist_ok=True)\n        target.write_text("# manually created file\\n", encoding="utf-8")\n\n        manifest = {"files": {}, "runs": {}}\n        plan, has_err = mig.build_plan(manifest, base_root=self.root, subset_new={"crypto_trading/runtime_context.py": mig.RUNTIME_CONTEXT_PY}, subset_transforms={})\n        self.assertTrue(has_err)\n        self.assertEqual(plan["crypto_trading/runtime_context.py"][0], "CONFLICT")\n\n    def test_apply_idempotency_manual_edit_conflict_and_rollback_manifest(self):\n        # 1. Initial apply of a single new file + a transformed file\n        env_file = self.root / ".env.example"\n        env_file.write_text("BRAIN_EMAIL=\\nBRAIN_API_KEY=\\n", encoding="utf-8")\n\n        rc = mig.execute_migration(\n            apply_mode=True,\n            base_root=self.root,\n            manifest_path=self.manifest_path,\n            backups_dir=self.backups_dir,\n            subset_new={"crypto_trading/instruments.py": mig.INSTRUMENTS_PY},\n            subset_transforms={".env.example": mig._transform_env_example},\n            run_tests=False,\n        )\n        self.assertEqual(rc, 0)\n        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))\n        self.assertEqual(len(manifest["runs"]), 1)\n        run_id = next(iter(manifest["runs"].keys()))\n        self.assertEqual(manifest["runs"][run_id]["status"], "APPLIED")\n\n        # 2. Second apply is idempotent (ALREADY_APPLIED)\n        plan2, err2 = mig.build_plan(\n            manifest,\n            base_root=self.root,\n            subset_new={"crypto_trading/instruments.py": mig.INSTRUMENTS_PY},\n            subset_transforms={".env.example": mig._transform_env_example},\n        )\n        self.assertFalse(err2)\n        self.assertEqual(plan2["crypto_trading/instruments.py"][0], "ALREADY_APPLIED")\n        self.assertEqual(plan2[".env.example"][0], "ALREADY_APPLIED")\n\n        # 3. Manual modification triggers CONFLICT\n        (self.root / "crypto_trading" / "instruments.py").write_text("# user edit\\n", encoding="utf-8")\n        plan3, err3 = mig.build_plan(\n            manifest,\n            base_root=self.root,\n            subset_new={"crypto_trading/instruments.py": mig.INSTRUMENTS_PY},\n            subset_transforms={".env.example": mig._transform_env_example},\n        )\n        self.assertTrue(err3)\n        self.assertEqual(plan3["crypto_trading/instruments.py"][0], "CONFLICT")\n\n        # 4. Rollback restores both files AND manifest entries\n        rb_rc = mig.do_rollback(run_id, base_root=self.root, manifest_path=self.manifest_path)\n        self.assertEqual(rb_rc, 0)\n        self.assertFalse((self.root / "crypto_trading" / "instruments.py").exists())\n        self.assertEqual(env_file.read_text(encoding="utf-8"), "BRAIN_EMAIL=\\nBRAIN_API_KEY=\\n")\n        manifest_after = json.loads(self.manifest_path.read_text(encoding="utf-8"))\n        self.assertEqual(manifest_after["runs"][run_id]["status"], "ROLLED_BACK")\n        self.assertNotIn("crypto_trading/instruments.py", manifest_after["files"])\n\n\nif __name__ == "__main__":\n    unittest.main()\n'

NEW_FILES = {
    "crypto_trading/runtime_context.py": RUNTIME_CONTEXT_PY,
    "crypto_trading/instruments.py": INSTRUMENTS_PY,
    "crypto_trading/trading_snapshot.py": TRADING_SNAPSHOT_PY,
    "crypto_trading/exchange_normalizer.py": EXCHANGE_NORMALIZER_PY,
    "crypto_trading/paper_broker.py": PAPER_BROKER_PY,
    "worldquant_brain/connectivity_service.py": BRAIN_CONNECTIVITY_SERVICE_PY,
    "taiwan_stock/twse_calendar.py": TWSE_CALENDAR_PY,
    "web_ui/trading_snapshot_api.py": TRADING_SNAPSHOT_API_PY,
    "web_ui/static/static_snapshot_patch.js": STATIC_SNAPSHOT_PATCH_JS,
    "tests/test_runtime_truth.py": TEST_RUNTIME_TRUTH_PY,
    "tests/test_full_audit_integration.py": BEHAVIOR_TESTS_PY,
    "tests/test_migration_engine.py": MIGRATION_ENGINE_TESTS_PY,
}

TRANSFORMS = {
    "crypto_trading/settings.py": _transform_settings_dotenv,
    ".env.example": _transform_env_example,
    "pyproject.toml": _transform_pyproject,
    "crypto_trading/bot_runner.py": _transform_bot_runner,
    "crypto_trading/feature_engineer.py": _transform_feature_engineer,
    "worldquant_brain/brain_client.py": _transform_brain_client,
    "worldquant_brain/brain_job_queue.py": _transform_brain_job_queue,
    "worldquant_brain/alpha_miner.py": _transform_alpha_miner,
    "taiwan_stock/scheduler.py": _transform_scheduler,
    "taiwan_stock/report_generator.py": _transform_report_generator,
    "web_ui/server.py": _transform_server,
    "web_ui/static/index.html": _transform_index_html,
}


def build_plan(
    manifest: dict,
    *,
    base_root: Path | None = None,
    subset_new: dict[str, str] | None = None,
    subset_transforms: dict | None = None,
) -> tuple[dict[str, tuple[str, str]], bool]:
    root = base_root or ROOT
    new_map = NEW_FILES if subset_new is None else subset_new
    tx_map = TRANSFORMS if subset_transforms is None else subset_transforms

    plan: dict[str, tuple[str, str]] = {}
    has_err = False
    known_files = manifest.get("files", {})

    for rel, content in new_map.items():
        p = root / rel
        if not p.exists():
            plan[rel] = ("PATCHED", content)
        else:
            old_hash = _sha256(p.read_text(encoding="utf-8"))
            new_hash = _sha256(content)
            if old_hash == new_hash:
                plan[rel] = ("ALREADY_APPLIED", content)
            elif rel in known_files and known_files[rel].get("post_sha256") == old_hash:
                plan[rel] = ("PATCHED", content)
            else:
                plan[rel] = ("CONFLICT", "")
                has_err = True

    for rel, fn in tx_map.items():
        txt = _read(rel, base_root=root)
        if txt is None:
            plan[rel] = ("TARGET_NOT_FOUND", "")
            has_err = True
            continue
        old_hash = _sha256(txt)
        if rel in known_files and known_files[rel].get("post_sha256") and old_hash != known_files[rel]["post_sha256"]:
            plan[rel] = ("CONFLICT", "")
            has_err = True
            continue
        status, new_txt = fn(txt)
        if status in {"TARGET_NOT_FOUND", "CONFLICT"}:
            has_err = True
        plan[rel] = (status, new_txt)

    return plan, has_err


def do_rollback(run_id: str, *, base_root: Path | None = None, manifest_path: Path | None = None) -> int:
    root = base_root or ROOT
    mp = manifest_path or MANIFEST_PATH
    if not mp.exists():
        print("[rollback] No .patch_backup_manifest.json found.")
        return 1
    manifest = _load_manifest(mp)
    run_entry = manifest.get("runs", {}).get(run_id)
    if not run_entry:
        print(f"[rollback] Run ID '{run_id}' not found.")
        return 1

    files_rec = run_entry.get("files", run_entry) if isinstance(run_entry, dict) else {}
    for rel, info in files_rec.items():
        if not isinstance(info, dict):
            continue
        target = root / rel
        bpath = info.get("backup_path")
        prev_entry = info.get("prev_manifest_entry")
        if bpath:
            src = root / bpath
            if src.exists():
                _atomic_write_text(target, src.read_text(encoding="utf-8"))
                print(f"  [restored] {rel}")
        else:
            if target.exists():
                target.unlink()
                print(f"  [removed-created] {rel}")

        if prev_entry is not None:
            manifest.setdefault("files", {})[rel] = prev_entry
        else:
            manifest.get("files", {}).pop(rel, None)

    if isinstance(run_entry, dict) and "files" in run_entry:
        run_entry["status"] = "ROLLED_BACK"
        run_entry["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
    else:
        manifest["runs"][run_id] = {
            "status": "ROLLED_BACK",
            "rolled_back_at": datetime.now(timezone.utc).isoformat(),
            "files": files_rec,
        }
    _save_manifest(manifest, mp)
    print(f"[rollback] Completed rollback for run {run_id}.")
    return 0


def execute_migration(
    *,
    apply_mode: bool,
    base_root: Path | None = None,
    manifest_path: Path | None = None,
    backups_dir: Path | None = None,
    subset_new: dict[str, str] | None = None,
    subset_transforms: dict | None = None,
    run_tests: bool = True,
) -> int:
    root = base_root or ROOT
    mp = manifest_path or MANIFEST_PATH
    bdir = backups_dir or BACKUPS_DIR
    manifest = _load_manifest(mp)

    # Detect interrupted APPLYING runs on startup
    for r_id, r_info in manifest.get("runs", {}).items():
        if isinstance(r_info, dict) and r_info.get("status") == "APPLYING":
            print(f"[apply_full_audit_bundle_v5] INTERRUPTED RUN DETECTED: {r_id}. Run with --recover-interrupted or --rollback {r_id} first.")
            return 1

    plan, has_err = build_plan(manifest, base_root=root, subset_new=subset_new, subset_transforms=subset_transforms)
    print(f"[apply_full_audit_bundle_v5] Mode={'APPLY' if apply_mode else 'PLAN_DRY_RUN'}")
    for rel, (status, _) in plan.items():
        print(f"  - {rel}: {status}")

    if has_err:
        print("[apply_full_audit_bundle_v5] ABORTED: Conflict or missing target detected.")
        return 1

    if not apply_mode:
        print("[apply_full_audit_bundle_v5] Dry-run complete. Re-run with --apply to stage, compile, backup, and apply.")
        return 0

    to_write = {rel: content for rel, (st, content) in plan.items() if st == "PATCHED"}
    if not to_write:
        print("[apply_full_audit_bundle_v5] Nothing to change (all ALREADY_APPLIED).")
        return 0

    # Stage all outputs in temp dir and run compileall BEFORE touching real tree
    with tempfile.TemporaryDirectory() as stage_dir:
        stage_root = Path(stage_dir)
        for rel, content in to_write.items():
            _atomic_write_text(stage_root / rel, content)
        comp = subprocess.run([sys.executable, "-m", "compileall", "-q", str(stage_root)], capture_output=True, text=True)
        if comp.returncode != 0:
            print("[apply_full_audit_bundle_v5] STAGED COMPILEALL FAILED:", comp.stdout, comp.stderr)
            return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    run_files: dict[str, dict] = {}
    manifest.setdefault("runs", {})[run_id] = {
        "status": "APPLYING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "files": run_files,
    }
    _save_manifest(manifest, mp)

    for rel, content in to_write.items():
        target = root / rel
        prev_manifest_entry = manifest.get("files", {}).get(rel)
        if target.exists():
            old_txt = target.read_text(encoding="utf-8")
            bpath = bdir / run_id / rel
            bpath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, bpath)
            b_rel = str(bpath.relative_to(root))
            pre_h = _sha256(old_txt)
        else:
            b_rel = None
            pre_h = None
        _atomic_write_text(target, content)
        post_h = _sha256(content)
        manifest.setdefault("files", {})[rel] = {"pre_sha256": pre_h, "post_sha256": post_h, "backup_path": b_rel}
        run_files[rel] = {
            "pre_sha256": pre_h,
            "post_sha256": post_h,
            "backup_path": b_rel,
            "prev_manifest_entry": prev_manifest_entry,
        }

    # Post-apply compileall & targeted behavioral tests; auto-rollback on failure
    post_comp = subprocess.run([sys.executable, "-m", "compileall", "-q", str(root)], capture_output=True, text=True)
    if post_comp.returncode != 0:
        print("[apply_full_audit_bundle_v5] POST-APPLY COMPILEALL FAILED, AUTO-ROLLING BACK:", post_comp.stderr)
        do_rollback(run_id, base_root=root, manifest_path=mp)
        return 1

    if run_tests:
        test_cmd = [
            sys.executable,
            "-m",
            "unittest",
            "tests/test_runtime_truth.py",
            "tests/test_full_audit_integration.py",
            "tests/test_migration_engine.py",
        ]
        test_res = subprocess.run(test_cmd, cwd=root, capture_output=True, text=True)
        if test_res.returncode != 0:
            print("[apply_full_audit_bundle_v5] TARGETED TESTS FAILED, AUTO-ROLLING BACK:\n", test_res.stdout, test_res.stderr)
            do_rollback(run_id, base_root=root, manifest_path=mp)
            return 1

    manifest["runs"][run_id]["status"] = "APPLIED"
    manifest["runs"][run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    _save_manifest(manifest, mp)
    print(f"[apply_full_audit_bundle_v5] Applied and verified {len(to_write)} files cleanly (run_id={run_id}).")
    return 0


def _is_git_dirty() -> bool:
    if not (ROOT / ".git").exists():
        return False
    res = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return res.returncode == 0 and bool(res.stdout.strip())


def cleanup_redundant_artifacts(*, dry_run: bool = True, force: bool = False) -> int:
    if not dry_run and not force and _is_git_dirty():
        print("[cleanup] REFUSED: Git working tree is dirty. Commit/stash first or pass --force.")
        return 2
    candidates = [
        "temp_test_models",
        "scripts/apply_direct_fixes.py",
        "scripts/apply_cockpit_brain_fixes.py",
        "scripts/apply_full_audit_bundle.py",
        "scripts/apply_full_audit_bundle_v3.py",
        "scripts/apply_full_audit_bundle_v4.py",
    ]
    found = [ROOT / r for r in candidates if (ROOT / r).exists()]
    for c in list(ROOT.rglob("__pycache__")) + list(ROOT.rglob(".pytest_cache")):
        if c.exists():
            found.append(c)
    docs_dir = ROOT / "docs"
    if docs_dir.exists():
        for gdoc in docs_dir.glob("*.md.gdoc"):
            if (docs_dir / gdoc.name[:-5]).exists():
                found.append(gdoc)
    print(f"[cleanup] Mode={'DRY-RUN' if dry_run else 'EXECUTE'}, candidates={len(found)}")
    for p in found:
        rel = p.relative_to(ROOT)
        if dry_run:
            print(f"  [would remove] {rel}")
        else:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
            print(f"  [removed] {rel}")
    return 0


def main() -> int:
    if "--recover-interrupted" in sys.argv:
        manifest = _load_manifest()
        for r_id, r_info in list(manifest.get("runs", {}).items()):
            if isinstance(r_info, dict) and r_info.get("status") == "APPLYING":
                return do_rollback(r_id)
        print("[recover] No interrupted APPLYING runs found.")
        return 0

    if "--rollback" in sys.argv:
        idx = sys.argv.index("--rollback")
        if idx + 1 >= len(sys.argv):
            print("Usage: --rollback <run_id>")
            return 1
        return do_rollback(sys.argv[idx + 1])

    if "--cleanup" in sys.argv or "--execute-cleanup" in sys.argv:
        return cleanup_redundant_artifacts(
            dry_run=("--execute-cleanup" not in sys.argv),
            force=("--force" in sys.argv),
        )

    return execute_migration(apply_mode=("--apply" in sys.argv))


if __name__ == "__main__":
    raise SystemExit(main())
