"""Immutable runtime context for one trading execution session."""
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
