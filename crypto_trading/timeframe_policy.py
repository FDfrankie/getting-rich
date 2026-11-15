"""
Timeframe status and policy manager.
Governs timeframe operational states, including flagging 15s as EXPERIMENTAL_RESEARCH_ONLY
and strictly disallowing live trading without real aggTrade WebSocket aggregation.
"""
from __future__ import annotations

from enum import Enum
from typing import Tuple


class TimeframeStatus(str, Enum):
    PRODUCTION_READY = "PRODUCTION_READY"
    EXPERIMENTAL_RESEARCH_ONLY = "EXPERIMENTAL_RESEARCH_ONLY"
    DEPRECATED = "DEPRECATED"


class TimeframePolicy:
    """Manages operational restrictions per timeframe."""

    _TIMEFRAME_STATUS = {
        "15s": TimeframeStatus.EXPERIMENTAL_RESEARCH_ONLY,
        "1m": TimeframeStatus.PRODUCTION_READY,
        "3m": TimeframeStatus.PRODUCTION_READY,
        "5m": TimeframeStatus.PRODUCTION_READY,
        "15m": TimeframeStatus.PRODUCTION_READY,
        "1h": TimeframeStatus.PRODUCTION_READY,
        "1d": TimeframeStatus.PRODUCTION_READY,
    }

    @classmethod
    def get_status(cls, timeframe: str) -> TimeframeStatus:
        key = timeframe.strip().lower()
        return cls._TIMEFRAME_STATUS.get(key, TimeframeStatus.EXPERIMENTAL_RESEARCH_ONLY)

    @classmethod
    def can_live_trade(cls, timeframe: str, has_aggtrade_feed: bool = False) -> Tuple[bool, str]:
        status = cls.get_status(timeframe)
        if status == TimeframeStatus.EXPERIMENTAL_RESEARCH_ONLY:
            if not has_aggtrade_feed:
                return (
                    False,
                    f"Timeframe '{timeframe}' is EXPERIMENTAL_RESEARCH_ONLY: Live orders forbidden without real aggTrade feed aggregation."
                )
        return True, "OK"
