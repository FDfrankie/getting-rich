"""Deduplicated, rate-limited AlertManager for risk halts, reconciliation drift, and BRAIN events."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class AlertEvent:
    category: str
    severity: AlertSeverity
    message: str
    timestamp: float
    details: dict[str, Any]


class AlertManager:
    """Prevents log/webhook flooding via per-category cooldowns while keeping recent alert history."""

    def __init__(
        self,
        *,
        cooldown_seconds: float = 60.0,
        max_history: int = 200,
        sink: Callable[[AlertEvent], None] | None = None,
    ) -> None:
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._history: deque[AlertEvent] = deque(maxlen=max(10, int(max_history)))
        self._last_sent: dict[str, float] = {}
        self._sink = sink

    def emit(
        self,
        category: str,
        message: str,
        *,
        severity: AlertSeverity = AlertSeverity.WARNING,
        details: dict[str, Any] | None = None,
        now: float | None = None,
        force: bool = False,
    ) -> bool:
        ts = time.time() if now is None else float(now)
        key = f"{severity.value}:{category}"
        last = self._last_sent.get(key)
        if not force and last is not None and (ts - last) < self.cooldown_seconds:
            return False

        ev = AlertEvent(
            category=category,
            severity=severity,
            message=message,
            timestamp=ts,
            details=dict(details or {}),
        )
        self._last_sent[key] = ts
        self._history.append(ev)
        if self._sink is not None:
            self._sink(ev)
        return True

    def recent_alerts(self, limit: int = 50) -> list[AlertEvent]:
        items = list(self._history)
        return items[-max(1, int(limit)) :]
