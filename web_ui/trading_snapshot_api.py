"""Framework-neutral handler helpers for GET /api/trading/snapshot."""
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
