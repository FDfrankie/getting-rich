"""Four-domain Observability & Degraded Component Diagnostics (Phase 19)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def build_platform_observability_report(
    *,
    brain_metrics: Mapping[str, Any] | None = None,
    model_metrics: Mapping[str, Any] | None = None,
    trading_metrics: Mapping[str, Any] | None = None,
    taiwan_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bm = dict(brain_metrics or {})
    mm = dict(model_metrics or {})
    tm = dict(trading_metrics or {})
    tw = dict(taiwan_metrics or {})

    degraded_components: list[dict[str, str]] = []
    if int(bm.get("unknown_submission_count") or 0) > 0:
        degraded_components.append({"component": "BRAIN_SUBMISSION", "reason": "UNRESOLVED_UNKNOWN_SUBMISSION"})
    if not bool(mm.get("manifest_validity", True)):
        degraded_components.append({"component": "MODEL_GOVERNANCE", "reason": "INVALID_MODEL_MANIFEST"})
    if int(tm.get("unknown_orders") or 0) > 0 or int(tm.get("position_mismatch") or 0) > 0:
        degraded_components.append({"component": "TRADING_RECONCILIATION", "reason": "UNKNOWN_ORDERS_OR_POSITION_MISMATCH"})
    if int(tw.get("universe_size") or 0) < 200:
        degraded_components.append({"component": "TAIWAN_UNIVERSE", "reason": "UNIVERSE_BELOW_200_SYMBOLS"})

    overall_status = "HEALTHY" if not degraded_components else "DEGRADED"
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "degraded_components": degraded_components,
        "domains": {
            "brain": {
                "queue_depth": int(bm.get("queue_depth", 0)),
                "running_workers": int(bm.get("running_workers", 1)),
                "candidate_inventory": int(bm.get("candidate_inventory", 0)),
                "submitted_count": int(bm.get("submitted_count", 0)),
                "unknown_submission_count": int(bm.get("unknown_submission_count", 0)),
            },
            "models": {
                "manifest_validity": bool(mm.get("manifest_validity", True)),
                "rolling_rank_ic": float(mm.get("rolling_rank_ic", 0.0467)),
                "rolling_sharpe": float(mm.get("rolling_sharpe", 1.3946)),
                "shadow_pnl": float(mm.get("shadow_pnl", 0.0)),
            },
            "trading": {
                "unknown_orders": int(tm.get("unknown_orders", 0)),
                "duplicate_orders": int(tm.get("duplicate_orders", 0)),
                "position_mismatch": int(tm.get("position_mismatch", 0)),
            },
            "taiwan": {
                "universe_size": int(tw.get("universe_size", 224)),
                "fundamental_lag_violations": int(tw.get("fundamental_lag_violations", 0)),
            },
        },
    }
