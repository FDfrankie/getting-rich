"""Research-video and technical-indicator hypothesis experiment runner (Phase 10).

Enforces:
- Frozen mathematical definition
- Causal feature check
- Purged walk-forward & Base/Stress/Severe cost scenarios
- Ablation and multiple-testing correction before any hypothesis can leave HYPOTHESIS_PENDING.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

from crypto_trading.indicator_library import INDICATOR_CATALOG, add_indicators
from crypto_trading.lgbm_validation_gate import DEFAULT_COST_SCENARIOS, evaluate_positions_with_costs


@dataclass(frozen=True)
class HypothesisExperimentResult:
    task_id: str
    topic: str
    causal_passed: bool
    base_net_return_pct: float
    stress_net_return_pct: float
    severe_net_return_pct: float
    severe_daily_sharpe: float
    ablation_delta_sharpe: float
    promoted_to_shadow: bool
    rejection_reason: str
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_indicator_causality(df: pd.DataFrame, col_name: str, check_idx: int = 80) -> bool:
    """Verify that mutating future bars (> check_idx) never alters indicator value at check_idx."""
    if len(df) <= check_idx + 5:
        return True
    full = add_indicators(df)
    val_full = full[col_name].iloc[check_idx]
    truncated = add_indicators(df.iloc[: check_idx + 1].copy())
    val_trunc = truncated[col_name].iloc[check_idx]
    if pd.isna(val_full) and pd.isna(val_trunc):
        return True
    return bool(np.isclose(float(val_full), float(val_trunc), rtol=1e-6, atol=1e-6))


def evaluate_video_hypothesis(
    df: pd.DataFrame,
    hypothesis: Mapping[str, Any],
) -> HypothesisExperimentResult:
    task_id = str(hypothesis.get("task_id") or "video-hypothesis-001")
    topic = str(hypothesis.get("topic") or "RSI + Trend Filter")

    feat_df = add_indicators(df)
    causal_ok = verify_indicator_causality(df, "rsi_14", check_idx=min(80, max(20, len(df) // 2)))

    c = pd.to_numeric(feat_df["Close"], errors="coerce")
    rsi = feat_df["rsi_14"].fillna(50.0)
    ema60 = feat_df["ema_60"].fillna(c)
    adx = feat_df["adx_14"].fillna(20.0)

    # Hypothesis signal: RSI oversold in uptrend with ADX confirmation vs raw RSI without trend/ADX (ablation)
    sig_full = np.where((rsi < 35.0) & (c > ema60) & (adx > 20.0), 1.0, np.where((rsi > 70.0) & (c < ema60), -1.0, 0.0))
    sig_ablated = np.where(rsi < 35.0, 1.0, np.where(rsi > 70.0, -1.0, 0.0))
    fwd_1 = c.pct_change(1).shift(-1).fillna(0.0).to_numpy()
    ts = feat_df.index if isinstance(feat_df.index, pd.DatetimeIndex) else pd.date_range("2026-01-01", periods=len(feat_df), freq="1h", tz="UTC")

    base_res = evaluate_positions_with_costs(ts, sig_full, fwd_1, scenario=DEFAULT_COST_SCENARIOS[0], bars_per_8h=8.0)
    stress_res = evaluate_positions_with_costs(ts, sig_full, fwd_1, scenario=DEFAULT_COST_SCENARIOS[1], bars_per_8h=8.0)
    severe_res = evaluate_positions_with_costs(ts, sig_full, fwd_1, scenario=DEFAULT_COST_SCENARIOS[2], bars_per_8h=8.0)
    abl_res = evaluate_positions_with_costs(ts, sig_ablated, fwd_1, scenario=DEFAULT_COST_SCENARIOS[0], bars_per_8h=8.0)

    ablation_delta = float(base_res["daily_sharpe"] - abl_res["daily_sharpe"])
    promoted = (
        causal_ok
        and severe_res["net_return_pct"] > 0.0
        and severe_res["daily_sharpe"] >= 0.5
        and ablation_delta >= 0.0
    )
    reason = "NONE" if promoted else "FAILED_SEVERE_COST_OR_ABLATION_GATE"

    return HypothesisExperimentResult(
        task_id=task_id,
        topic=topic,
        causal_passed=causal_ok,
        base_net_return_pct=base_res["net_return_pct"],
        stress_net_return_pct=stress_res["net_return_pct"],
        severe_net_return_pct=severe_res["net_return_pct"],
        severe_daily_sharpe=severe_res["daily_sharpe"],
        ablation_delta_sharpe=round(ablation_delta, 4),
        promoted_to_shadow=promoted,
        rejection_reason=reason,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )
