"""Strict OOS validation & cost-stress gate for pooled crypto LightGBM models.

Implements:
1. Global timestamp split (60% Train / 20% Val / 20% Test) with purge & embargo.
2. Purged Walk-Forward (5 folds) over global timestamps.
3. Leave-One-Asset-Out (LOAO) cross-asset generalization check.
4. Realistic Turnover & 3-tier Cost Stress Test (Base / Stress 1.5x / Severe 2.0x + Funding),
   reporting un-inflated per-bar & daily-aggregated Sharpe alongside annualized metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostScenario:
    name: str
    fee_bps_per_side: float
    slippage_bps_per_side: float
    funding_bps_per_8h: float = 0.0

    @property
    def total_bps_per_side(self) -> float:
        return self.fee_bps_per_side + self.slippage_bps_per_side

    def funding_per_bar(self, bars_per_8h: float = 480.0) -> float:
        return (self.funding_bps_per_8h / 10000.0) / max(bars_per_8h, 1.0)


DEFAULT_COST_SCENARIOS: tuple[CostScenario, ...] = (
    # Base: 2 bps blended maker/taker fee + 1 bps spread/slippage = 3 bps/side (6 bps round-trip)
    CostScenario(name="base", fee_bps_per_side=2.0, slippage_bps_per_side=1.0, funding_bps_per_8h=1.0),
    # Stress: 1.5x base cost
    CostScenario(name="stress_1_5x", fee_bps_per_side=3.0, slippage_bps_per_side=1.5, funding_bps_per_8h=2.0),
    # Severe: 2.0x base cost + 95th percentile funding (3 bps / 8h)
    CostScenario(name="severe_2_0x_funding", fee_bps_per_side=4.0, slippage_bps_per_side=2.0, funding_bps_per_8h=3.0),
)


def compute_rank_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) < 5 or np.std(preds) < 1e-12 or np.std(targets) < 1e-12:
        return 0.0
    ic = pd.Series(preds).corr(pd.Series(targets), method="spearman")
    return float(ic) if pd.notna(ic) else 0.0


def evaluate_positions_with_costs(
    timestamps: pd.DatetimeIndex | pd.Series,
    preds: np.ndarray,
    forward_returns_1bar: np.ndarray,
    *,
    threshold: float = 0.0,
    scenario: CostScenario = DEFAULT_COST_SCENARIOS[0],
    bars_per_8h: float = 480.0,
) -> dict[str, Any]:
    """Simulate positions on 1-bar forward returns with turnover, fees, slippage, and funding."""
    n = len(preds)
    if n < 10:
        return {
            "scenario": scenario.name,
            "bars": int(n),
            "trades": 0,
            "total_turnover": 0.0,
            "net_return_pct": 0.0,
            "per_bar_sharpe": 0.0,
            "daily_sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
        }

    pos = np.where(preds > threshold, 1.0, np.where(preds < -threshold, -1.0, 0.0))
    prev_pos = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - prev_pos)
    cost_rate = scenario.total_bps_per_side / 10000.0
    funding_rate = scenario.funding_per_bar(bars_per_8h=bars_per_8h)

    gross_ret = pos * forward_returns_1bar
    tx_cost = turnover * cost_rate
    funding_cost = np.abs(pos) * funding_rate
    net_ret = gross_ret - tx_cost - funding_cost

    trades = int(np.sum(turnover > 0))
    total_turnover = float(np.sum(turnover))

    per_bar_std = float(np.std(net_ret, ddof=1)) if n > 1 else 0.0
    per_bar_sharpe = float(np.mean(net_ret) / (per_bar_std + 1e-12))

    # Aggregate to daily returns to eliminate 1m autocorrelation & sqrt(525600) inflation
    ts_index = pd.to_datetime(timestamps, utc=True)
    daily_series = pd.Series(net_ret, index=ts_index).resample("1D").sum()
    if len(daily_series) >= 5 and daily_series.std(ddof=1) > 1e-12:
        daily_sharpe = float((daily_series.mean() / daily_series.std(ddof=1)) * np.sqrt(365.0))
    else:
        daily_sharpe = 0.0

    equity = np.cumprod(1.0 + np.clip(net_ret, -0.95, 5.0))
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity / (running_max + 1e-12)) - 1.0
    max_dd_pct = float(abs(np.min(drawdowns)) * 100.0)
    net_return_pct = float((equity[-1] - 1.0) * 100.0)

    active_mask = np.abs(pos) > 0
    win_rate_pct = float(np.mean(net_ret[active_mask] > 0) * 100.0) if np.any(active_mask) else 0.0

    return {
        "scenario": scenario.name,
        "bars": int(n),
        "trades": trades,
        "total_turnover": round(total_turnover, 2),
        "net_return_pct": round(net_return_pct, 4),
        "per_bar_sharpe": round(per_bar_sharpe, 6),
        "daily_sharpe": round(daily_sharpe, 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "win_rate_pct": round(win_rate_pct, 2),
    }


def make_global_timestamp_splits(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    purge_bars: int = 15,
    embargo_bars: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split multi-asset panel strictly by unique sorted timestamps with purge + embargo."""
    unique_ts = np.sort(pd.to_datetime(df[timestamp_col], utc=True).unique())
    n_ts = len(unique_ts)
    if n_ts < 200:
        raise ValueError(f"Insufficient unique timestamps for global split: {n_ts}")

    gap = max(0, purge_bars) + max(0, embargo_bars)
    train_end = int(n_ts * train_frac)
    val_start = train_end + gap
    val_end = int(n_ts * (train_frac + val_frac))
    test_start = val_end + gap

    if val_start >= val_end or test_start >= n_ts:
        raise ValueError("Purge + embargo gap exceeds split boundaries.")

    ts_series = pd.to_datetime(df[timestamp_col], utc=True)
    train_df = df[ts_series <= unique_ts[train_end - 1]].copy()
    val_df = df[(ts_series >= unique_ts[val_start]) & (ts_series <= unique_ts[val_end - 1])].copy()
    test_df = df[ts_series >= unique_ts[test_start]].copy()
    return train_df, val_df, test_df


def make_purged_walk_forward_folds(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    n_folds: int = 5,
    purge_bars: int = 15,
    embargo_bars: int = 60,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create expanding-window global-timestamp walk-forward folds with purge & embargo."""
    unique_ts = np.sort(pd.to_datetime(df[timestamp_col], utc=True).unique())
    n_ts = len(unique_ts)
    fold_size = n_ts // (n_folds + 1)
    if fold_size <= purge_bars + embargo_bars:
        raise ValueError("Dataset too small for purged walk-forward folds.")

    ts_series = pd.to_datetime(df[timestamp_col], utc=True)
    gap = max(0, purge_bars) + max(0, embargo_bars)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for fold in range(1, n_folds + 1):
        train_end_idx = fold * fold_size
        test_start_idx = train_end_idx + gap
        test_end_idx = min(n_ts, (fold + 1) * fold_size)
        if test_start_idx >= test_end_idx:
            continue
        train_mask = ts_series <= unique_ts[train_end_idx - 1]
        test_mask = (ts_series >= unique_ts[test_start_idx]) & (ts_series <= unique_ts[test_end_idx - 1])
        folds.append((df[train_mask].copy(), df[test_mask].copy()))
    return folds


def run_validation_gate(
    panel_df: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable[[], Any],
    *,
    target_col: str = "target_ret",
    forward_1bar_col: str = "fwd_ret_1",
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    purge_bars: int = 15,
    embargo_bars: int = 60,
    scenarios: tuple[CostScenario, ...] = DEFAULT_COST_SCENARIOS,
) -> dict[str, Any]:
    """Execute Global Split, Purged Walk-Forward, LOAO, and Cost Stress Tests."""
    df = panel_df.sort_values([timestamp_col, symbol_col]).reset_index(drop=True)

    # 1. Global Timestamp Split (60/20/20)
    train_df, val_df, test_df = make_global_timestamp_splits(
        df, timestamp_col=timestamp_col, purge_bars=purge_bars, embargo_bars=embargo_bars
    )
    model_global = model_factory()
    model_global.fit(train_df[feature_cols].to_numpy(), train_df[target_col].to_numpy())
    val_preds = model_global.predict(val_df[feature_cols].to_numpy())
    test_preds = model_global.predict(test_df[feature_cols].to_numpy())

    global_val_ic = compute_rank_ic(val_preds, val_df[target_col].to_numpy())
    global_test_ic = compute_rank_ic(test_preds, test_df[target_col].to_numpy())

    # Evaluate global test across cost scenarios (aggregate per timestamp portfolio)
    test_eval_df = test_df[[timestamp_col, symbol_col, forward_1bar_col]].copy()
    test_eval_df["pred"] = test_preds
    cost_results: dict[str, Any] = {}
    for sc in scenarios:
        # Average across symbols per scenario to avoid double-counting simultaneous bars
        per_sym = [
            evaluate_positions_with_costs(
                sub[timestamp_col],
                sub["pred"].to_numpy(),
                sub[forward_1bar_col].to_numpy(),
                scenario=sc,
            )
            for _, sub in test_eval_df.groupby(symbol_col)
        ]
        cost_results[sc.name] = {
            "avg_net_return_pct": round(float(np.mean([r["net_return_pct"] for r in per_sym])), 4),
            "avg_daily_sharpe": round(float(np.mean([r["daily_sharpe"] for r in per_sym])), 4),
            "avg_max_drawdown_pct": round(float(np.mean([r["max_drawdown_pct"] for r in per_sym])), 4),
            "total_trades": int(sum(r["trades"] for r in per_sym)),
            "per_symbol": {sym: r for sym, r in zip(sorted(test_eval_df[symbol_col].unique()), per_sym)},
        }

    # 2. Purged Walk-Forward (5 Folds)
    wf_folds = make_purged_walk_forward_folds(
        df, timestamp_col=timestamp_col, n_folds=5, purge_bars=purge_bars, embargo_bars=embargo_bars
    )
    wf_results = []
    for idx, (f_train, f_test) in enumerate(wf_folds, start=1):
        m = model_factory()
        m.fit(f_train[feature_cols].to_numpy(), f_train[target_col].to_numpy())
        p = m.predict(f_test[feature_cols].to_numpy())
        ic = compute_rank_ic(p, f_test[target_col].to_numpy())
        cost_base = evaluate_positions_with_costs(
            f_test[timestamp_col], p, f_test[forward_1bar_col].to_numpy(), scenario=scenarios[0]
        )
        wf_results.append(
            {
                "fold": idx,
                "train_rows": len(f_train),
                "test_rows": len(f_test),
                "rank_ic": round(ic, 6),
                "base_daily_sharpe": cost_base["daily_sharpe"],
                "base_net_return_pct": cost_base["net_return_pct"],
            }
        )

    # 3. Leave-One-Asset-Out (LOAO)
    loao_results = {}
    symbols = sorted(df[symbol_col].unique())
    for held_out in symbols:
        tr = df[df[symbol_col] != held_out]
        te = df[df[symbol_col] == held_out]
        m = model_factory()
        m.fit(tr[feature_cols].to_numpy(), tr[target_col].to_numpy())
        p = m.predict(te[feature_cols].to_numpy())
        ic = compute_rank_ic(p, te[target_col].to_numpy())
        base_eval = evaluate_positions_with_costs(
            te[timestamp_col], p, te[forward_1bar_col].to_numpy(), scenario=scenarios[0]
        )
        loao_results[held_out] = {
            "rank_ic": round(ic, 6),
            "base_daily_sharpe": base_eval["daily_sharpe"],
            "base_net_return_pct": base_eval["net_return_pct"],
        }

    # Governance decision gate
    severe_name = scenarios[-1].name
    severe_res = cost_results[severe_name]
    loao_ics = [v["rank_ic"] for v in loao_results.values()]

    gate_checks = {
        "global_test_rank_ic_ge_0_015": bool(global_test_ic >= 0.015),
        "global_test_base_daily_sharpe_ge_1_0": bool(cost_results[scenarios[0].name]["avg_daily_sharpe"] >= 1.0),
        "loao_all_positive_ic": bool(all(ic > 0.0 for ic in loao_ics) and (np.mean(loao_ics) >= 0.01)),
        "severe_cost_positive_return": bool(severe_res["avg_net_return_pct"] > 0.0),
        "severe_cost_daily_sharpe_ge_0_5": bool(severe_res["avg_daily_sharpe"] >= 0.5),
    }
    passed_all = bool(all(gate_checks.values()))

    return {
        "gate_passed": passed_all,
        "recommended_stage": "PAPER_CANDIDATE" if passed_all else "SHADOW",
        "approved_for_order_entry": False,  # Always require explicit human sign-off for live orders
        "gate_checks": gate_checks,
        "global_split": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "purge_bars": purge_bars,
            "embargo_bars": embargo_bars,
            "val_rank_ic": round(global_val_ic, 6),
            "test_rank_ic": round(global_test_ic, 6),
            "cost_scenarios": cost_results,
        },
        "purged_walk_forward": wf_results,
        "leave_one_asset_out": loao_results,
    }
