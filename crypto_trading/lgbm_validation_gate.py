"""Strict OOS validation & cost-stress gate for pooled crypto LightGBM / Rank-IC models.

Implements:
1. Global timestamp split (60% Train / 20% Val / 20% Test) with purge & embargo.
2. Purged Walk-Forward (5 folds) over global timestamps.
3. Leave-One-Asset-Out (LOAO) cross-asset generalization check.
4. Realistic Turnover & 3-tier Cost Stress Test (Base / Stress 1.5x / Severe 2.0x + Funding),
   supporting both raw threshold and causal Z-score hysteresis execution filters.
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
    CostScenario(name="base", fee_bps_per_side=2.0, slippage_bps_per_side=1.0, funding_bps_per_8h=1.0),
    CostScenario(name="stress_1_5x", fee_bps_per_side=3.0, slippage_bps_per_side=1.5, funding_bps_per_8h=2.0),
    CostScenario(name="severe_2_0x_funding", fee_bps_per_side=4.0, slippage_bps_per_side=2.0, funding_bps_per_8h=3.0),
)


def compute_rank_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) < 5 or np.std(preds) < 1e-12 or np.std(targets) < 1e-12:
        return 0.0
    ic = pd.Series(preds).corr(pd.Series(targets), method="spearman")
    return float(ic) if pd.notna(ic) else 0.0


def compute_causal_hysteresis_positions(
    preds: np.ndarray,
    *,
    entry_z: float = 1.8,
    exit_z: float = -0.5,
    max_holding_bars: int = 4,
    min_warmup: int = 20,
) -> np.ndarray:
    """Convert raw model predictions into low-turnover positions using causal expanding Z-score hysteresis."""
    n = len(preds)
    if n == 0:
        return np.zeros(0, dtype=float)
    s = pd.Series(np.asarray(preds, dtype=float))
    rmean = s.expanding(min_periods=min_warmup).mean()
    rstd = s.expanding(min_periods=min_warmup).std(ddof=1).fillna(1.0) + 1e-8
    z = ((s - rmean) / rstd).fillna(0.0).to_numpy()

    pos = np.zeros(n, dtype=float)
    cur = 0.0
    held = 0
    for i, val in enumerate(z):
        if cur == 0.0:
            if val > entry_z:
                cur = 1.0
                held = 1
            elif val < -entry_z:
                cur = -1.0
                held = 1
        elif cur == 1.0:
            held += 1
            if val < exit_z or held > max_holding_bars:
                cur = -1.0 if val < -entry_z else 0.0
                held = 1 if cur != 0.0 else 0
        elif cur == -1.0:
            held += 1
            if val > -exit_z or held > max_holding_bars:
                cur = 1.0 if val > entry_z else 0.0
                held = 1 if cur != 0.0 else 0
        pos[i] = cur
    return pos


class ICStabilityLinearWrapper:
    """Wraps a base regression estimator with causal in-fold Rank-IC feature stability weighting."""

    def __init__(self, base_factory: Callable[[], Any] | None = None, min_ic: float = 0.015, blend_alpha: float = 0.85):
        self.base_factory = base_factory
        self.min_ic = float(min_ic)
        self.blend_alpha = float(blend_alpha)
        self.base_model_: Any = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ICStabilityLinearWrapper":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = len(X_arr)
        split = max(10, int(n * 0.75))
        X1, y1 = X_arr[:split], y_arr[:split]
        X2, y2 = X_arr[split:], y_arr[split:]

        self.mean_ = np.nanmean(X_arr, axis=0)
        self.std_ = np.nanstd(X_arr, axis=0) + 1e-8
        Xz = np.clip((np.nan_to_num(X_arr) - self.mean_) / self.std_, -3.5, 3.5)

        weights = np.zeros(X_arr.shape[1], dtype=float)
        for j in range(X_arr.shape[1]):
            ic1 = compute_rank_ic(X1[:, j], y1)
            ic2 = compute_rank_ic(X2[:, j], y2) if len(X2) >= 10 else ic1
            if ic1 * ic2 > 0 and abs(ic1) >= self.min_ic and abs(ic2) >= self.min_ic * 0.5:
                weights[j] = 0.5 * (ic1 + ic2)
        if np.sum(np.abs(weights)) < 1e-9:
            for j in range(X_arr.shape[1]):
                weights[j] = compute_rank_ic(X_arr[:, j], y_arr)
        norm = np.linalg.norm(weights) + 1e-8
        self.coef_ = weights / norm

        if self.base_factory is not None:
            try:
                self.base_model_ = self.base_factory()
                self.base_model_.fit(Xz, y_arr)
            except Exception:
                self.base_model_ = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        Xz = np.clip((np.nan_to_num(X_arr) - self.mean_) / self.std_, -3.5, 3.5)
        linear_pred = Xz @ self.coef_
        if self.base_model_ is not None and self.blend_alpha < 1.0:
            try:
                base_pred = np.asarray(self.base_model_.predict(Xz), dtype=float)
                b_std = np.std(base_pred) + 1e-8
                l_std = np.std(linear_pred) + 1e-8
                return self.blend_alpha * (linear_pred / l_std) + (1.0 - self.blend_alpha) * (base_pred / b_std)
            except Exception:
                pass
        return linear_pred


def evaluate_positions_with_costs(
    timestamps: pd.DatetimeIndex | pd.Series,
    preds: np.ndarray,
    forward_returns_1bar: np.ndarray,
    *,
    threshold: float = 0.0,
    scenario: CostScenario = DEFAULT_COST_SCENARIOS[0],
    bars_per_8h: float = 480.0,
    use_hysteresis: bool = False,
    entry_z: float = 1.8,
    exit_z: float = -0.5,
    max_holding_bars: int = 4,
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

    if use_hysteresis:
        pos = compute_causal_hysteresis_positions(
            preds, entry_z=entry_z, exit_z=exit_z, max_holding_bars=max_holding_bars
        )
    else:
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
    use_hysteresis: bool = False,
    entry_z: float = 1.8,
    exit_z: float = -0.5,
    max_holding_bars: int = 4,
    bars_per_8h: float = 480.0,
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

    test_eval_df = test_df[[timestamp_col, symbol_col, forward_1bar_col]].copy()
    test_eval_df["pred"] = test_preds
    cost_results: dict[str, Any] = {}
    for sc in scenarios:
        per_sym = [
            evaluate_positions_with_costs(
                sub[timestamp_col],
                sub["pred"].to_numpy(),
                sub[forward_1bar_col].to_numpy(),
                scenario=sc,
                use_hysteresis=use_hysteresis,
                entry_z=entry_z,
                exit_z=exit_z,
                max_holding_bars=max_holding_bars,
                bars_per_8h=bars_per_8h,
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
            f_test[timestamp_col],
            p,
            f_test[forward_1bar_col].to_numpy(),
            scenario=scenarios[0],
            use_hysteresis=use_hysteresis,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding_bars=max_holding_bars,
            bars_per_8h=bars_per_8h,
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
            te[timestamp_col],
            p,
            te[forward_1bar_col].to_numpy(),
            scenario=scenarios[0],
            use_hysteresis=use_hysteresis,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding_bars=max_holding_bars,
            bars_per_8h=bars_per_8h,
        )
        loao_results[held_out] = {
            "rank_ic": round(ic, 6),
            "base_daily_sharpe": base_eval["daily_sharpe"],
            "base_net_return_pct": base_eval["net_return_pct"],
        }

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
    passed_all = all(gate_checks.values())

    return {
        "gate_passed": passed_all,
        "recommended_stage": "PAPER_CANDIDATE" if passed_all else "SHADOW",
        "approved_for_order_entry": False,
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
