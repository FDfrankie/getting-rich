"""Strict chronological Walk-Forward & Purge/Embargo Guard for crypto strategy & ML validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True, slots=True)
class WalkForwardVerdict:
    passed: bool
    folds_evaluated: int
    mean_oos_sharpe: float
    worst_oos_drawdown: float
    reasons: tuple[str, ...]


def build_purged_walk_forward_folds(
    n_bars: int,
    *,
    train_size: int = 240,
    test_size: int = 60,
    purge_gap: int = 6,
    step_size: int | None = None,
) -> list[WalkForwardFold]:
    """Build strictly chronological non-overlapping OOS folds with a causal purge gap."""
    if n_bars <= 0 or train_size <= 0 or test_size <= 0:
        return []
    step = step_size if (step_size is not None and step_size > 0) else test_size
    folds: list[WalkForwardFold] = []
    idx = 0
    t_start = 0
    while True:
        tr_end = t_start + train_size
        te_start = tr_end + max(0, purge_gap)
        te_end = te_start + test_size
        if te_end > n_bars:
            break
        folds.append(
            WalkForwardFold(
                fold_index=idx,
                train_start=t_start,
                train_end=tr_end,
                test_start=te_start,
                test_end=te_end,
            )
        )
        idx += 1
        t_start += step
    return folds


def evaluate_walk_forward_oos(
    fold_returns: Sequence[Iterable[float]],
    *,
    min_folds: int = 3,
    min_mean_sharpe: float = 0.50,
    max_drawdown_limit: float = 0.25,
    periods_per_year: float = 8760.0,
) -> WalkForwardVerdict:
    """Validate out-of-sample fold return series against Sharpe and drawdown guardrails."""
    sharpes: list[float] = []
    drawdowns: list[float] = []
    reasons: list[str] = []

    for series in fold_returns:
        arr = np.asarray(list(series), dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 5:
            continue
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        sharpe = (mu / sd) * np.sqrt(periods_per_year) if sd > 1e-12 else 0.0
        equity = np.cumprod(1.0 + np.clip(arr, -0.95, 5.0))
        peak = np.maximum.accumulate(equity)
        dd = float(np.max((peak - equity) / np.maximum(peak, 1e-12)))
        sharpes.append(float(sharpe))
        drawdowns.append(dd)

    if len(sharpes) < min_folds:
        reasons.append(f"INSUFFICIENT_OOS_FOLDS:{len(sharpes)}<{min_folds}")

    mean_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
    worst_dd = float(np.max(drawdowns)) if drawdowns else 1.0

    if mean_sharpe < min_mean_sharpe:
        reasons.append(f"LOW_OOS_SHARPE:{mean_sharpe:.3f}<{min_mean_sharpe:.3f}")
    if worst_dd > max_drawdown_limit:
        reasons.append(f"EXCESSIVE_OOS_DRAWDOWN:{worst_dd:.3f}>{max_drawdown_limit:.3f}")

    return WalkForwardVerdict(
        passed=len(reasons) == 0,
        folds_evaluated=len(sharpes),
        mean_oos_sharpe=round(mean_sharpe, 4),
        worst_oos_drawdown=round(worst_dd, 4),
        reasons=tuple(reasons),
    )
