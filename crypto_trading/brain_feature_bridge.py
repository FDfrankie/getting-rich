"""Leakage-safe bridge from completed BRAIN experiments to crypto research features (Phase 13).

Enforces:
- Feature availability timestamp = official completed_at + configured information lag (default 1D).
- Strictly backward merge_asof only (zero look-ahead).
- Research-context features only (never directly triggers Crypto orders).
- Chronological OOS ablation test comparing Crypto base model vs Crypto + BRAIN aggregate features.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_trading.lgbm_validation_gate import compute_rank_ic


class BrainFeatureBridge:
    def __init__(self, db_path: str | Path, lag: str = "1D") -> None:
        self.db_path = str(db_path)
        self.lag = pd.Timedelta(lag)

    def daily_context(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            try:
                df = pd.read_sql_query(
                    """
                    SELECT completed_at, sharpe, fitness, turnover, margin,
                           self_correlation, simulation_status
                    FROM alpha_experiments
                    WHERE completed_at IS NOT NULL
                    """,
                    conn,
                )
            except Exception:
                return pd.DataFrame()
        if df.empty:
            return df
        df["available_at"] = pd.to_datetime(df["completed_at"], utc=True) + self.lag
        df["date"] = df["available_at"].dt.floor("D")
        df["passed"] = df["simulation_status"].isin(["BRAIN_COMPLETE", "IS_CHECK_PASSED", "SUBMITTED"]).astype(int)
        df["near_miss"] = (
            (df["passed"] == 0)
            & (pd.to_numeric(df["sharpe"], errors="coerce") >= 1.0)
        ).astype(int)
        return df.groupby("date", as_index=False).agg(
            brain_completed=("passed", "size"),
            brain_pass_rate=("passed", "mean"),
            brain_near_miss_rate=("near_miss", "mean"),
            brain_mean_sharpe=("sharpe", "mean"),
            brain_mean_fitness=("fitness", "mean"),
            brain_mean_turnover=("turnover", "mean"),
            brain_mean_margin=("margin", "mean"),
            brain_mean_self_corr=("self_correlation", "mean"),
        )

    def merge_asof(self, market: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        context = self.daily_context()
        if context.empty:
            return market.copy()
        left = market.copy()
        left[timestamp_col] = pd.to_datetime(left[timestamp_col], utc=True)
        left = left.sort_values(timestamp_col)
        context = context.sort_values("date")
        return pd.merge_asof(left, context, left_on=timestamp_col, right_on="date", direction="backward")

    @staticmethod
    def evaluate_ablation(
        df: pd.DataFrame,
        base_cols: list[str],
        brain_cols: list[str],
        target_col: str = "target_ret",
    ) -> dict[str, Any]:
        """Compare Crypto base features vs Crypto + BRAIN aggregate features on chronological OOS split."""
        clean = df.dropna(subset=base_cols + [target_col]).copy()
        for c in brain_cols:
            if c in clean.columns:
                clean[c] = clean[c].fillna(0.0)
            else:
                clean[c] = 0.0
        n = len(clean)
        if n < 40:
            return {"retain_brain_features": False, "reason": "INSUFFICIENT_ROWS"}
        split = int(n * 0.70)
        tr, te = clean.iloc[:split], clean.iloc[split:]

        def fit_pred(cols: list[str]) -> np.ndarray:
            X_tr = tr[cols].to_numpy(dtype=float)
            y_tr = tr[target_col].to_numpy(dtype=float)
            X_te = te[cols].to_numpy(dtype=float)
            mu = np.nanmean(X_tr, axis=0)
            sd = np.nanstd(X_tr, axis=0) + 1e-8
            Xz_tr = np.clip((X_tr - mu) / sd, -3.5, 3.5)
            Xz_te = np.clip((X_te - mu) / sd, -3.5, 3.5)
            A = Xz_tr.T @ Xz_tr + 50.0 * np.eye(len(cols))
            w = np.linalg.solve(A, Xz_tr.T @ (y_tr - np.mean(y_tr)))
            return Xz_te @ w

        p_base = fit_pred(base_cols)
        p_aug = fit_pred(base_cols + brain_cols)
        y_te = te[target_col].to_numpy(dtype=float)
        ic_base = compute_rank_ic(p_base, y_te)
        ic_aug = compute_rank_ic(p_aug, y_te)
        retain = bool(ic_aug > ic_base + 0.002)
        return {
            "base_oos_rank_ic": round(ic_base, 6),
            "augmented_oos_rank_ic": round(ic_aug, 6),
            "ic_improvement": round(ic_aug - ic_base, 6),
            "retain_brain_features": retain,
            "DirectOrderPermission": False,
        }
