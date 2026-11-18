"""Leakage-safe bridge from completed BRAIN experiments to crypto research features.

BRAIN metrics are only aggregate research-context features. They never become an
order signal directly and are shifted by a configurable availability lag.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

class BrainFeatureBridge:
    def __init__(self, db_path: str | Path, lag: str = "1D") -> None:
        self.db_path, self.lag = str(db_path), pd.Timedelta(lag)

    def daily_context(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            try:
                df = pd.read_sql_query("""SELECT completed_at,sharpe,fitness,turnover,margin,
                    self_correlation,simulation_status FROM alpha_experiments
                    WHERE completed_at IS NOT NULL""", conn)
            except Exception:
                return pd.DataFrame()
        if df.empty: return df
        df["available_at"] = pd.to_datetime(df["completed_at"], utc=True) + self.lag
        df["date"] = df["available_at"].dt.floor("D")
        df["passed"] = df["simulation_status"].isin(["BRAIN_COMPLETE","IS_CHECK_PASSED","SUBMITTED"]).astype(int)
        return df.groupby("date", as_index=False).agg(
            brain_completed=("passed","size"), brain_pass_rate=("passed","mean"),
            brain_mean_sharpe=("sharpe","mean"), brain_mean_fitness=("fitness","mean"),
            brain_mean_turnover=("turnover","mean"), brain_mean_margin=("margin","mean"),
            brain_mean_self_corr=("self_correlation","mean"))

    def merge_asof(self, market: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        context = self.daily_context()
        if context.empty: return market.copy()
        left = market.copy(); left[timestamp_col] = pd.to_datetime(left[timestamp_col], utc=True)
        left = left.sort_values(timestamp_col); context = context.sort_values("date")
        return pd.merge_asof(left, context, left_on=timestamp_col, right_on="date", direction="backward")
