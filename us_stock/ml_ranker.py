"""US Equity Cross-Sectional Multi-Factor Ranker & BRAIN Local Screener Wide-Panel Bridge."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class USStockMLRanker:
    """PIT-safe US equity ranker using 20-day excess returns and 21-day embargo."""

    TARGET_COLUMN = "target_fwd_20d_excess"
    EMBARGO_TRADING_DAYS = 21

    def __init__(self) -> None:
        self.feature_cols: list[str] = []
        self.model: Any = None

    @staticmethod
    def extract_price_features(df: pd.DataFrame, pit_fundamentals: pd.DataFrame | None = None) -> pd.DataFrame:
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        c = pd.to_numeric(out["Close"], errors="coerce")
        v = pd.to_numeric(out["Volume"], errors="coerce")
        ret = c.pct_change(fill_method=None)

        out["mom_12_1"] = c.shift(21).div(c.shift(252)).sub(1.0)
        out["mom_3m"] = c.pct_change(63, fill_method=None)
        out["rev_1m"] = -c.pct_change(21, fill_method=None)
        out["vol_20d"] = ret.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252.0)
        out["vol_60d"] = ret.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(252.0)
        out["vol_scaled_mom"] = out["mom_12_1"] / (out["vol_60d"] + 1e-6)
        out["adv_20d_log"] = np.log1p((c * v).rolling(20, min_periods=10).mean().clip(lower=0.0))

        pos_63 = (ret > 0).rolling(63, min_periods=40).mean()
        neg_63 = (ret < 0).rolling(63, min_periods=40).mean()
        out["fip_63d"] = np.sign(out["mom_3m"]).fillna(0.0) * (pos_63 - neg_63)

        if pit_fundamentals is not None and not pit_fundamentals.empty:
            fcols = [
                col
                for col in ("gross_profitability", "cash_accrual_quality", "operating_margin")
                if col in pit_fundamentals.columns
            ]
            if fcols:
                f_aligned = pit_fundamentals[fcols].sort_index()
                idx_name = out.index.name or "index"
                f_idx_name = f_aligned.index.name or "index"
                merged = pd.merge_asof(
                    out.reset_index().rename(columns={idx_name: "_dt"}),
                    f_aligned.reset_index().rename(columns={f_idx_name: "_dt"}),
                    on="_dt",
                    direction="backward",
                ).set_index("_dt")
                out = merged

        out["raw_fwd_20d"] = c.shift(-20).div(c).sub(1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def prepare_panel(
        self,
        price_map: Mapping[str, pd.DataFrame],
        fundamentals_map: Mapping[str, pd.DataFrame] | None = None,
        sector_map: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        panels: list[pd.DataFrame] = []
        fmap = fundamentals_map or {}
        smap = sector_map or {}
        for sym, df in price_map.items():
            if df is None or len(df) < 260:
                continue
            p = self.extract_price_features(df, fmap.get(sym))
            p["symbol"] = sym
            p["sector"] = smap.get(sym, "UNKNOWN")
            panels.append(p)
        if not panels:
            raise ValueError("No valid US equity data for panel construction")

        combined = pd.concat(panels, axis=0, sort=False).sort_index(kind="mergesort")
        for col in ("mom_12_1", "vol_scaled_mom", "fip_63d", "gross_profitability", "cash_accrual_quality"):
            if col in combined.columns:
                combined[f"sec_rank_{col}"] = combined.groupby([combined.index, "sector"])[col].rank(pct=True)
        cs_med = combined.groupby(level=0)["raw_fwd_20d"].transform("median")
        combined[self.TARGET_COLUMN] = combined["raw_fwd_20d"] - cs_med
        return combined

    @staticmethod
    def build_brain_screener_wide_panel(price_map: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Build the exact wide-table dict[str, DataFrame] expected by worldquant_brain/local_cs_screener.py."""
        fields_map = {
            "close": "Close",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
        }
        wide: dict[str, pd.DataFrame] = {}
        for key, col in fields_map.items():
            series_dict = {}
            for sym, df in price_map.items():
                if df is not None and not df.empty and col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    s.index = pd.to_datetime(df.index)
                    series_dict[sym] = s
            if series_dict:
                wide[key] = pd.DataFrame(series_dict).sort_index()
        if "close" in wide:
            wide["returns"] = wide["close"].pct_change(fill_method=None)
            if "high" in wide and "low" in wide:
                wide["vwap"] = (wide["high"] + wide["low"] + wide["close"]) / 3.0
        return wide

    @staticmethod
    def export_panel_for_brain_screener(
        price_map: Mapping[str, pd.DataFrame],
        out_path: str | Path,
    ) -> Path:
        """Export wide-table US panel as .pkl compatible with LocalCrossSectionalScreener (or .parquet if requested)."""
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        wide = USStockMLRanker.build_brain_screener_wide_panel(price_map)
        if target.suffix.lower() == ".parquet":
            stacked = pd.concat(wide, axis=1)
            stacked.to_parquet(target, index=True)
        else:
            pd.to_pickle(wide, target)
        return target
