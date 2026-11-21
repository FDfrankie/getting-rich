"""Cross-Sectional Crypto Alpha: BTC-Beta-Neutral Residual Momentum + Funding/OI Crowding + Vol-Targeted Sizing."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


class CrossSectionalCryptoAlpha:
    """Multi-symbol cross-sectional factor engine for liquid perpetual futures."""

    def __init__(
        self,
        *,
        benchmark_symbol: str = "BTC/USDT",
        beta_window: int = 72,
        mom_window: int = 168,
        funding_z_window: int = 168,
        target_ann_vol: float = 0.35,
        bars_per_year: float = 8760.0,
        max_weight_per_asset: float = 0.25,
        long_short_quantile: float = 0.30,
    ) -> None:
        self.benchmark_symbol = benchmark_symbol
        self.beta_window = int(beta_window)
        self.mom_window = int(mom_window)
        self.funding_z_window = int(funding_z_window)
        self.target_ann_vol = float(target_ann_vol)
        self.bars_per_year = float(bars_per_year)
        self.max_weight_per_asset = float(max_weight_per_asset)
        self.long_short_quantile = float(long_short_quantile)

    def compute_target_weights(self, ohlcv_map: Mapping[str, pd.DataFrame]) -> dict[str, float]:
        """Return volatility-scaled, dollar-neutral target weights in [-max_w, +max_w] using ONLY causal data."""
        closes: dict[str, pd.Series] = {}
        fundings: dict[str, pd.Series] = {}
        oi_chg: dict[str, pd.Series] = {}

        for sym, df in ohlcv_map.items():
            if df is None or len(df) < max(self.beta_window, self.mom_window) + 5:
                continue
            c = pd.to_numeric(df["Close"], errors="coerce")
            if c.isna().tail(10).any():
                continue
            closes[sym] = c
            if "funding_rate" in df.columns:
                fr = pd.to_numeric(df["funding_rate"], errors="coerce")
                fr_z = (fr - fr.rolling(self.funding_z_window, min_periods=24).mean()) / (
                    fr.rolling(self.funding_z_window, min_periods=24).std(ddof=1) + 1e-8
                )
                fundings[sym] = fr_z
            if "open_interest" in df.columns:
                oi = pd.to_numeric(df["open_interest"], errors="coerce")
                oi_chg[sym] = oi.pct_change(24).replace([np.inf, -np.inf], np.nan)

        if len(closes) < 3:
            return {sym: 0.0 for sym in ohlcv_map}

        price_panel = pd.DataFrame(closes).sort_index().ffill(limit=2)
        ret_panel = price_panel.pct_change().dropna(how="all")
        bench_col = self.benchmark_symbol if self.benchmark_symbol in ret_panel.columns else ret_panel.columns[0]
        bench_ret = ret_panel[bench_col]
        bench_var = bench_ret.rolling(self.beta_window, min_periods=24).var(ddof=1) + 1e-12

        resid_mom: dict[str, float] = {}
        realized_vol: dict[str, float] = {}
        for sym in price_panel.columns:
            r = ret_panel[sym]
            cov = r.rolling(self.beta_window, min_periods=24).cov(bench_ret)
            beta = (cov / bench_var).clip(-3.0, 3.0).shift(1).fillna(1.0)
            resid = r - beta * bench_ret
            vol = float(r.tail(self.beta_window).std(ddof=1) * np.sqrt(self.bars_per_year))
            if not np.isfinite(vol) or vol <= 1e-6:
                continue
            rm = float(resid.tail(self.mom_window).sum()) / (
                vol / np.sqrt(self.bars_per_year / self.mom_window) + 1e-8
            )
            if np.isfinite(rm):
                resid_mom[sym] = rm
                realized_vol[sym] = vol

        if len(resid_mom) < 3:
            return {sym: 0.0 for sym in ohlcv_map}

        scores = pd.Series(resid_mom).rank(pct=True) - 0.5
        if len(fundings) >= 3:
            last_fz = pd.Series(
                {s: float(v.iloc[-1]) for s, v in fundings.items() if s in scores.index and np.isfinite(v.iloc[-1])}
            )
            if len(last_fz) >= 3:
                crowd_score = -(last_fz.rank(pct=True) - 0.5)
                if len(oi_chg) >= 3:
                    last_oi = pd.Series(
                        {s: float(v.iloc[-1]) for s, v in oi_chg.items() if s in crowd_score.index and np.isfinite(v.iloc[-1])}
                    )
                    if len(last_oi) >= 3:
                        oi_boost = 1.0 + 0.25 * np.sign(last_oi.reindex(crowd_score.index).fillna(0.0))
                        crowd_score = crowd_score * oi_boost
                scores = 0.65 * scores + 0.35 * crowd_score.reindex(scores.index).fillna(0.0)

        q_hi = scores.quantile(1.0 - self.long_short_quantile)
        q_lo = scores.quantile(self.long_short_quantile)
        raw_w: dict[str, float] = {}
        for sym, sc in scores.items():
            if sc >= q_hi and sc > 0:
                raw_w[sym] = 1.0 / realized_vol[sym]
            elif sc <= q_lo and sc < 0:
                raw_w[sym] = -1.0 / realized_vol[sym]
            else:
                raw_w[sym] = 0.0

        w_ser = pd.Series(raw_w)
        long_sum = w_ser[w_ser > 0].sum()
        short_sum = abs(w_ser[w_ser < 0].sum())
        if long_sum > 0:
            w_ser[w_ser > 0] = 0.5 * w_ser[w_ser > 0] / long_sum
        if short_sum > 0:
            w_ser[w_ser < 0] = 0.5 * w_ser[w_ser < 0] / short_sum

        sub_ret = ret_panel[w_ser.index].tail(self.beta_window).fillna(0.0)
        cov_mat = sub_ret.cov().to_numpy() * self.bars_per_year
        port_vol = float(np.sqrt(max(w_ser.to_numpy() @ cov_mat @ w_ser.to_numpy(), 1e-12)))
        scale = min(1.5, self.target_ann_vol / port_vol)
        w_final = (w_ser * scale).clip(-self.max_weight_per_asset, self.max_weight_per_asset)
        return {sym: round(float(w_final.get(sym, 0.0)), 4) for sym in ohlcv_map}
