"""P54 Institutional Crypto Regime-Adaptive Alpha Router.

Combines 6 institutional alpha & microstructure engines (100% causal, zero lookahead):
  1. Kaufman Efficiency Ratio (ER) + Volatility Regime Router:
     - TREND regime (ER >= 0.30): VWAP Pullback-to-Trend + HTF Momentum + Fresh OI Continuation
     - SWEEP regime (ER < 0.24): Liquidity-Pool Stop-Hunt Sweep + CVD Divergence Reversal
     - CASH regime (dead chop or extreme panic): 0 trades, 0 fees
  2. Liquidity-Pool Stop-Hunt Sweep Detector (breaks rolling 20-bar swing low/high & reclaims inside)
  3. Cumulative Volume Delta (CVD) Divergence Detector (price lower-low + CVD higher-low = absorption)
  4. Real Perpetual Derivatives Filter (Funding Rate crowding block + OI-Price Quadrant confirmation)
  5. Cross-Asset BTC Lead-Lag Filter (optional causal BTC trend/OFI confirmation for ETH/SOL)
  6. Bayesian Conditional Probability Posterior Fusion (from ConditionalProbRLStrategy)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from crypto_trading.strategies.conditional_prob_rl_strategy import ConditionalProbRLStrategy
from crypto_trading.strategies.momentum_breakout import BaseStrategy


class InstitutionalAlphaRouter(BaseStrategy):
    """Regime-Adaptive Institutional Crypto Strategy supporting 1m / 3m / 15m / 1h USDT-M perpetuals."""

    def __init__(
        self,
        timeframe: str = "3m",
        fee_roundtrip: float = 0.0004,
        maker_mode: bool = True,
        er_window: int = 20,
        swing_window: int = 20,
        trend_er_threshold: float = 0.30,
        sweep_er_threshold: float = 0.26,
        crowded_funding_threshold: float = 0.0005,  # 5 bps / 8h
        partial_tp_atr_mult: float = 2.0,
        tp_atr_mult: float = 5.0,
        sl_atr_mult: float = 2.0,
        cooldown_bars: int | None = None,
        min_holding_bars: int | None = None,
    ) -> None:
        tf = str(timeframe or "3m").strip().lower()
        self.timeframe = tf
        self.preferred_timeframe = tf
        self.fee_roundtrip = float(fee_roundtrip)
        self.maker_mode = bool(maker_mode)
        self.er_window = max(10, int(er_window))
        self.swing_window = max(10, int(swing_window))
        self.trend_er_threshold = float(trend_er_threshold)
        self.sweep_er_threshold = float(sweep_er_threshold)
        self.crowded_funding_threshold = float(crowded_funding_threshold)
        self.partial_tp_atr_mult = float(partial_tp_atr_mult)
        self.tp_atr_mult = float(tp_atr_mult)
        self.sl_atr_mult = float(sl_atr_mult)

        default_cd = {"1m": 30, "3m": 20, "5m": 14, "15m": 8, "1h": 6}.get(tf, 20)
        default_mh = {"1m": 15, "3m": 10, "5m": 8, "15m": 5, "1h": 4}.get(tf, 10)
        self.cooldown_bars = int(cooldown_bars if cooldown_bars is not None else default_cd)
        self.min_holding_bars = int(min_holding_bars if min_holding_bars is not None else default_mh)
        self.cond_prob = ConditionalProbRLStrategy(
            timeframe=tf,
            fee_roundtrip=self.fee_roundtrip,
            maker_mode=self.maker_mode,
        )
        self.last_diagnostics: dict[str, Any] = {}

    def compute_microstructure_features(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        out = self.cond_prob.compute_regime_states(df)
        close = pd.to_numeric(out["Close"], errors="coerce")
        high = pd.to_numeric(out["High"], errors="coerce")
        low = pd.to_numeric(out["Low"], errors="coerce")
        vol = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0)
        ofi = pd.to_numeric(out["OFI"], errors="coerce").fillna(0.0)

        # 1. Kaufman Efficiency Ratio (strictly causal)
        net_change = (close - close.shift(self.er_window)).abs()
        sum_abs_change = close.diff().abs().rolling(self.er_window, min_periods=self.er_window).sum().replace(0, np.nan)
        er = (net_change / sum_abs_change).fillna(0.0).clip(0.0, 1.0)
        out["KAUFMAN_ER"] = er

        # 2. Cumulative Volume Delta (CVD) & CVD Divergence (rolling 12-bar causal)
        signed_vol = ofi * vol
        cvd_roll = signed_vol.rolling(30, min_periods=10).sum()
        prev_low_12 = low.shift(1).rolling(12, min_periods=6).min()
        prev_high_12 = high.shift(1).rolling(12, min_periods=6).max()
        prev_cvd_low_12 = cvd_roll.shift(1).rolling(12, min_periods=6).min()
        prev_cvd_high_12 = cvd_roll.shift(1).rolling(12, min_periods=6).max()

        # Bullish CVD divergence: price sweeps below prior 12-bar low, but CVD is strictly higher than prior 12-bar CVD low
        bull_cvd_div = (low < prev_low_12) & (cvd_roll > prev_cvd_low_12) & (close > low + 0.35 * (high - low))
        bear_cvd_div = (high > prev_high_12) & (cvd_roll < prev_cvd_high_12) & (close < high - 0.35 * (high - low))
        out["CVD_ROLL"] = cvd_roll.fillna(0.0)
        out["BULL_CVD_DIV"] = bull_cvd_div.fillna(False)
        out["BEAR_CVD_DIV"] = bear_cvd_div.fillna(False)

        # 3. Liquidity-Pool Stop-Hunt Sweep (breaks rolling N-bar swing extreme & closes back inside!)
        swing_low = low.shift(1).rolling(self.swing_window, min_periods=10).min()
        swing_high = high.shift(1).rolling(self.swing_window, min_periods=10).max()
        liq_sweep_bull = (low < swing_low) & (close > swing_low)
        liq_sweep_bear = (high > swing_high) & (close < swing_high)
        out["LIQ_SWEEP_BULL"] = liq_sweep_bull.fillna(False)
        out["LIQ_SWEEP_BEAR"] = liq_sweep_bear.fillna(False)

        # 4. Cross-Asset BTC Lead-Lag Confirmation (strictly causal)
        if btc_df is not None and len(btc_df) > 20 and "Close" in btc_df.columns:
            btc_c = pd.to_numeric(btc_df["Close"], errors="coerce")
            btc_ret5 = btc_c.pct_change(5).fillna(0.0)
            btc_ema_f = btc_c.ewm(span=12, adjust=False, min_periods=6).mean()
            btc_ema_s = btc_c.ewm(span=36, adjust=False, min_periods=12).mean()
            btc_dir = np.sign((btc_ema_f - btc_ema_s).fillna(0.0))
            if "OpenTime" in out.columns and "OpenTime" in btc_df.columns:
                tmp_btc = pd.DataFrame(
                    {
                        "OpenTime": pd.to_numeric(btc_df["OpenTime"], errors="coerce").astype("int64"),
                        "BTC_RET5": btc_ret5.to_numpy(dtype=np.float64),
                        "BTC_DIR": btc_dir.to_numpy(dtype=np.float64),
                    }
                ).sort_values("OpenTime")
                out_sorted = out.sort_values("OpenTime").reset_index(drop=True)
                merged = pd.merge_asof(out_sorted, tmp_btc, on="OpenTime", direction="backward")
                out["BTC_RET5"] = merged["BTC_RET5"].fillna(0.0).to_numpy()
                out["BTC_DIR"] = merged["BTC_DIR"].fillna(0.0).to_numpy()
            elif len(btc_df) == len(out):
                out["BTC_RET5"] = btc_ret5.to_numpy()
                out["BTC_DIR"] = btc_dir.to_numpy()
            else:
                out["BTC_RET5"] = 0.0
                out["BTC_DIR"] = 0.0
        else:
            out["BTC_RET5"] = 0.0
            out["BTC_DIR"] = 0.0

        # Defaults if derivatives columns were not pre-enriched
        if "FUNDING_RATE_8H" not in out.columns:
            out["FUNDING_RATE_8H"] = 0.0001
        if "OI_QUADRANT" not in out.columns:
            out["OI_QUADRANT"] = "NEUTRAL"
        if "TOP_LS_ZSCORE" not in out.columns:
            out["TOP_LS_ZSCORE"] = 0.0

        # Regime router label
        vol_reg = pd.to_numeric(out["VOL_REGIME"], errors="coerce").fillna(0.0)
        regime = np.where(
            (vol_reg < -0.25) | (vol_reg > 2.50),
            "CASH",
            np.where(er >= self.trend_er_threshold, "TREND", np.where(er <= self.sweep_er_threshold, "SWEEP", "MIXED")),
        )
        out["ALPHA_REGIME"] = regime
        return out

    def generate_signals(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        out = self.compute_microstructure_features(df, btc_df=btc_df)
        # Obtain causal Bayesian posterior from ConditionalProbRLStrategy
        cp_df = self.cond_prob.generate_signals(df)
        p_long = pd.to_numeric(cp_df["cond_prob_long"], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)
        p_short = pd.to_numeric(cp_df["cond_prob_short"], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)
        cp_sig = pd.to_numeric(cp_df["signal"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

        n = len(out)
        atr_pct = out["ATR_PCT"].to_numpy(dtype=np.float64)
        htf_z = out["HTF_TREND_Z"].to_numpy(dtype=np.float64)
        vwap_dev = out["VWAP_DEV_ATR"].to_numpy(dtype=np.float64)
        ofi = out["OFI"].to_numpy(dtype=np.float64)
        regime = out["ALPHA_REGIME"].tolist()
        bull_sweep = out["LIQ_SWEEP_BULL"].to_numpy(dtype=bool)
        bear_sweep = out["LIQ_SWEEP_BEAR"].to_numpy(dtype=bool)
        bull_cvd = out["BULL_CVD_DIV"].to_numpy(dtype=bool)
        bear_cvd = out["BEAR_CVD_DIV"].to_numpy(dtype=bool)
        funding = pd.to_numeric(out["FUNDING_RATE_8H"], errors="coerce").fillna(0.0001).to_numpy(dtype=np.float64)
        oi_quad = out["OI_QUADRANT"].tolist()
        btc_dir = pd.to_numeric(out["BTC_DIR"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        btc_ret5 = pd.to_numeric(out["BTC_RET5"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

        signals = np.zeros(n, dtype=float)
        pos = 0.0
        held = 0
        cool = self.cooldown_bars
        min_move_req = self.fee_roundtrip * 6.0  # Require >= 24 bps expected move on Maker

        for i in range(max(120, self.swing_window + 20), n):
            if pos != 0.0:
                held += 1
            if (pos == 0.0 and cool < self.cooldown_bars) or (pos != 0.0 and held < self.min_holding_bars):
                signals[i] = pos
                cool += 1
                continue

            # Exit on strong regime reversal after min holding period
            if pos > 0 and (htf_z[i] < -0.35 or (bear_sweep[i] and bear_cvd[i])):
                pos = 0.0
                held = 0
                cool = 1
                signals[i] = 0.0
                continue
            if pos < 0 and (htf_z[i] > 0.35 or (bull_sweep[i] and bull_cvd[i])):
                pos = 0.0
                held = 0
                cool = 1
                signals[i] = 0.0
                continue

            if pos != 0.0:
                signals[i] = pos
                cool += 1
                continue

            # Brand-new entry evaluation
            if regime[i] == "CASH" or (atr_pct[i] * self.tp_atr_mult) < min_move_req:
                signals[i] = 0.0
                cool += 1
                continue

            target = 0.0
            reg = regime[i]

            # Engine A: TREND Regime (VWAP Pullback-to-Trend + HTF Momentum + OFI + CondProb)
            if reg in ("TREND", "MIXED"):
                long_trend = (
                    htf_z[i] >= 0.35
                    and -1.5 <= vwap_dev[i] <= 0.85
                    and ofi[i] >= 0.05
                    and (p_long[i] >= 0.52 or cp_sig[i] > 0)
                )
                short_trend = (
                    htf_z[i] <= -0.35
                    and -0.85 <= vwap_dev[i] <= 1.5
                    and ofi[i] <= -0.05
                    and (p_short[i] >= 0.52 or cp_sig[i] < 0)
                )
                if long_trend:
                    target = 1.0
                elif short_trend:
                    target = -1.0

            # Engine B: SWEEP Regime (Liquidity-Pool Stop-Hunt + CVD Absorption Divergence)
            if target == 0.0 and reg in ("SWEEP", "MIXED"):
                if (bull_sweep[i] and (bull_cvd[i] or ofi[i] > 0.15)) and htf_z[i] > -0.60:
                    target = 0.85
                elif (bear_sweep[i] and (bear_cvd[i] or ofi[i] < -0.15)) and htf_z[i] < 0.60:
                    target = -0.85

            # Institutional Filter 1: Perpetual Funding Rate Crowding Block (He et al. 2026)
            if target > 0 and funding[i] > self.crowded_funding_threshold:
                target = 0.0
            elif target < 0 and funding[i] < -self.crowded_funding_threshold:
                target = 0.0

            # Institutional Filter 2: Cross-Asset BTC Lead-Lag Guard (block altcoin longs when BTC dumps)
            if target > 0 and (btc_dir[i] < 0 and btc_ret5[i] < -0.003):
                target = 0.0
            elif target < 0 and (btc_dir[i] > 0 and btc_ret5[i] > 0.003):
                target = 0.0

            # Institutional Filter 3: OI-Price Quadrant Conviction Scaling
            if target != 0.0:
                q = oi_quad[i]
                if target > 0:
                    if q in ("FRESH_LONG", "LIQ_FLUSH"):
                        target = min(1.30, target * 1.25)
                    elif q == "SHORT_COVER":
                        target *= 0.65
                else:
                    if q == "FRESH_SHORT":
                        target = max(-1.30, target * 1.25)

            if abs(target) >= 0.35:
                pos = round(float(target), 2)
                held = 0
                cool = 1
            else:
                cool += 1
            signals[i] = pos

        out["signal"] = signals
        out["cond_prob_long"] = p_long
        out["cond_prob_short"] = p_short
        if n > 0:
            self.last_diagnostics = {
                "timeframe": self.timeframe,
                "regime": regime[-1],
                "kaufman_er": round(float(out["KAUFMAN_ER"].iloc[-1]), 3),
                "oi_quadrant": oi_quad[-1],
                "funding_bps_8h": round(float(funding[-1] * 10000.0), 2),
                "signal": float(signals[-1]),
            }
        return out
