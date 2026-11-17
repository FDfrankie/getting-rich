"""
Conditional Probability + Technical Indicators Strategy (ConditionalProbTechStrategy).
Combines:
  1. Strictly causal historical conditional probability P(NetWin | TechnicalState)
     using Bayesian shrinkage toward prior win rate when state sample count is small.
  2. Real-time technical indicator requirements (EMA trend, RSI zone, BB %B, Volume Z-score).
  3. Asymmetric hysteresis thresholds (entry_prob vs exit_prob) to extend holding time
     and eliminate excessive micro-churning fees.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from crypto_trading.strategies.momentum_breakout import BaseStrategy


class ConditionalProbTechStrategy(BaseStrategy):
    """過往條件機率 + 即時技術指標要件 + 非對稱遲滯持倉策略"""

    def __init__(
        self,
        horizon: int = 3,
        entry_prob: float = 0.535,
        exit_prob: float = 0.480,
        min_ev: float = 0.0001,
        fee_roundtrip: float = 0.0002,
        prior_strength: float = 25.0,
        min_history_bars: int = 120,
        warmup_fraction: float = 0.0,
    ) -> None:
        self.horizon = max(1, int(horizon))
        self.entry_prob = float(entry_prob)
        self.exit_prob = float(exit_prob)
        self.min_ev = float(min_ev)
        self.fee_roundtrip = float(fee_roundtrip)
        self.prior_strength = float(prior_strength)
        self.min_history_bars = max(40, int(min_history_bars))
        self.warmup_fraction = float(warmup_fraction)
        self.last_state_summary: dict[str, Any] = {}

    @staticmethod
    def compute_technical_states(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = pd.to_numeric(out["Close"], errors="coerce")
        high = pd.to_numeric(out["High"], errors="coerce")
        low = pd.to_numeric(out["Low"], errors="coerce")
        vol = pd.to_numeric(out["Volume"], errors="coerce")

        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        ma20 = close.rolling(20, min_periods=20).mean()
        sd20 = close.rolling(20, min_periods=20).std(ddof=1)

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Bollinger %B & Volume Z-score
        bb_pct_b = (close - (ma20 - 2.0 * sd20)) / (4.0 * sd20).replace(0, np.nan)
        vol_mean = vol.rolling(24, min_periods=24).mean()
        vol_std = vol.rolling(24, min_periods=24).std(ddof=1)
        vol_z = (vol - vol_mean) / vol_std.replace(0, np.nan)

        # ATR for risk_manager compatibility
        prev_c = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_c).abs(), (low - prev_c).abs()))
        out["ATR"] = tr.rolling(14, min_periods=14).mean()
        out["RSI_14"] = rsi
        out["BB_PCT_B"] = bb_pct_b
        out["VOL_Z"] = vol_z
        out["EMA_SPREAD"] = (ema12 - ema26) / close.replace(0, np.nan)

        # Discretize technical conditions into interpretable state keys
        trend = np.where((ema12 > ema26) & (close > ma20), "UP", np.where((ema12 < ema26) & (close < ma20), "DOWN", "FLAT"))
        rsi_zone = np.where(
            rsi < 35,
            "OS",
            np.where(rsi > 65, "OB", np.where(rsi >= 50, "BULL", "BEAR")),
        )
        bb_zone = np.where(bb_pct_b > 0.8, "HI", np.where(bb_pct_b < 0.2, "LO", "MID"))
        vol_zone = np.where(vol_z > 0.5, "HV", "NV")

        out["state_fine"] = [f"{t}|{r}|{b}|{v}" for t, r, b, v in zip(trend, rsi_zone, bb_zone, vol_zone)]
        out["state_coarse"] = [f"{t}|{r}" for t, r in zip(trend, rsi_zone)]
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.compute_technical_states(df)
        close = pd.to_numeric(out["Close"], errors="coerce").to_numpy(dtype=np.float64)
        fine_states = out["state_fine"].tolist()
        coarse_states = out["state_coarse"].tolist()
        ema_spread = out["EMA_SPREAD"].to_numpy(dtype=np.float64)
        rsi_arr = out["RSI_14"].to_numpy(dtype=np.float64)

        n = len(out)
        h = self.horizon
        signals = np.zeros(n, dtype=float)
        prob_long_arr = np.full(n, 0.5, dtype=float)
        prob_short_arr = np.full(n, 0.5, dtype=float)
        ev_long_arr = np.zeros(n, dtype=float)
        ev_short_arr = np.zeros(n, dtype=float)

        # Strictly causal expanding statistics:
        # At bar i, only bars j <= i - h have realized horizon returns (close[j + h] / close[j] - 1)!
        fine_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])   # [n, wins_long, wins_short, sum_ret]
        coarse_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        global_n = 0.0
        global_wl = 0.0
        global_ws = 0.0
        global_ret = 0.0

        warmup_idx = max(self.min_history_bars, int(n * self.warmup_fraction))
        pos = 0

        for i in range(n):
            # 1. Update causal history with bar j = i - h whose future return is now realized at bar i
            j = i - h
            if j >= 26 and not np.isnan(close[j]) and not np.isnan(close[i]) and close[j] > 0:
                fwd_ret = (close[i] - close[j]) / close[j]
                win_l = 1.0 if fwd_ret > self.fee_roundtrip else 0.0
                win_s = 1.0 if fwd_ret < -self.fee_roundtrip else 0.0

                fs = fine_states[j]
                cs = coarse_states[j]
                fine_stats[fs][0] += 1.0
                fine_stats[fs][1] += win_l
                fine_stats[fs][2] += win_s
                fine_stats[fs][3] += fwd_ret

                coarse_stats[cs][0] += 1.0
                coarse_stats[cs][1] += win_l
                coarse_stats[cs][2] += win_s
                coarse_stats[cs][3] += fwd_ret

                global_n += 1.0
                global_wl += win_l
                global_ws += win_s
                global_ret += fwd_ret

            if i < warmup_idx or np.isnan(ema_spread[i]) or np.isnan(rsi_arr[i]):
                signals[i] = 0.0
                continue

            # 2. Compute hierarchical Bayesian-shrunk conditional probability at bar i
            p0_l = (global_wl / global_n) if global_n > 0 else 0.5
            p0_s = (global_ws / global_n) if global_n > 0 else 0.5
            ev0 = (global_ret / global_n) if global_n > 0 else 0.0

            cs_i = coarse_states[i]
            cn, cwl, cws, cret = coarse_stats[cs_i]
            alpha_c = self.prior_strength
            pc_l = (cwl + alpha_c * p0_l) / (cn + alpha_c)
            pc_s = (cws + alpha_c * p0_s) / (cn + alpha_c)
            evc = (cret + alpha_c * ev0) / (cn + alpha_c)

            fs_i = fine_states[i]
            fn, fwl, fws, fret = fine_stats[fs_i]
            alpha_f = self.prior_strength * 0.6
            p_long = (fwl + alpha_f * pc_l) / (fn + alpha_f)
            p_short = (fws + alpha_f * pc_s) / (fn + alpha_f)
            ev_long = (fret + alpha_f * evc) / (fn + alpha_f) - self.fee_roundtrip
            ev_short = -(fret + alpha_f * evc) / (fn + alpha_f) - self.fee_roundtrip

            prob_long_arr[i] = p_long
            prob_short_arr[i] = p_short
            ev_long_arr[i] = ev_long
            ev_short_arr[i] = ev_short

            # 3. Technical confirmation + Asymmetric Hysteresis (solves low holding rate)
            tech_long_ok = (ema_spread[i] > -0.002) and (rsi_arr[i] < 78.0)
            tech_short_ok = (ema_spread[i] < 0.002) and (rsi_arr[i] > 22.0)

            enter_long = (p_long >= self.entry_prob) and (ev_long >= self.min_ev) and tech_long_ok
            enter_short = (p_short >= self.entry_prob) and (ev_short >= self.min_ev) and tech_short_ok

            if pos == 0:
                if enter_long and not enter_short:
                    pos = 1
                elif enter_short and not enter_long:
                    pos = -1
            elif pos == 1:
                if enter_short:
                    pos = -1
                elif p_long < self.exit_prob:
                    pos = 0
            elif pos == -1:
                if enter_long:
                    pos = 1
                elif p_short < self.exit_prob:
                    pos = 0

            signals[i] = float(pos)

        out["signal"] = signals
        out["cond_prob_long"] = prob_long_arr
        out["cond_prob_short"] = prob_short_arr
        out["cond_ev_long"] = ev_long_arr
        out["cond_ev_short"] = ev_short_arr

        if n > 0:
            last_fs = fine_states[-1]
            self.last_state_summary = {
                "state": last_fs,
                "samples": int(fine_stats[last_fs][0]),
                "prob_long": round(float(prob_long_arr[-1]), 4),
                "prob_short": round(float(prob_short_arr[-1]), 4),
                "ev_long_bps": round(float(ev_long_arr[-1] * 10000.0), 2),
                "ev_short_bps": round(float(ev_short_arr[-1] * 10000.0), 2),
                "signal": int(signals[-1]),
            }
        return out
