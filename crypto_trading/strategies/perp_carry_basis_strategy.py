"""Perpetual Futures Funding-Rate Carry & Basis Momentum Strategy.

Implements institutional crypto perpetual alpha (He, Manela, Ross & von Wachter; Coinbase Quant):
1. Funding Rate Carry / Contrarian Crowding:
   - Extremely high positive funding (> +0.025% / 8h) indicates crowded long leverage -> fade on momentum exhaustion.
   - Deep negative funding (< -0.015% / 8h) indicates crowded short squeeze risk -> long carry + squeeze capture.
2. Spot-Perp Basis & Open-Interest Momentum Proxy:
   - Combines funding-rate Z-score, VWAP premium/discount (basis proxy), and ADX/Supertrend regime confirmation.
3. Asymmetric Hysteresis & Minimum Holding Window:
   - Prevents fee churning and preserves positive net Sharpe after maker/taker fees and 8h funding settlements.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_trading.strategies.momentum_breakout import BaseStrategy


class PerpCarryBasisStrategy(BaseStrategy):
    """合約資金費率套利與基差動量策略 (Funding Rate Carry & Basis Momentum)"""

    def __init__(
        self,
        entry_z: float = 1.35,
        exit_z: float = 0.15,
        max_holding_bars: int = 8,
        crowded_funding_bps: float = 2.5,  # 2.5 bps per 8h (0.025%)
    ) -> None:
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.max_holding_bars = max(1, int(max_holding_bars))
        self.crowded_funding_bps = float(crowded_funding_bps)
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def estimate_funding_and_basis(df: pd.DataFrame, crowded_funding_bps: float = 2.5) -> pd.DataFrame:
        out = df.copy()
        close = pd.to_numeric(out["Close"], errors="coerce")
        high = pd.to_numeric(out["High"], errors="coerce")
        low = pd.to_numeric(out["Low"], errors="coerce")
        open_ = pd.to_numeric(out["Open"], errors="coerce")
        vol = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0)

        # Rolling VWAP as spot anchor & basis proxy
        tp = (high + low + close) / 3.0
        vwap24 = (tp * vol).rolling(24, min_periods=4).sum() / vol.rolling(24, min_periods=4).sum().replace(0, np.nan)
        basis_bps = ((close - vwap24) / vwap24.replace(0, np.nan) * 10000.0).fillna(0.0)

        # Real funding_rate when present; otherwise Binance clamped funding proxy (He et al., arXiv:2212.06888v7)
        # Phi(p) = clamp(iota - p, -gamma, +gamma) + p, with iota = 1.0 bps, gamma = 5.0 bps
        used_proxy = "funding_rate" not in out.columns
        if not used_proxy:
            funding_bps = pd.to_numeric(out["funding_rate"], errors="coerce").fillna(1.0) * 10000.0
        else:
            prem_bps = basis_bps.ewm(span=8, adjust=False).mean()
            iota_bps, gamma_bps = 1.0, 5.0
            funding_bps = (np.clip(iota_bps - prem_bps, -gamma_bps, gamma_bps) + prem_bps).clip(-75.0, 75.0).fillna(1.0)
        out["funding_is_proxy"] = used_proxy

        ret_1 = close.pct_change(1).fillna(0.0)
        ret_6 = close.pct_change(6).fillna(0.0)
        vol_24 = ret_1.rolling(24, min_periods=6).std(ddof=1).replace(0, np.nan)

        # Intrabar buying/selling pressure (Orderbook imbalance proxy)
        hl_range = (high - low).replace(0, np.nan)
        order_flow_imbalance = ((close - open_) / hl_range).fillna(0.0)

        # Composite Carry + Basis + Reversion Alpha Score
        # When funding is extreme positive & basis is overstretched, negative score (short carry + mean reversion)
        # When funding is negative or basis oversold with positive flow turn, positive score (long squeeze)
        raw_alpha = (
            -0.45 * (basis_bps / 25.0)
            - 0.30 * ((funding_bps - 1.0) / max(float(crowded_funding_bps), 0.5))
            - 0.25 * (ret_6 / (vol_24 * np.sqrt(6.0) + 1e-8)).fillna(0.0)
            + 0.15 * order_flow_imbalance.rolling(3, min_periods=1).mean()
        )

        # ATR for RiskManager compatibility
        prev_c = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_c).abs(), (low - prev_c).abs()))
        out["ATR"] = tr.rolling(14, min_periods=1).mean()
        out["basis_bps"] = basis_bps
        out["funding_bps_8h"] = funding_bps
        out["perp_carry_alpha"] = raw_alpha.fillna(0.0)
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.estimate_funding_and_basis(df, crowded_funding_bps=self.crowded_funding_bps)
        alpha = out["perp_carry_alpha"]
        win = min(168, max(24, len(out) // 3))
        rmean = alpha.rolling(win, min_periods=20).mean()
        rstd = alpha.rolling(win, min_periods=20).std(ddof=1).fillna(1.0) + 1e-8
        z = ((alpha - rmean) / rstd).fillna(0.0).to_numpy()
        min_edge_bps = float(getattr(self, "min_edge_bps", 10.0))
        edge_ok_arr = (
            (out["basis_bps"].abs() >= min_edge_bps)
            | ((out["funding_bps_8h"] - 1.0).abs() >= self.crowded_funding_bps)
        ).to_numpy()

        n = len(out)
        signals = np.zeros(n, dtype=float)
        cur = 0.0
        held = 0
        for i, val in enumerate(z):
            if cur == 0.0:
                if not edge_ok_arr[i]:
                    signals[i] = cur
                    continue
                if val > self.entry_z:
                    cur = 1.0
                    held = 1
                elif val < -self.entry_z:
                    cur = -1.0
                    held = 1
            elif cur == 1.0:
                held += 1
                if val < self.exit_z or held > self.max_holding_bars:
                    cur = -1.0 if val < -self.entry_z else 0.0
                    held = 1 if cur != 0.0 else 0
            elif cur == -1.0:
                held += 1
                if val > -self.exit_z or held > self.max_holding_bars:
                    cur = 1.0 if val > self.entry_z else 0.0
                    held = 1 if cur != 0.0 else 0
            signals[i] = cur

        out["signal"] = signals
        out["perp_carry_z"] = z
        if n > 0:
            self.last_diagnostics = {
                "basis_bps": round(float(out["basis_bps"].iloc[-1]), 2),
                "funding_bps_8h": round(float(out["funding_bps_8h"].iloc[-1]), 3),
                "carry_zscore": round(float(z[-1]), 3),
                "signal": int(signals[-1]),
            }
        return out
