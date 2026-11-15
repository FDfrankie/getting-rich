"""
Crypto Regime Filter (Hierarchical Regime Engine).
Provides high-level (1d/1h) trend and volatility regime classification
to govern and filter micro execution (15s/1m/5m).

Regime types:
- 'BULL_TREND': High-level bullish trend, allow long execution, restrict shorting.
- 'BEAR_TREND': High-level bearish trend, restrict long execution, allow defensive shorting / flat.
- 'CHOP_RANGE': High-level range-bound sideways, compress position sizing, require higher breakout hurdle.
- 'HIGH_VOL_SHOCK': High volatility spike regime, trigger circuit breaker caution.
"""

from typing import Any

import numpy as np
import pandas as pd


class HierarchicalRegimeFilter:
    def __init__(self, high_tf: str = "1h", low_tf: str = "1m"):
        self.high_tf = high_tf
        self.low_tf = low_tf

    def detect_regime(self, df_high: pd.DataFrame) -> dict[str, Any]:
        """
        Input df_high: DataFrame of 1h or 1d OHLCV bars.
        Returns regime status, score (-100 to +100), and permission for micro trade direction.
        """
        if df_high is None or len(df_high) < 30:
            return {
                "regime": "CHOP_RANGE",
                "score": 0,
                "allow_long": True,
                "allow_short": True,
                "sizing_multiplier": 0.5,
                "reason": "High-level data insufficient, fallback to conservative range sizing.",
            }

        close = df_high["Close"]
        high = df_high["High"]
        low = df_high["Low"]

        # 1. 均線層級
        ma10 = close.rolling(10).mean().iloc[-1]
        ma30 = close.rolling(30).mean().iloc[-1]
        latest_c = close.iloc[-1]

        # 2. 趨勢與乖離
        trend_up = latest_c > ma10 > ma30
        trend_down = latest_c < ma10 < ma30

        # 3. 波動度與 ATR 異常
        tr = np.maximum(
            high - low,
            np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()),
        )
        atr14 = tr.rolling(14).mean().iloc[-1]
        atr_pct = (atr14 / latest_c) * 100

        # ADX 代理 (趨勢強度)
        mom20 = (latest_c / close.iloc[-20] - 1.0) * 100 if len(close) >= 20 else 0.0

        if atr_pct > 4.5:
            regime = "HIGH_VOL_SHOCK"
            score = -20
            allow_long = False
            allow_short = False
            mult = 0.0  # 熔斷觀望
            reason = f"高階市場波動異常劇烈 (ATR: {atr_pct:.2f}%)，微觀高頻策略強制熔斷靜默。"
        elif trend_up and mom20 > 0.5:
            regime = "BULL_TREND"
            score = 80
            allow_long = True
            allow_short = False
            mult = 1.0
            reason = "高階時框多頭格局確立，允許微觀時框順勢做多，禁止逆勢做空。"
        elif trend_down and mom20 < -0.5:
            regime = "BEAR_TREND"
            score = -80
            allow_long = False
            allow_short = True
            mult = 0.8
            reason = "高階時框空頭防守格局，微觀時框禁止多頭追價，僅允許防禦性對沖或空手。"
        else:
            regime = "CHOP_RANGE"
            score = 10
            allow_long = True
            allow_short = True
            mult = 0.5
            reason = "高階時框均線糾結震盪，微觀交易縮減 50% 倉位規模。"

        return {
            "regime": regime,
            "score": score,
            "allow_long": allow_long,
            "allow_short": allow_short,
            "sizing_multiplier": mult,
            "atr_pct": round(float(atr_pct), 2),
            "reason": reason,
        }

    def filter_micro_signal(self, raw_signal: int, regime_info: dict[str, Any]) -> int:
        """
        依據高階市場狀態濾波微觀交易信號 (1: Long, -1: Short, 0: Flat)
        """
        if raw_signal == 1 and not regime_info.get("allow_long", True):
            return 0
        if raw_signal == -1 and not regime_info.get("allow_short", True):
            return 0
        if regime_info.get("sizing_multiplier", 1.0) <= 0.0:
            return 0
        return raw_signal
