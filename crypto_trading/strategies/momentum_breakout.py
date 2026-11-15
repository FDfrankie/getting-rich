
import numpy as np
import pandas as pd

from crypto_trading.config import STRATEGY_PARAMS


class BaseStrategy:
    """策略基底介面"""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class MomentumBreakoutStrategy(BaseStrategy):
    """
    動量與波動率通道突破策略 (Bollinger & Donchian Breakout)
    - 結合布林通道 (Bollinger Bands) 帶寬壓縮與爆發突破
    - 結合 Donchian 通道 (最高點/最低點) 確認趨勢強度
    - 信號定義：1 為做多, -1 為做空, 0 為平倉/觀望
    """

    def __init__(
        self,
        window: int = STRATEGY_PARAMS["bollinger_window"],
        std_multiplier: float = STRATEGY_PARAMS["bollinger_std"],
    ):
        self.window = window
        self.std_multiplier = std_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # 計算布林帶
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        df["BB_UPPER"] = ma + (std * self.std_multiplier)
        df["BB_LOWER"] = ma - (std * self.std_multiplier)
        df["BB_MID"] = ma

        # 計算 Donchian 20 週期通道
        df["DONCHIAN_HIGH"] = high.rolling(self.window).max()
        df["DONCHIAN_LOW"] = low.rolling(self.window).min()

        # 計算 ATR 作為波動度濾網
        tr = np.maximum(
            high - low,
            np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()),
        )
        df["ATR"] = tr.rolling(14).mean()

        # 生成交易信號 (Signal)
        # 突破布林上軌且創近 20 根 K 線最高價 -> 做多
        long_condition = (close > df["BB_UPPER"].shift(1)) & (
            close >= df["DONCHIAN_HIGH"].shift(1)
        )
        # 跌破布林下軌且創近 20 根 K 線最低價 -> 做空
        short_condition = (close < df["BB_LOWER"].shift(1)) & (
            close <= df["DONCHIAN_LOW"].shift(1)
        )
        # 回歸均線 (BB_MID) -> 平倉信號
        exit_long = close < df["BB_MID"]
        exit_short = close > df["BB_MID"]

        signals = np.zeros(len(df))
        position = 0  # 1: Long, -1: Short, 0: Flat

        for i in range(self.window, len(df)):
            if position == 0:
                if long_condition.iloc[i]:
                    position = 1
                elif short_condition.iloc[i]:
                    position = -1
            elif position == 1:
                if exit_long.iloc[i] or short_condition.iloc[i]:
                    position = -1 if short_condition.iloc[i] else 0
            elif position == -1:
                if exit_short.iloc[i] or long_condition.iloc[i]:
                    position = 1 if long_condition.iloc[i] else 0
            signals[i] = position

        df["signal"] = signals
        return df


if __name__ == "__main__":
    from crypto_trading.data_loader import CryptoDataLoader

    loader = CryptoDataLoader()
    data = loader.fetch_ohlcv(limit=200)
    strat = MomentumBreakoutStrategy()
    res = strat.generate_signals(data)
    print("=== 動量突破策略信號統計 ===")
    print(res["signal"].value_counts())
