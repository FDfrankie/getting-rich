import pandas as pd
import numpy as np
from crypto_trading.config import STRATEGY_PARAMS

class StatisticalArbitrageStrategy:
    """
    統計套利與均值回歸策略 (Z-Score Mean Reversion)
    - 基於時間序列統計學檢驗 (Statistically Stationarity & Z-score)
    - 價格偏離滾動均值達極端統計顯著性時 (Z-score > 2.0 或 < -2.0) 進行反向均值回歸交易
    - 價格收斂至中樞帶 (Z-score 接近 0) 時獲利平倉
    """
    def __init__(self, window: int = 30, entry_z: float = STRATEGY_PARAMS['zscore_entry_threshold'], exit_z: float = STRATEGY_PARAMS['zscore_exit_threshold']):
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['Close']

        # 計算滾動均值與標準差
        rolling_mean = close.rolling(self.window).mean()
        rolling_std = close.rolling(self.window).std()

        # 標準化 Z-Score: (Price - Mean) / Std
        df['ZSCORE'] = (close - rolling_mean) / (rolling_std + 1e-9)

        signals = np.zeros(len(df))
        position = 0

        for i in range(self.window, len(df)):
            z = df['ZSCORE'].iloc[i]
            if position == 0:
                if z < -self.entry_z:
                    # 價格極度超跌 (偏離超過 -2 個標準差)，做多博取均值回歸
                    position = 1
                elif z > self.entry_z:
                    # 價格極度超漲 (偏離超過 +2 個標準差)，做空博取回落
                    position = -1
            elif position == 1:
                # 做多部位當 Z-Score 回歸到 -exit_z 以上時平倉獲利
                if z >= -self.exit_z:
                    position = 0
            elif position == -1:
                # 做空部位當 Z-Score 回歸到 +exit_z 以下時平倉獲利
                if z <= self.exit_z:
                    position = 0
            signals[i] = position

        df['signal'] = signals
        return df

if __name__ == "__main__":
    from crypto_trading.data_loader import CryptoDataLoader
    loader = CryptoDataLoader()
    data = loader.fetch_ohlcv(limit=200)
    strat = StatisticalArbitrageStrategy()
    res = strat.generate_signals(data)
    print("=== 統計套利 Z-Score 策略信號統計 ===")
    print(res['signal'].value_counts())
