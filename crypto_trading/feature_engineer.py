"""
Crypto Feature Engineering for Machine Learning & Deep Learning.
Constructs stationary, normalized features from OHLCV data.
Features include:
1. Normalized Log Returns & Multi-period Momentum
2. Technical Oscillators: RSI, MACD, Bollinger Band width & %B, ATR ratio
3. Order Flow & Volume features: Volume z-score, On-Balance Volume (OBV) slope, VWAP deviation
4. Volatility Estimators: Parkinson & Garman-Klass volatility estimators
5. Directional classification targets and multi-horizon forward returns
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

class CryptoFeatureEngineer:
    def __init__(self, forward_horizons: List[int] = [1, 3, 5, 12]):
        self.forward_horizons = forward_horizons
        self.feature_columns: List[str] = []

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input DataFrame must have: Open, High, Low, Close, Volume.
        Outputs a clean DataFrame with stationary feature columns and target variables.
        """
        df = df.copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        close = df['Close']
        high = df['High']
        low = df['Low']
        open_p = df['Open']
        vol = df['Volume']

        # 1. Log Returns & Momentum
        df['log_ret_1'] = np.log(close / close.shift(1))
        df['log_ret_3'] = np.log(close / close.shift(3))
        df['log_ret_6'] = np.log(close / close.shift(6))
        df['log_ret_12'] = np.log(close / close.shift(12))
        df['log_ret_24'] = np.log(close / close.shift(24))

        # 2. Moving Average Ratios (Stationary relative differences)
        for w in [5, 10, 20, 50]:
            ma = close.rolling(w).mean()
            df[f'dist_ma_{w}'] = (close - ma) / (ma + 1e-9)

        # 3. Volatility Features
        # Rolling realized volatility (annualized assuming 1h bars)
        df['hist_vol_12'] = df['log_ret_1'].rolling(12).std() * np.sqrt(8760)
        df['hist_vol_24'] = df['log_ret_1'].rolling(24).std() * np.sqrt(8760)

        # Parkinson Volatility (Extreme value estimator: uses High and Low)
        hl_ratio = np.log(high / low)
        df['parkinson_vol_14'] = np.sqrt((hl_ratio ** 2).rolling(14).mean() / (4 * np.log(2))) * np.sqrt(8760)

        # Garman-Klass Volatility (Uses OHLC)
        co_ratio = np.log(close / open_p)
        gk = 0.5 * (hl_ratio ** 2) - (2 * np.log(2) - 1) * (co_ratio ** 2)
        df['garman_klass_vol_14'] = np.sqrt(np.maximum(gk.rolling(14).mean(), 0)) * np.sqrt(8760)

        # 4. Bollinger Bands (%B and Bandwidth)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        df['bb_pct_b'] = (close - lower) / (upper - lower + 1e-9)
        df['bb_bandwidth'] = (upper - lower) / (ma20 + 1e-9)

        # 5. Relative Strength Index (RSI 14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14_norm'] = (df['rsi_14'] - 50.0) / 50.0  # Normalized to [-1, 1]

        # 6. MACD Normalized
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_diff_norm'] = (macd - macd_signal) / (close + 1e-9)

        # 7. Volume & Microstructure Features
        vol_mean24 = vol.rolling(24).mean()
        vol_std24 = vol.rolling(24).std()
        df['vol_zscore_24'] = (vol - vol_mean24) / (vol_std24 + 1e-9)

        # Intraday buy/sell pressure proxy (Close - Low) / (High - Low)
        df['hl_close_location'] = (close - low) / (high - low + 1e-9) * 2 - 1.0  # [-1, 1]

        # VWAP deviation (24 period)
        typical_price = (high + low + close) / 3.0
        vwap_24 = (typical_price * vol).rolling(24).sum() / (vol.rolling(24).sum() + 1e-9)
        df['dist_vwap_24'] = (close - vwap_24) / (vwap_24 + 1e-9)

        # 8. Targets: Multi-horizon Forward Returns & Binary/Ternary Direction
        # Primary target: 3-period forward return
        df['target_fwd_ret_1'] = close.shift(-1) / close - 1.0
        df['target_fwd_ret_3'] = close.shift(-3) / close - 1.0
        df['target_fwd_ret_6'] = close.shift(-6) / close - 1.0

        # Classification targets:
        # Binary: 1 if fwd_ret_3 > 0 else 0
        df['target_binary'] = (df['target_fwd_ret_3'] > 0).astype(int)
        
        # Ternary target with fee barrier (e.g. 0.2% return hurdle)
        hurdle = 0.002
        conditions = [
            df['target_fwd_ret_3'] > hurdle,
            df['target_fwd_ret_3'] < -hurdle
        ]
        choices = [1, -1] # 1: Long, -1: Short, 0: Neutral
        df['target_ternary'] = np.select(conditions, choices, default=0)

        # Track feature column names
        exclude_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'target_fwd_ret_1', 
                        'target_fwd_ret_3', 'target_fwd_ret_6', 'target_binary', 'target_ternary'}
        self.feature_columns = [c for c in df.columns if c not in exclude_cols]

        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Cleans NaNs and extracts (X, y, feature_names) for 3-step forward return prediction.
        """
        processed = self.compute_features(df)
        clean = processed.dropna().copy()
        X = clean[self.feature_columns]
        y = clean['target_fwd_ret_3']
        return X, y, self.feature_columns
