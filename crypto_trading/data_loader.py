"""
CryptoDataLoader: provides real and synthetic OHLCV data under explicit causal policies.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    ccxt = None

from crypto_trading.config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME
from crypto_trading.data_quality import (
    DataQualityError,
    parse_timeframe_delta,
    validate_ohlcv,
)


class DataSource(str, Enum):
    LIVE = "LIVE"
    CACHE = "CACHE"
    DELAYED = "DELAYED"
    SYNTHETIC = "SYNTHETIC"


class DataUnavailableError(Exception):
    """Raised when data cannot be retrieved and synthetic fallback is disallowed."""


@dataclass
class OHLCVResult:
    df: pd.DataFrame
    source: DataSource
    symbol: str
    timeframe: str
    is_synthetic: bool = False


class CryptoDataLoader:
    """
    加密貨幣行情載入與快取模組
    遵循嚴格因果防護 (Causal OHLCV Policy)：
    1. 實盤交易與嚴謹行情預設 allow_synthetic=False，抓取失敗直接拋出 DataUnavailableError，禁止靜默合成。
    2. timestamp 一律為 UTC-aware，並嚴格排除未收盤的當前即時 candle。
    3. 合成數據 seed 改用 hashlib.sha256 具備確定性與統計可重現性。
    """

    def __init__(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = DEFAULT_TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_dir = os.path.join(os.path.dirname(__file__), "data_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_ohlcv_result(
        self,
        limit: int = 500,
        symbol: str | None = None,
        timeframe: str | None = None,
        allow_synthetic: bool = False,
        validate: bool = False,
    ) -> OHLCVResult:
        """獲取 OHLCVResult 包含結構與來源元資料"""
        target_symbol = symbol or self.symbol
        target_timeframe = timeframe or self.timeframe
        delta = parse_timeframe_delta(target_timeframe)
        now_utc = datetime.now(timezone.utc)

        # 1. 嘗試透過 CCXT 抓取
        ccxt_err = None
        try:
            if ccxt is None:
                raise RuntimeError("ccxt is not installed")
            exchange_class = ccxt.binance({"enableRateLimit": True, "timeout": 5000})
            ohlcv = exchange_class.fetch_ohlcv(target_symbol, timeframe=target_timeframe, limit=limit + 1)
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("timestamp", inplace=True)

                # 排除未收盤 candle (若最新 candle close > now_utc，將其剃除)
                if len(df) > 0:
                    last_open = df.index[-1].to_pydatetime()
                    if last_open.tzinfo is None:
                        last_open = last_open.replace(tzinfo=timezone.utc)
                    last_close = last_open + delta.to_pytimedelta()
                    if last_close > now_utc:
                        df = df.iloc[:-1]

                if len(df) > limit:
                    df = df.iloc[-limit:]

                if len(df) > 0:
                    if validate:
                        ok, reason = validate_ohlcv(df, timeframe=target_timeframe, now_utc=now_utc)
                        if not ok:
                            raise DataQualityError(f"Remote OHLCV failed data quality check: {reason}")
                    return OHLCVResult(
                        df=df,
                        source=DataSource.LIVE,
                        symbol=target_symbol,
                        timeframe=target_timeframe,
                        is_synthetic=False,
                    )
        except Exception as e:
            ccxt_err = e

        # 2. 備援或測試模式判定
        if not allow_synthetic:
            err_msg = f"Failed to fetch live OHLCV for {target_symbol} ({target_timeframe}) and allow_synthetic=False."
            if ccxt_err is not None:
                err_msg += f" CCXT Error: {ccxt_err}"
            raise DataUnavailableError(err_msg)

        synth_df = self._generate_synthetic_ohlcv(
            limit=limit,
            symbol=target_symbol,
            timeframe=target_timeframe,
            now_utc=now_utc,
        )
        if validate:
            ok, reason = validate_ohlcv(synth_df, timeframe=target_timeframe, now_utc=now_utc)
            if not ok:
                raise DataQualityError(f"Synthetic OHLCV failed data quality check: {reason}")

        return OHLCVResult(
            df=synth_df,
            source=DataSource.SYNTHETIC,
            symbol=target_symbol,
            timeframe=target_timeframe,
            is_synthetic=True,
        )

    def fetch_ohlcv(
        self,
        limit: int = 500,
        symbol: str | None = None,
        timeframe: str | None = None,
        allow_synthetic: bool = True,
    ) -> pd.DataFrame:
        """
        向後相容接口：預設 allow_synthetic=True (若回測/離線研究未特別要求)
        若需嚴格真實資料請使用 fetch_ohlcv_result 或傳入 allow_synthetic=False
        """
        res = self.fetch_ohlcv_result(
            limit=limit,
            symbol=symbol,
            timeframe=timeframe,
            allow_synthetic=allow_synthetic,
        )
        return res.df

    def _generate_synthetic_ohlcv(
        self,
        limit: int = 500,
        symbol: str | None = None,
        timeframe: str | None = None,
        now_utc: datetime | None = None,
    ) -> pd.DataFrame:
        """使用 hashlib.sha256 確定性 seed 生成幾何布朗運動 (GBM) 與厚尾特徵的量化測試數據"""
        target_symbol = symbol or self.symbol
        target_tf = timeframe or self.timeframe

        seed_str = f"{target_symbol}_{target_tf}"
        seed_hash = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
        seed_int = int(seed_hash[:8], 16)
        rng = np.random.RandomState(seed_int)

        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        elif now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        delta = parse_timeframe_delta(target_tf)
        if "s" in target_tf:
            dt = delta.total_seconds() / (365 * 24 * 3600)
        elif "m" in target_tf:
            dt = (delta.total_seconds() / 60) / (365 * 24 * 60)
        elif "d" in target_tf:
            dt = (delta.total_seconds() / 86400) / 365
        else:
            dt = (delta.total_seconds() / 3600) / (365 * 24)

        # 確保最後一根 candle 完全收盤 (last close <= now_utc)
        # last candle open = now_utc - delta
        last_candle_open = now_utc - delta
        timestamps = [last_candle_open - delta * (limit - 1 - i) for i in range(limit)]

        if "BTC" in target_symbol:
            base_price = 68000.0
            sigma = 0.55
        elif "ETH" in target_symbol:
            base_price = 3500.0
            sigma = 0.65
        elif "SOL" in target_symbol:
            base_price = 180.0
            sigma = 0.85
        else:
            base_price = 100.0
            sigma = 0.70

        mu = 0.05
        prices = [base_price]
        for _ in range(limit - 1):
            shock = rng.normal(0, 1)
            if rng.rand() < 0.05:
                shock *= 2.5
            p = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock)
            prices.append(max(p, 0.01))

        df_list = []
        for i, p in enumerate(prices):
            bar_volatility = p * 0.008 * rng.uniform(0.5, 2.0)
            high = p + abs(rng.normal(0, bar_volatility))
            low = max(p - abs(rng.normal(0, bar_volatility)), 0.001)
            open_p = p + rng.normal(0, bar_volatility * 0.5)
            close_p = p + rng.normal(0, bar_volatility * 0.5)
            high = max(high, open_p, close_p)
            low = min(low, open_p, close_p)
            vol = rng.uniform(50, 500) * (p / 1000)

            df_list.append(
                {
                    "timestamp": timestamps[i],
                    "Open": round(open_p, 2),
                    "High": round(high, 2),
                    "Low": round(low, 2),
                    "Close": round(close_p, 2),
                    "Volume": round(vol, 2),
                }
            )

        df = pd.DataFrame(df_list)
        df.set_index("timestamp", inplace=True)
        return df


if __name__ == "__main__":
    loader = CryptoDataLoader()
    res = loader.fetch_ohlcv_result(limit=10, allow_synthetic=True, validate=True)
    print(f"=== 加密貨幣 K 線獲取測試 (Source: {res.source}) ===")
    print(res.df.tail())
