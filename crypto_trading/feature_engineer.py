"""加密貨幣因果特徵工程，支援多時間週期動態年化。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_trading.instrument_registry import (
    InstrumentRegistry,
    get_instrument_registry,
)


class CryptoFeatureEngineer:
    _PERIODS_PER_YEAR = {
        "15s": 365.0 * 24 * 60 * 60 / 15,
        "1m": 365.0 * 24 * 60,
        "3m": 365.0 * 24 * 60 / 3,
        "5m": 365.0 * 24 * 60 / 5,
        "15m": 365.0 * 24 * 60 / 15,
        "1h": 365.0 * 24,
        "1d": 365.0,
    }

    def __init__(self, forward_horizons: list[int] | None = None) -> None:
        self.forward_horizons = list(forward_horizons or [1, 3, 5, 12])
        self.feature_columns: list[str] = []

    @classmethod
    def get_annualization_factor(cls, timeframe: str) -> float:
        key = str(timeframe).strip().lower()
        if key not in cls._PERIODS_PER_YEAR:
            raise ValueError(f"不支援 timeframe={timeframe!r}；可用值: {', '.join(cls._PERIODS_PER_YEAR)}")
        return float(cls._PERIODS_PER_YEAR[key])

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        lower = {str(c).lower(): c for c in out.columns}
        rename = {}
        for c in ("Open", "High", "Low", "Close", "Volume"):
            if c.lower() in lower:
                rename[lower[c.lower()]] = c
        out = out.rename(columns=rename)
        missing = set(("Open", "High", "Low", "Close", "Volume")) - set(out.columns)
        if missing:
            raise ValueError(f"缺少 OHLCV 欄位: {sorted(missing)}")
        return out.sort_index()

    def compute_features(self, df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
        out = self._normalise(df)
        o, h, l, c, v = [pd.to_numeric(out[x], errors="coerce") for x in ("Open", "High", "Low", "Close", "Volume")]
        annual = np.sqrt(self.get_annualization_factor(timeframe))
        for p in (1, 3, 6, 12, 24):
            out[f"log_ret_{p}"] = np.log(c.div(c.shift(p)))
        for w in (5, 10, 20, 50):
            out[f"bias_ma_{w}"] = c.div(c.rolling(w, min_periods=w).mean()).sub(1.0)
        log_hl = np.log(h.div(l.clip(lower=1e-12)))
        out["parkinson_vol_14"] = np.sqrt(log_hl.pow(2).rolling(14, min_periods=14).mean() / (4 * np.log(2))) * annual
        gk = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * np.log(c.div(o.clip(lower=1e-12))).pow(2)
        out["garman_klass_vol_14"] = np.sqrt(gk.clip(lower=0).rolling(14, min_periods=14).mean()) * annual
        out["hist_vol_24"] = out["log_ret_1"].rolling(24, min_periods=24).std(ddof=1) * annual
        prev = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        out["atr_ratio_14"] = tr.rolling(14, min_periods=14).mean().div(c)
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        out["rsi_14_norm"] = (100 - 100 / (1 + gain.div(loss.replace(0, np.nan))) - 50) / 50
        ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
        out["macd_ratio"] = (ema12 - ema26).div(c)
        ma20 = c.rolling(20, min_periods=20).mean()
        sd20 = c.rolling(20, min_periods=20).std(ddof=1)
        out["bb_width_20"] = (4 * sd20).div(ma20)
        out["bb_pct_b_20"] = (c - (ma20 - 2 * sd20)).div((4 * sd20).replace(0, np.nan))
        vm = v.rolling(24, min_periods=24).mean()
        vs = v.rolling(24, min_periods=24).std(ddof=1)
        out["volume_zscore_24"] = (v - vm).div(vs.replace(0, np.nan))
        typical = (h + l + c) / 3
        rolling_vwap = (typical * v).rolling(24, min_periods=24).sum().div(v.rolling(24, min_periods=24).sum())
        out["vwap_deviation_24"] = c.div(rolling_vwap).sub(1)
        out["hl_location"] = (2 * c - h - l).div((h - l).replace(0, np.nan))

        # 10 期滾動淨主動量比 (Rolling Net Active Volume Ratio):
        # direction * v 滾動 10 期加總除以總成交量 10 期加總
        direction = np.sign(c.diff()).fillna(0)
        net_vol_10 = (direction * v).rolling(10, min_periods=10).sum()
        tot_vol_10 = v.rolling(10, min_periods=10).sum()
        out["obv_slope_10"] = net_vol_10.div(tot_vol_10.replace(0, np.nan))

        for horizon in self.forward_horizons:
            fwd_ret = c.shift(-horizon).div(c).sub(1)
            out[f"target_fwd_ret_{horizon}"] = fwd_ret
            # 確保尾端未來收益為 NaN 時，target_direction 也保持為 pd.NA，避免誤標為 0 造成前瞻洩漏
            # 使用 pd.Series 構建以保留 pd.NA
            dir_series = pd.Series(pd.NA, index=out.index, dtype="Int64")
            valid_mask = ~fwd_ret.isna()
            dir_series[valid_mask] = (fwd_ret[valid_mask] > 0).astype(int)
            out[f"target_direction_{horizon}"] = dir_series

        raw = {"Open", "High", "Low", "Close", "Volume"}
        targets = {col for col in out.columns if col.startswith("target_")}
        self.feature_columns = [x for x in out.columns if x not in raw | targets]
        return out.replace([np.inf, -np.inf], np.nan)

    def compute_multi_asset_cross_features(
        self,
        asset_dfs: dict,
        max_gap: int = 3,
        min_coverage: float = 0.5,
        registry: InstrumentRegistry | None = None,
    ) -> pd.DataFrame:
        """
        計算多幣種截面相關性、動量離散度與相對強弱特徵 (Cross-sectional Alpha)
        使用 InstrumentRegistry 識別資產標的；
        移除 .bfill()，採用 .ffill(limit=max_gap) 與 stale 排除及最低覆蓋率驗證。
        """
        reg = registry or get_instrument_registry()
        closes = {}
        canonical_map = {}

        for sym, df in asset_dfs.items():
            norm_df = self._normalise(df)
            base_key = reg.resolve_base_asset(sym)
            canonical_map[base_key] = sym
            closes[base_key] = norm_df["Close"]

        # 構建跨資產收盤價 DataFrame，嚴格禁止 .bfill()，只允許向前傳播且限制 max_gap 根 candle
        raw_close_df = pd.DataFrame(closes)
        close_df = raw_close_df.ffill(limit=max_gap)

        # 覆蓋率過濾：每根 bar 必須滿足最低資產覆蓋率，否則標記為 NaN
        total_assets = len(asset_dfs)
        valid_counts = close_df.notna().sum(axis=1)
        insufficient_coverage = valid_counts < (total_assets * min_coverage)
        close_df[insufficient_coverage] = np.nan

        ret_df = close_df.pct_change(fill_method=None)

        cross_df = pd.DataFrame(index=close_df.index)
        cross_df["cross_momentum_spread"] = ret_df.max(axis=1) - ret_df.min(axis=1)
        cross_df["cross_volatility_mean"] = ret_df.std(axis=1)

        # 使用 InstrumentRegistry 解析 canonical 資產名稱，杜絕 "BTC" in close_df 猜測
        btc_inst = reg.get("BTC/USDT")
        eth_inst = reg.get("ETH/USDT")
        sol_inst = reg.get("SOL/USDT")

        btc_key = btc_inst.base_asset if btc_inst else "BTC"
        eth_key = eth_inst.base_asset if eth_inst else "ETH"
        sol_key = sol_inst.base_asset if sol_inst else "SOL"

        if btc_key in close_df and eth_key in close_df:
            cross_df["ETH_corr_BTC_24"] = ret_df[eth_key].rolling(24, min_periods=5).corr(ret_df[btc_key])
            cross_df["ETH_corr_BTC_24_valid"] = cross_df["ETH_corr_BTC_24"].notna().astype(int)

        if sol_key in close_df and eth_key in close_df:
            cross_df["SOL_to_ETH_ratio"] = close_df[sol_key] / close_df[eth_key]

        if len(asset_dfs) >= 2:
            cols = list(close_df.columns)
            cross_df["corr_top2"] = ret_df[cols[0]].rolling(20, min_periods=5).corr(ret_df[cols[1]])

        return cross_df
