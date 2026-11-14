"""加密貨幣因果特徵工程，支援多時間週期動態年化。"""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

class CryptoFeatureEngineer:
    _PERIODS_PER_YEAR = {"15s": 365.0*24*60*60/15, "1m": 365.0*24*60, "3m": 365.0*24*60/3, "5m": 365.0*24*60/5, "15m": 365.0*24*60/15, "1h": 365.0*24, "1d": 365.0}

    def __init__(self, forward_horizons: List[int] | None = None) -> None:
        self.forward_horizons = list(forward_horizons or [1,3,5,12])
        self.feature_columns: List[str] = []

    @classmethod
    def get_annualization_factor(cls, timeframe: str) -> float:
        key = str(timeframe).strip().lower()
        if key not in cls._PERIODS_PER_YEAR:
            raise ValueError(f"不支援 timeframe={timeframe!r}；可用值: {', '.join(cls._PERIODS_PER_YEAR)}")
        return float(cls._PERIODS_PER_YEAR[key])

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        out=df.copy(); lower={str(c).lower():c for c in out.columns}; rename={}
        for c in ("Open","High","Low","Close","Volume"):
            if c.lower() in lower: rename[lower[c.lower()]]=c
        out=out.rename(columns=rename)
        missing=set(("Open","High","Low","Close","Volume"))-set(out.columns)
        if missing: raise ValueError(f"缺少 OHLCV 欄位: {sorted(missing)}")
        return out.sort_index()

    def compute_features(self, df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
        out=self._normalise(df)
        o,h,l,c,v=[pd.to_numeric(out[x],errors="coerce") for x in ("Open","High","Low","Close","Volume")]
        annual=np.sqrt(self.get_annualization_factor(timeframe))
        for p in (1,3,6,12,24): out[f"log_ret_{p}"]=np.log(c.div(c.shift(p)))
        for w in (5,10,20,50): out[f"bias_ma_{w}"]=c.div(c.rolling(w,min_periods=w).mean()).sub(1.0)
        log_hl=np.log(h.div(l.clip(lower=1e-12)))
        out["parkinson_vol_14"]=np.sqrt(log_hl.pow(2).rolling(14,min_periods=14).mean()/(4*np.log(2)))*annual
        gk=0.5*log_hl.pow(2)-(2*np.log(2)-1)*np.log(c.div(o.clip(lower=1e-12))).pow(2)
        out["garman_klass_vol_14"]=np.sqrt(gk.clip(lower=0).rolling(14,min_periods=14).mean())*annual
        out["hist_vol_24"]=out["log_ret_1"].rolling(24,min_periods=24).std(ddof=1)*annual
        prev=c.shift(1); tr=pd.concat([(h-l).abs(),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
        out["atr_ratio_14"]=tr.rolling(14,min_periods=14).mean().div(c)
        delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
        out["rsi_14_norm"]=(100-100/(1+gain.div(loss.replace(0,np.nan)))-50)/50
        ema12=c.ewm(span=12,adjust=False,min_periods=12).mean(); ema26=c.ewm(span=26,adjust=False,min_periods=26).mean()
        out["macd_ratio"]=(ema12-ema26).div(c)
        ma20=c.rolling(20,min_periods=20).mean(); sd20=c.rolling(20,min_periods=20).std(ddof=1)
        out["bb_width_20"]=(4*sd20).div(ma20); out["bb_pct_b_20"]=(c-(ma20-2*sd20)).div((4*sd20).replace(0,np.nan))
        vm=v.rolling(24,min_periods=24).mean(); vs=v.rolling(24,min_periods=24).std(ddof=1)
        out["volume_zscore_24"]=(v-vm).div(vs.replace(0,np.nan))
        typical=(h+l+c)/3; rolling_vwap=(typical*v).rolling(24,min_periods=24).sum().div(v.rolling(24,min_periods=24).sum())
        out["vwap_deviation_24"]=c.div(rolling_vwap).sub(1); out["hl_location"]=(2*c-h-l).div((h-l).replace(0,np.nan))
        direction=np.sign(c.diff()).fillna(0); obv=(direction*v).cumsum(); out["obv_slope_10"]=obv.diff(10).div(v.rolling(10,min_periods=10).mean()*10)
        for horizon in self.forward_horizons:
            out[f"target_fwd_ret_{horizon}"]=c.shift(-horizon).div(c).sub(1)
            out[f"target_direction_{horizon}"]=(out[f"target_fwd_ret_{horizon}"]>0).astype("Int64")
        raw={"Open","High","Low","Close","Volume"}; targets={c for c in out.columns if c.startswith("target_")}
        self.feature_columns=[x for x in out.columns if x not in raw|targets]
        return out.replace([np.inf,-np.inf],np.nan)

    def compute_multi_asset_cross_features(self, asset_dfs: dict) -> pd.DataFrame:
        """計算多幣種截面相關性、動量離散度與相對強弱特徵 (Cross-sectional Alpha)"""
        closes = {}
        for sym, df in asset_dfs.items():
            norm_df = self._normalise(df)
            closes[sym] = norm_df["Close"]
        close_df = pd.DataFrame(closes).ffill().bfill()
        ret_df = close_df.pct_change()
        
        cross_df = pd.DataFrame(index=close_df.index)
        cross_df["cross_momentum_spread"] = ret_df.max(axis=1) - ret_df.min(axis=1)
        cross_df["cross_volatility_mean"] = ret_df.std(axis=1)
        
        # 兼容跨幣種相關性與相對比率特徵
        if "BTC" in close_df and "ETH" in close_df:
            cross_df["ETH_corr_BTC_24"] = ret_df["ETH"].rolling(24, min_periods=5).corr(ret_df["BTC"]).fillna(0.0)
        if "SOL" in close_df and "ETH" in close_df:
            cross_df["SOL_to_ETH_ratio"] = close_df["SOL"] / close_df["ETH"]
        if len(asset_dfs) >= 2:
            cols = list(asset_dfs.keys())
            cross_df["corr_top2"] = ret_df[cols[0]].rolling(20, min_periods=5).corr(ret_df[cols[1]]).fillna(0.0)
        return cross_df
