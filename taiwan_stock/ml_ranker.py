"""台股多因子 ML 排序器，採全域日期排序與 5 交易日 embargo。"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None


class TaiwanStockMLRanker:
    TARGET_COLUMN = "target_fwd_5d"
    EMBARGO_TRADING_DAYS = 5

    def __init__(self, model_type: str = "lightgbm") -> None:
        self.model_type = model_type.lower()
        self.feature_cols: List[str] = []
        self.model: Any = None
        self.last_split_info: Dict[str, Any] = {}

    @staticmethod
    def _normalise_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        aliases = {str(c).lower(): c for c in out.columns}
        rename = {}
        for canonical in ("Open", "High", "Low", "Close", "Volume"):
            current = aliases.get(canonical.lower())
            if current is not None and current != canonical:
                rename[current] = canonical
        return out.rename(columns=rename)

    def extract_features(self, df: pd.DataFrame, forward_period: int = 5) -> pd.DataFrame:
        """只用 t 與過去資料建構特徵；唯一前瞻欄位為訓練目標。"""
        out = self._normalise_ohlcv_columns(df).copy()
        required = {"High", "Low", "Close", "Volume"}
        missing = required.difference(out.columns)
        if missing:
            raise ValueError(f"缺少必要 OHLCV 欄位: {sorted(missing)}")
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="raise")
        out = out.sort_index()
        c = pd.to_numeric(out["Close"], errors="coerce")
        h = pd.to_numeric(out["High"], errors="coerce")
        lo = pd.to_numeric(out["Low"], errors="coerce")
        v = pd.to_numeric(out["Volume"], errors="coerce")

        for p in (5, 10, 20, 60):
            out[f"mom_{p}d"] = c.pct_change(p, fill_method=None)
        for w in (5, 20, 60):
            ma = c.rolling(w, min_periods=w).mean()
            out[f"bias_ma{w}"] = c.div(ma).sub(1.0)
            out[f"ma{w}_slope_5d"] = ma.pct_change(5, fill_method=None)

        daily_ret = c.pct_change(fill_method=None)
        out["hist_vol_20"] = daily_ret.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252.0)
        prev_close = c.shift(1)
        tr = pd.concat([(h-lo).abs(), (h-prev_close).abs(), (lo-prev_close).abs()], axis=1).max(axis=1)
        out["atr_ratio_14"] = tr.rolling(14, min_periods=14).mean().div(c)
        out["vol_surge"] = v.div(v.rolling(5, min_periods=5).mean())

        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        out["rsi_14"] = 100.0 - 100.0 / (1.0 + gain.div(loss.replace(0.0, np.nan)))
        low9 = lo.rolling(9, min_periods=9).min()
        high9 = h.rolling(9, min_periods=9).max()
        rsv = 100.0 * (c-low9).div((high9-low9).replace(0.0, np.nan))
        out["k_9"] = rsv.ewm(alpha=1/3, adjust=False).mean()
        out["d_9"] = out["k_9"].ewm(alpha=1/3, adjust=False).mean()
        out[self.TARGET_COLUMN] = c.shift(-forward_period).div(c).sub(1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def prepare_dataset(self, stock_data_map: Mapping[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
        panels: List[pd.DataFrame] = []
        for symbol, raw in stock_data_map.items():
            if raw is None or raw.empty:
                continue
            panel = self.extract_features(raw)
            panel["symbol"] = str(symbol)
            panels.append(panel)
        if not panels:
            raise ValueError("stock_data_map 沒有可用資料")

        combined = pd.concat(panels, axis=0, sort=False)
        # 關鍵：跨股票合併後，依日期索引進行全域穩定排序。
        combined = combined.sort_index(kind="mergesort")
        excluded = {"Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "symbol", self.TARGET_COLUMN}
        self.feature_cols = [c for c in combined.columns if c not in excluded and pd.api.types.is_numeric_dtype(combined[c])]
        clean = combined.dropna(subset=self.feature_cols + [self.TARGET_COLUMN]).copy()
        clean = clean.sort_index(kind="mergesort")
        if clean.empty:
            raise ValueError("特徵暖機與目標清理後沒有可訓練資料")
        return clean, list(self.feature_cols)

    def split_dataset_by_date(self, dataset: pd.DataFrame, test_fraction: float = 0.20, embargo_days: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction 必須介於 0 與 1")
        data = dataset.sort_index(kind="mergesort")
        dates = pd.DatetimeIndex(data.index).normalize().unique().sort_values()
        if len(dates) < embargo_days + 3:
            raise ValueError(f"至少需要 {embargo_days + 3} 個不同交易日才能切分")
        test_pos = max(embargo_days + 1, int(np.floor(len(dates) * (1.0-test_fraction))))
        test_pos = min(test_pos, len(dates)-1)
        test_start = dates[test_pos]
        embargo_dates = dates[test_pos-embargo_days:test_pos]
        train_end_pos = test_pos-embargo_days-1
        if train_end_pos < 0:
            raise ValueError("資料不足以建立 embargo")
        train_end = dates[train_end_pos]
        normalised = pd.DatetimeIndex(data.index).normalize()
        train = data.loc[normalised <= train_end].copy()
        test = data.loc[normalised >= test_start].copy()
        if train.empty or test.empty or not train.index.max() < test.index.min():
            raise AssertionError("日期切分失敗：Train 必須嚴格早於 Test")
        self.last_split_info = {
            "train_max_date": pd.Timestamp(train.index.max()),
            "test_min_date": pd.Timestamp(test.index.min()),
            "embargo_dates": [pd.Timestamp(d) for d in embargo_dates],
            "embargo_trading_days": len(embargo_dates),
            "train_rows": len(train),
            "test_rows": len(test),
        }
        return train, test

    def _build_model(self) -> Any:
        if self.model_type == "lightgbm" and lgb is not None:
            return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
        if self.model_type == "xgboost" and xgb is not None:
            return xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=1)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=5, random_state=42)

    def train_model(self, dataset: pd.DataFrame, test_fraction: float = 0.20, embargo_days: int = EMBARGO_TRADING_DAYS) -> Dict[str, Any]:
        if not self.feature_cols:
            excluded = {"Open", "High", "Low", "Close", "Volume", "symbol", self.TARGET_COLUMN}
            self.feature_cols = [c for c in dataset.columns if c not in excluded and pd.api.types.is_numeric_dtype(dataset[c])]
        train, test = self.split_dataset_by_date(dataset, test_fraction, embargo_days)
        self.model = self._build_model()
        self.model.fit(train[self.feature_cols], train[self.TARGET_COLUMN])
        pred = np.asarray(self.model.predict(test[self.feature_cols]), dtype=float)
        actual = test[self.TARGET_COLUMN].to_numpy(dtype=float)
        rank_ic = pd.Series(pred).corr(pd.Series(actual), method="spearman")
        rmse = float(np.sqrt(np.mean((pred-actual)**2)))
        return {"rank_ic": 0.0 if pd.isna(rank_ic) else float(rank_ic), "rmse": rmse, **self.last_split_info}

    def predict_rankings(self, stock_data_map: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("請先呼叫 train_model()")
        rows = []
        for symbol, raw in stock_data_map.items():
            feat = self.extract_features(raw)
            valid = feat.dropna(subset=self.feature_cols)
            if valid.empty:
                continue
            latest = valid.iloc[-1]
            pred = float(self.model.predict(pd.DataFrame([latest[self.feature_cols].to_dict()], columns=self.feature_cols))[0])
            rows.append({"symbol": symbol, "current_price": round(float(latest["Close"]), 4), "predicted_fwd_5d_ret_pct": pred*100.0, "bias_ma20": float(latest.get("bias_ma20", np.nan))*100.0, "rsi_14": float(latest.get("rsi_14", np.nan))})
        result = pd.DataFrame(rows).sort_values("predicted_fwd_5d_ret_pct", ascending=False).reset_index(drop=True)
        if not result.empty:
            result["rank"] = np.arange(1, len(result)+1)
        return result

    def save_model(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("沒有可儲存的模型")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols, "model_type": self.model_type}, path)

    @classmethod
    def load_model(cls, path: str) -> "TaiwanStockMLRanker":
        payload = joblib.load(path)
        obj = cls(payload.get("model_type", "lightgbm"))
        obj.model = payload["model"]
        obj.feature_cols = list(payload["feature_cols"])
        return obj
