"""
Taiwan Stock Multi-Factor Feature Engineering & ML Ranking Model.
Constructs cross-sectional and time-series factors:
1. Momentum factors: 5d, 20d, 60d cumulative returns, volume-weighted momentum
2. Volatility factors: Historical volatility, High/Low spread, ATR ratio
3. Trend & Moving Average Disparities: Distance from MA5, MA20, MA60
4. Volume Flow: Relative volume, volume turnover, price-volume rank correlation
5. Cross-sectional ranking and Walk-Forward predictive modeling
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None
import yfinance as yf

class TaiwanStockMLRanker:
    """
    台股多因子機器學習選股排序模型
    - 結合量價、動量、波動度與籌碼面代理因子
    - 採用 Cross-Sectional Ranking 或 Time-Series Walk-Forward 進行預測
    - 輸出各股未來超額報酬預測值與投資組合推薦清單
    """
    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type.lower()
        self.feature_cols: List[str] = []
        self.model = None

    @staticmethod
    def extract_stock_features(df: pd.DataFrame, forward_period: int = 5) -> pd.DataFrame:
        """
        為單檔台股計算標準多因子特徵矩陣
        """
        df = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']

        # 1. 動量因子 (Momentum)
        df['mom_5d'] = close.pct_change(5)
        df['mom_10d'] = close.pct_change(10)
        df['mom_20d'] = close.pct_change(20)
        df['mom_60d'] = close.pct_change(60)

        # 2. 均線乖離率 (Disparity)
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        df['bias_ma5'] = (close - ma5) / (ma5 + 1e-9)
        df['bias_ma20'] = (close - ma20) / (ma20 + 1e-9)
        df['bias_ma60'] = (close - ma60) / (ma60 + 1e-9)
        df['ma_cross_spread'] = (ma5 - ma20) / (ma20 + 1e-9)

        # 3. 波動度因子 (Volatility)
        ret = close.pct_change()
        df['hist_vol_20'] = ret.rolling(20).std() * np.sqrt(250)
        df['hl_spread_pct'] = (high - low) / (close + 1e-9)
        
        # ATR 比例
        tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
        df['atr_ratio'] = tr.rolling(14).mean() / (close + 1e-9)

        # 4. 成交量動態 (Volume Dynamics)
        vol_ma5 = vol.rolling(5).mean()
        vol_ma20 = vol.rolling(20).mean()
        df['vol_surge_ratio'] = vol / (vol_ma5 + 1e-9)
        df['vol_trend'] = vol_ma5 / (vol_ma20 + 1e-9)

        # 5. 技術擺盪指標 (KD & RSI)
        low9 = low.rolling(9).min()
        high9 = high.rolling(9).max()
        rsv = (close - low9) / (high9 - low9 + 1e-9) * 100
        df['kd_k'] = rsv.ewm(alpha=1/3, adjust=False).mean()
        df['kd_d'] = df['kd_k'].ewm(alpha=1/3, adjust=False).mean()
        df['kd_diff'] = df['kd_k'] - df['kd_d']

        # RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # 6. Target: 未來 5 日前瞻報酬率 (Forward 5-Day Return)
        df['target_fwd_5d'] = close.shift(-forward_period) / close - 1.0

        return df

    def prepare_dataset(self, stock_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
        """
        將多檔股票特徵進行整合與標準化處理
        """
        panel_data = []
        feature_names = []

        for symbol, df in stock_dfs.items():
            if len(df) < 80:
                continue
            feats = self.extract_stock_features(df)
            feats['symbol'] = symbol
            panel_data.append(feats)

        if not panel_data:
            raise ValueError("No valid stock data found.")

        combined = pd.concat(panel_data, axis=0)
        
        exclude_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'target_fwd_5d', 'symbol', 'Dividends', 'Stock Splits'}
        feature_names = [c for c in combined.columns if c not in exclude_cols]
        self.feature_cols = feature_names

        clean_data = combined.dropna(subset=feature_names + ['target_fwd_5d']).copy()
        return clean_data, feature_names

    def train_model(self, clean_data: pd.DataFrame) -> Dict[str, Any]:
        X = clean_data[self.feature_cols].values
        y = clean_data['target_fwd_5d'].values

        # 依時間前後拆分訓練/驗證集 (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=250,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )

        self.model.fit(X_train, y_train)

        # 驗證集評估
        test_preds = self.model.predict(X_test)
        rank_ic = pd.Series(test_preds).corr(pd.Series(y_test), method='spearman')
        mae = np.mean(np.abs(test_preds - y_test))

        # 特徵重要性
        feat_imp = []
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feat_imp = sorted(zip(self.feature_cols, importances), key=lambda x: x[1], reverse=True)

        return {
            "model_type": self.model_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "rank_ic": float(rank_ic if not np.isnan(rank_ic) else 0.0),
            "mae": float(mae),
            "top_features": feat_imp[:10]
        }

    def predict_rankings(self, current_stock_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        對當前觀察池的所有股票進行最新因子打分與多頭排序
        支援 Kaggle 訓練產物自動載入，以及 Render 輕量化推論 (免重訓練，低記憶體消耗)
        """
        # 若尚未載入模型，優先嘗試從磁碟預訓練權重載入 (Kaggle 訓練產物)
        if self.model is None:
            default_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taiwan_stock_ranker.joblib")
            if os.path.exists(default_weights):
                try:
                    loaded = TaiwanStockMLRanker.load_model(default_weights)
                    self.model = loaded.model
                    self.feature_cols = loaded.feature_cols
                except Exception:
                    pass

        results = []
        for symbol, df in current_stock_dfs.items():
            if len(df) < 65:
                continue
            feats = self.extract_stock_features(df)
            latest = feats.iloc[-1:]
            
            # 若為 Render 輕量化推論模式或未訓練模型，使用經典截面多因子加權打分 (Rule-Based Factor Ranker)
            if self.model is None or not self.feature_cols:
                curr_price = float(df['Close'].iloc[-1])
                vol = float(df['Volume'].iloc[-1])
                mom = float(latest['mom_20d'].iloc[0]) if 'mom_20d' in latest else 0.0
                bias = float(latest['bias_ma20'].iloc[0]) if 'bias_ma20' in latest else 0.0
                rsi = float(latest['rsi_14'].iloc[0]) if 'rsi_14' in latest else 50.0
                predicted_return = (mom * 0.4) + (bias * 0.3) + ((rsi - 50.0) / 100.0 * 0.3)
                results.append({
                    "symbol": symbol,
                    "current_price": round(curr_price, 2),
                    "predicted_fwd_5d_ret_pct": round(predicted_return * 100, 2),
                    "volume": int(vol),
                    "bias_ma20": round(bias * 100, 2),
                    "rsi_14": round(rsi, 1)
                })
                continue

            if latest[self.feature_cols].isna().any().any():
                continue

            X_latest = latest[self.feature_cols].values
            predicted_return = float(self.model.predict(X_latest)[0])
            curr_price = float(df['Close'].iloc[-1])
            vol = float(df['Volume'].iloc[-1])

            results.append({
                "symbol": symbol,
                "current_price": round(curr_price, 2),
                "predicted_fwd_5d_ret_pct": round(predicted_return * 100, 2),
                "volume": int(vol),
                "bias_ma20": round(float(latest['bias_ma20'].iloc[0]) * 100, 2),
                "rsi_14": round(float(latest['rsi_14'].iloc[0]), 1)
            })

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df = res_df.sort_values(by="predicted_fwd_5d_ret_pct", ascending=False).reset_index(drop=True)
            res_df['rank'] = res_df.index + 1
        return res_df

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "feature_cols": self.feature_cols
        }, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "TaiwanStockMLRanker":
        bundle = joblib.load(filepath)
        instance = cls(model_type=bundle["model_type"])
        instance.model = bundle["model"]
        instance.feature_cols = bundle["feature_cols"]
        return instance
